#include "VideoFrameExtractor.h"
#include <stdexcept>
#include <filesystem>
#include <algorithm>

#include <list>
#include <unordered_map>
//#include <iostream>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
}

namespace vfe {

struct VideoFrameExtractor::LRUCache {
    static constexpr size_t MAX_BYTES = 1ULL * 1024 * 1024 * 1024; // max 1GB cache

    std::list<int> access_order;
    std::unordered_map<int, std::shared_ptr<FrameData>> data_map;
    size_t current_bytes = 0;

    void insert(int idx, std::shared_ptr<FrameData> frame) {
        size_t frame_size = frame->pixels->size();

        while (!access_order.empty() && current_bytes + frame_size > MAX_BYTES) {
            int lru_idx = access_order.front();
            access_order.pop_front();
            auto it = data_map.find(lru_idx);
            if (it != data_map.end()) {
                current_bytes -= it->second->pixels->size();
                data_map.erase(it);
            }
        }

        access_order.push_back(idx);
        data_map[idx] = frame;
        current_bytes += frame_size;
    }

    std::shared_ptr<const FrameData> get(int idx) {
        auto it = data_map.find(idx);
        if (it != data_map.end()) {
            access_order.remove(idx);
            access_order.push_back(idx);
            return it->second;
        }
        return nullptr;
    }
};

struct VideoFrameExtractor::Impl {
    AVFormatContext* format_ctx = nullptr;
    AVCodecContext* codec_ctx = nullptr;
    const AVCodec* codec = nullptr;
    int video_stream_idx = -1;
    SwsContext* sws_ctx = nullptr;
    mutable LRUCache cache;
    std::string video_path;
    std::string filename;
    std::string container_format;
    int width = 0;
    int height = 0;
    int64_t frame_count = 0;
    double duration_sec = 0.0;
    bool is_open = false;

    ~Impl() {
        if (sws_ctx) {
            sws_freeContext(sws_ctx);
        }
        if (codec_ctx) {
            avcodec_free_context(&codec_ctx);
        }
        if (format_ctx) {
            avformat_close_input(&format_ctx);
        }
    }
};

VideoFrameExtractor::VideoFrameExtractor(const std::string& videoPath)
    : pImpl(std::make_unique<Impl>()) {

    pImpl->video_path = videoPath;

    std::filesystem::path path(videoPath);
    pImpl->filename = path.filename().string();

    if (avformat_open_input(&pImpl->format_ctx, videoPath.c_str(), nullptr, nullptr) != 0) {
        throw std::runtime_error("Failed to open video file: " + videoPath);
    }

    if (avformat_find_stream_info(pImpl->format_ctx, nullptr) < 0) {
        throw std::runtime_error("Failed to find stream information");
    }

    for (unsigned int i = 0; i < pImpl->format_ctx->nb_streams; i++) {
        if (pImpl->format_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            pImpl->video_stream_idx = i;
            break;
        }
    }

    if (pImpl->video_stream_idx == -1) {
        throw std::runtime_error("No video stream found");
    }

    AVCodecParameters* codecpar = pImpl->format_ctx->streams[pImpl->video_stream_idx]->codecpar;
    pImpl->codec = avcodec_find_decoder(codecpar->codec_id);
    if (!pImpl->codec) {
        throw std::runtime_error("Unsupported codec");
    }

    pImpl->codec_ctx = avcodec_alloc_context3(pImpl->codec);
    if (!pImpl->codec_ctx) {
        throw std::runtime_error("Failed to allocate codec context");
    }

    if (avcodec_parameters_to_context(pImpl->codec_ctx, codecpar) < 0) {
        throw std::runtime_error("Failed to copy codec parameters");
    }

    if (avcodec_open2(pImpl->codec_ctx, pImpl->codec, nullptr) < 0) {
        throw std::runtime_error("Failed to open codec");
    }

    pImpl->width = codecpar->width;
    pImpl->height = codecpar->height;
    pImpl->container_format = pImpl->format_ctx->iformat->name;

    AVStream* video_stream = pImpl->format_ctx->streams[pImpl->video_stream_idx];
    AVRational avg_frame_rate = video_stream->avg_frame_rate;
    if (avg_frame_rate.den > 0) {
        double fps = av_q2d(avg_frame_rate);
        pImpl->duration_sec = pImpl->format_ctx->duration / (double)AV_TIME_BASE;
        pImpl->frame_count = static_cast<int64_t>(pImpl->duration_sec * fps);
    }

    pImpl->sws_ctx = sws_getContext(
        pImpl->width, pImpl->height, pImpl->codec_ctx->pix_fmt,
        pImpl->width, pImpl->height, AV_PIX_FMT_RGB24,
        SWS_BILINEAR, nullptr, nullptr, nullptr
    );

    if (!pImpl->sws_ctx) {
        throw std::runtime_error("Failed to initialize SwsContext");
    }

    pImpl->is_open = true;
}

VideoFrameExtractor::~VideoFrameExtractor() = default;

std::string VideoFrameExtractor::getFileName() const {
    return pImpl->filename;
}

std::string VideoFrameExtractor::getContainerFormat() const {
    return pImpl->container_format;
}

int VideoFrameExtractor::getWidth() const {
    return pImpl->width;
}

int VideoFrameExtractor::getHeight() const {
    return pImpl->height;
}

int64_t VideoFrameExtractor::getFrameCount() const {
    return pImpl->frame_count;
}

double VideoFrameExtractor::getDurationSec() const {
    return pImpl->duration_sec;
}

bool VideoFrameExtractor::isOpen() const {
    return pImpl->is_open;
}

std::shared_ptr<const FrameData> VideoFrameExtractor::getFrameAt(int frameIndex) const {
    if (!pImpl->is_open) {
        return nullptr;
    }

    if (frameIndex < 0 || frameIndex >= pImpl->frame_count) {
        throw std::out_of_range("Frame index out of range: " + std::to_string(frameIndex));
    }

    auto cached_frame = pImpl->cache.get(frameIndex);
    if (cached_frame) {
        return cached_frame;
    }

    AVStream* video_stream = pImpl->format_ctx->streams[pImpl->video_stream_idx];
    AVRational time_base = video_stream->time_base;
    AVRational frame_rate = av_guess_frame_rate(pImpl->format_ctx, video_stream, nullptr);

    int64_t target_ts = av_rescale_q(frameIndex, av_inv_q(frame_rate), time_base);

    if (avformat_seek_file(pImpl->format_ctx, pImpl->video_stream_idx, INT64_MIN, target_ts, INT64_MAX, AVSEEK_FLAG_BACKWARD) < 0) {
        throw std::runtime_error("Failed to seek to frame");
    }

    avcodec_flush_buffers(pImpl->codec_ctx);

    //AVPacket* packet = av_packet_alloc();
    //AVFrame* frame = av_frame_alloc();
	
	// use unique_ptr to auto free the object, so no need to free
	auto packet_deleter = [](AVPacket* p){ if(p) av_packet_free(&p); };
	auto frame_deleter = [](AVFrame* f){ if(f) av_frame_free(&f); };
	std::unique_ptr<AVPacket, decltype(packet_deleter)> packet(av_packet_alloc(), packet_deleter);
	std::unique_ptr<AVFrame, decltype(frame_deleter)> frame(av_frame_alloc(), frame_deleter);
	
	
    std::shared_ptr<FrameData> result = nullptr;
    int decoded_count = 0;

    try {
        while (decoded_count <= frameIndex) {
            int ret = av_read_frame(pImpl->format_ctx, packet.get());
            if (ret < 0) {
                if (ret == AVERROR_EOF) {
                    break;
                }
                throw std::runtime_error("Failed to read frame");
            }

            if (packet.get()->stream_index == pImpl->video_stream_idx) {
                ret = avcodec_send_packet(pImpl->codec_ctx, packet.get());
                if (ret < 0) {
                    av_packet_unref(packet.get());
                    throw std::runtime_error("Failed to send packet to decoder");
                }

                while (ret >= 0) {
                    ret = avcodec_receive_frame(pImpl->codec_ctx, frame.get());
                    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                        break;
                    } else if (ret < 0) {
                        av_packet_unref(packet.get());
                        throw std::runtime_error("Failed to receive frame from decoder");
                    }

                    if (decoded_count == frameIndex) {
                        int rgb_stride = pImpl->width * 3;
                        //auto pixels = std::make_shared<std::vector<uint8_t>>(pImpl->height * pImpl->width * rgb_stride);
                        auto pixels = std::make_shared<std::vector<uint8_t>>(pImpl->height * rgb_stride);

                        uint8_t* dest[4] = {pixels->data(), nullptr, nullptr, nullptr};
                        int dest_linesize[4] = {static_cast<int>(rgb_stride), 0, 0, 0};

                        //av_packet_unref(packet);
                        sws_scale(pImpl->sws_ctx, frame.get()->data, frame.get()->linesize, 0, pImpl->height, dest, dest_linesize);

                        result = std::make_shared<FrameData>();
                        result->width = pImpl->width;
                        result->height = pImpl->height;
                        result->pixels = pixels;

                        pImpl->cache.insert(frameIndex, result);
                    }

                    decoded_count++;
                    av_frame_unref(frame.get());
                }
            }

            av_packet_unref(packet.get());
        }

        //av_packet_free(&packet);
        //av_frame_free(&frame);

    } catch (...) {
        //if (packet) av_packet_free(&packet);
        //if (frame) av_frame_free(&frame);
        throw;
    }

    return result;
}

} // namespace vfe
