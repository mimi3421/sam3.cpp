#ifndef VIDEFRAMEEXTRACTOR_H
#define VIDEFRAMEEXTRACTOR_H

#include <string>
#include <vector>
#include <memory>
#include <cstdint>

namespace vfe {

struct FrameData {
    int width = 0;
    int height = 0;
    std::shared_ptr<const std::vector<uint8_t>> pixels;
};

class VideoFrameExtractor {
public:
    explicit VideoFrameExtractor(const std::string& videoPath);
    ~VideoFrameExtractor();

    VideoFrameExtractor(const VideoFrameExtractor&) = delete;
    VideoFrameExtractor& operator=(const VideoFrameExtractor&) = delete;

    VideoFrameExtractor(VideoFrameExtractor&&) = delete;
    VideoFrameExtractor& operator=(VideoFrameExtractor&&) = delete;

    std::string getFileName() const;
    std::string getContainerFormat() const;
    int getWidth() const;
    int getHeight() const;
    int64_t getFrameCount() const;
    double getDurationSec() const;
    bool isOpen() const;

    std::shared_ptr<const FrameData> getFrameAt(int frameIndex) const;

private:
    struct LRUCache;
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace vfe

#endif // VIDEFRAMEEXTRACTOR_H
