/**
 * SAM3 ViT Block Vulkan vs CPU 对比测试
 *
 * 目的：测试单个 ViT block 在不同后端下的正确性
 *
 * 使用方法：
 *   ./test_vit_block_vulkan_vs_cpu --block <block_idx> --backend [cpu|metal|vulkan] [--output file.txt]
 *   ./test_vit_block_vulkan_vs_cpu --compare [cpu|metal|vulkan] [cpu|metal|vulkan] --block <block_idx>
 */

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif
#ifdef GGML_USE_VULKAN
#include "ggml-vulkan.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>
#include <cstdint>
#include <map>

// 统计信息
struct TensorStats {
    float max_error = 0.0f;
    float mean_error = 0.0f;
    float std_error = 0.0f;
    float min_error = 1e20f;
    size_t count = 0;
};

// 计算误差统计
TensorStats compute_stats(const std::vector<float>& ref, const std::vector<float>& test) {
    TensorStats stats;
    if (ref.size() != test.size()) {
        fprintf(stderr, "Error: tensor sizes don't match: %zu vs %zu\n",
                ref.size(), test.size());
        return stats;
    }

    std::vector<float> errors;
    errors.reserve(ref.size());

    for (size_t i = 0; i < ref.size(); ++i) {
        float diff = std::abs(ref[i] - test[i]);
        errors.push_back(diff);
        stats.mean_error += diff;
        stats.max_error = std::max(stats.max_error, diff);
        stats.min_error = std::min(stats.min_error, diff);
    }

    stats.count = ref.size();
    stats.mean_error /= stats.count;

    // 计算标准差
    for (float e : errors) {
        float diff = e - stats.mean_error;
        stats.std_error += diff * diff;
    }
    stats.std_error = std::sqrt(stats.std_error / stats.count);

    return stats;
}

void print_stats(const TensorStats& stats, const char* name) {
    fprintf(stdout, "=== %s ===\n", name);
    fprintf(stdout, "Count: %zu\n", stats.count);
    fprintf(stdout, "Max error: %.6e\n", stats.max_error);
    fprintf(stdout, "Min error: %.6e\n", stats.min_error);
    fprintf(stdout, "Mean error: %.6e\n", stats.mean_error);
    fprintf(stdout, "Std error: %.6e\n", stats.std_error);
    fprintf(stdout, "\n");
}

// 简化的 LayerNorm
struct ggml_tensor* layer_norm(ggml_context* ctx, ggml_tensor* x,
                              ggml_tensor* w, ggml_tensor* b) {
    // x: [embed_dim, W, H, B]
    auto* mean = ggml_mean(ctx, x);
    auto* mean_sq = ggml_mul(ctx, mean, mean);
    auto* x_sq = ggml_mul(ctx, x, x);
    auto* var = ggml_sub(ctx, ggml_mean(ctx, x_sq), mean_sq);
    auto* eps = ggml_new_f32(ctx, 1e-6f);
    auto* std = ggml_sqrt(ctx, ggml_add(ctx, var, eps));

    auto* norm = ggml_mul(ctx, ggml_div(ctx, ggml_sub(ctx, x, mean), std), w);
    norm = ggml_add(ctx, norm, b);

    return norm;
}

// 简化的 RoPE
struct ggml_tensor* apply_rope(ggml_context* ctx, ggml_tensor* x, ggml_tensor* freqs) {
    // x: [head_dim, N, nheads_B]
    // freqs: [2, half, N]

    const int head_dim = x->ne[0];
    const int N = x->ne[1];
    const int nheads_B = x->ne[2];
    const int half = head_dim / 2;

    // Reshape x 到 [2, half, N, nheads_B]
    auto* x_pairs = ggml_reshape_4d(ctx, x, 2, half, N, nheads_B);

    // freqs_cis: [2, half, N] -> [2, half, N, 1]
    auto* fc = ggml_reshape_4d(ctx, freqs, 2, half, N, 1);

    // 提取 cos 和 sin
    auto* cos_f = ggml_view_4d(ctx, fc, 1, half, N, 1,
                                fc->nb[1], fc->nb[2], fc->nb[3], 0);
    auto* sin_f = ggml_view_4d(ctx, fc, 1, half, N, 1,
                                fc->nb[1], fc->nb[2], fc->nb[3], fc->nb[0]);

    // 提取实部和虚部
    auto* x_re = ggml_view_4d(ctx, x_pairs, 1, half, N, nheads_B,
                               x_pairs->nb[1], x_pairs->nb[2], x_pairs->nb[3], 0);
    auto* x_im = ggml_view_4d(ctx, x_pairs, 1, half, N, nheads_B,
                               x_pairs->nb[1], x_pairs->nb[2], x_pairs->nb[3], x_pairs->nb[0]);

    // 复数乘法
    auto* out_re = ggml_sub(ctx, ggml_mul(ctx, x_re, cos_f),
                                   ggml_mul(ctx, x_im, sin_f));
    auto* out_im = ggml_add(ctx, ggml_mul(ctx, x_re, sin_f),
                                   ggml_mul(ctx, x_im, cos_f));

    // 交错合并
    auto* out = ggml_concat(ctx, out_re, out_im, 0);
    return ggml_reshape_3d(ctx, ggml_cont(ctx, out), head_dim, N, nheads_B);
}

// 测试单个 ViT block
bool test_vit_block(ggml_backend_t backend, const std::string& backend_name,
                    const std::vector<float>& x_input,
                    const std::vector<float>& qkv_w, const std::vector<float>& qkv_b,
                    const std::vector<float>& proj_w, const std::vector<float>& proj_b,
                    const std::vector<float>& mlp_w1, const std::vector<float>& mlp_b1,
                    const std::vector<float>& mlp_w2, const std::vector<float>& mlp_b2,
                    const std::vector<float>& norm1_w, const std::vector<float>& norm1_b,
                    const std::vector<float>& norm2_w, const std::vector<float>& norm2_b,
                    const std::vector<float>& freqs,
                    int block_idx,
                    std::vector<float>& output) {

    const int E = 1024;  // embed_dim
    const int NH = 16;    // num_heads
    const int HD = 64;    // head_dim
    const int W = 72;     // 72x72 grid
    const int H = 72;
    const int B = 1;      // batch
    const int half = HD / 2;

    const size_t buf_size = ggml_tensor_overhead() * 1024 + ggml_graph_overhead();
    struct ggml_init_params gparams = {buf_size, nullptr, true};
    auto* ctx = ggml_init(gparams);
    if (!ctx) {
        fprintf(stderr, "Failed to init ggml context\n");
        return false;
    }

    // 创建输入张量
    auto* x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, E, W, H, B);
    ggml_set_name(x, "x");
    ggml_set_input(x);

    // 创建权重张量
    auto* qkv_w_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3 * E, E);
    ggml_set_name(qkv_w_tensor, "qkv_w");
    ggml_set_input(qkv_w_tensor);

    auto* qkv_b_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3 * E);
    ggml_set_name(qkv_b_tensor, "qkv_b");
    ggml_set_input(qkv_b_tensor);

    auto* proj_w_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, E, E);
    ggml_set_name(proj_w_tensor, "proj_w");
    ggml_set_input(proj_w_tensor);

    auto* proj_b_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, E);
    ggml_set_name(proj_b_tensor, "proj_b");
    ggml_set_input(proj_b_tensor);

    auto* mlp_w1_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4 * E, E);
    ggml_set_name(mlp_w1_tensor, "mlp_w1");
    ggml_set_input(mlp_w1_tensor);

    auto* mlp_b1_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4 * E);
    ggml_set_name(mlp_b1_tensor, "mlp_b1");
    ggml_set_input(mlp_b1_tensor);

    auto* mlp_w2_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, E, 4 * E);
    ggml_set_name(mlp_w2_tensor, "mlp_w2");
    ggml_set_input(mlp_w2_tensor);

    auto* mlp_b2_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, E);
    ggml_set_name(mlp_b2_tensor, "mlp_b2");
    ggml_set_input(mlp_b2_tensor);

    auto* norm1_w_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, E);
    ggml_set_name(norm1_w_tensor, "norm1_w");
    ggml_set_input(norm1_w_tensor);

    auto* norm1_b_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, E);
    ggml_set_name(norm1_b_tensor, "norm1_b");
    ggml_set_input(norm1_b_tensor);

    auto* norm2_w_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, E);
    ggml_set_name(norm2_w_tensor, "norm2_w");
    ggml_set_input(norm2_w_tensor);

    auto* norm2_b_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, E);
    ggml_set_name(norm2_b_tensor, "norm2_b");
    ggml_set_input(norm2_b_tensor);

    auto* freqs_tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, half, W * H);
    ggml_set_name(freqs_tensor, "freqs");
    ggml_set_input(freqs_tensor);

    // 构建block前向传播
    auto* shortcut = x;

    // Pre-norm 1
    x = layer_norm(ctx, x, norm1_w_tensor, norm1_b_tensor);

    // QKV projection
    auto* qkv = ggml_mul_mat(ctx, qkv_w_tensor, x);
    qkv = ggml_add(ctx, qkv, qkv_b_tensor);

    // Reshape: [3*E, W, H, B] -> [E, 3, W*H, B]
    qkv = ggml_reshape_4d(ctx, qkv, E, 3, W * H, B);
    qkv = ggml_cont(ctx, ggml_permute(ctx, qkv, 0, 3, 1, 2));

    // 提取 Q, K, V
    auto* Q = ggml_view_3d(ctx, qkv, E, W * H, B,
                           qkv->nb[1], qkv->nb[2], 0);
    auto* K = ggml_view_3d(ctx, qkv, E, W * H, B,
                           qkv->nb[1], qkv->nb[2], 1 * qkv->nb[3]);
    auto* V = ggml_view_3d(ctx, qkv, E, W * H, B,
                           qkv->nb[1], qkv->nb[2], 2 * qkv->nb[3]);

    // Reshape 到 [HD, N, NH, B]
    Q = ggml_reshape_4d(ctx, Q, HD, NH, W * H, B);
    Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));
    Q = ggml_reshape_3d(ctx, Q, HD, W * H, NH * B);

    K = ggml_reshape_4d(ctx, K, HD, NH, W * H, B);
    K = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));
    K = ggml_reshape_3d(ctx, K, HD, W * H, NH * B);

    V = ggml_reshape_4d(ctx, V, HD, NH, W * H, B);
    V = ggml_permute(ctx, V, 0, 2, 1, 3);

    // 应用 RoPE
    Q = apply_rope(ctx, Q, freqs_tensor);
    K = apply_rope(ctx, K, freqs_tensor);

    // Flash attention (简化版本，使用标准attention)
    auto* K_T = ggml_cont(ctx, ggml_permute(ctx, K, 1, 0, 2));
    auto* QK = ggml_mul_mat(ctx, K_T, Q);
    auto* scale = ggml_new_f32(ctx, 1.0f / std::sqrt((float)HD));
    QK = ggml_scale(ctx, QK, scale);

    // Softmax
    auto* attn = ggml_soft_max(ctx, QK);

    // Apply to V
    auto* attn_out = ggml_mul_mat(ctx, V, attn);

    // Reshape 回 [E, W, H, B]
    attn_out = ggml_reshape_4d(ctx, attn_out, HD, NH, W * H, B);
    attn_out = ggml_cont(ctx, ggml_permute(ctx, attn_out, 0, 2, 1, 3));
    attn_out = ggml_reshape_4d(ctx, attn_out, E, W, H, B);

    // Output projection
    attn_out = ggml_mul_mat(ctx, proj_w_tensor, attn_out);
    attn_out = ggml_add(ctx, attn_out, proj_b_tensor);

    // Residual connection
    x = ggml_add(ctx, attn_out, shortcut);

    // Pre-norm 2
    shortcut = x;
    x = layer_norm(ctx, x, norm2_w_tensor, norm2_b_tensor);

    // MLP
    auto* mlp = ggml_mul_mat(ctx, mlp_w1_tensor, x);
    mlp = ggml_add(ctx, mlp, mlp_b1_tensor);
    mlp = ggml_gelu(ctx, mlp);
    mlp = ggml_mul_mat(ctx, mlp_w2_tensor, mlp);
    mlp = ggml_add(ctx, mlp, mlp_b2_tensor);

    // Residual connection
    x = ggml_add(ctx, mlp, shortcut);

    ggml_set_name(x, "output");
    ggml_set_output(x);

    // 构建图
    auto* graph = ggml_new_graph_custom(ctx, 4096, false);
    ggml_build_forward_expand(graph, x);

    // 分配
    auto* galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_reserve(galloc, graph) ||
        !ggml_gallocr_alloc_graph(galloc, graph)) {
        fprintf(stderr, "Failed to allocate graph\n");
        ggml_gallocr_free(galloc);
        ggml_free(ctx);
        return false;
    }

    // 设置输入数据
    ggml_backend_tensor_set(x, x_input.data(), 0, x_input.size() * sizeof(float));
    ggml_backend_tensor_set(qkv_w_tensor, qkv_w.data(), 0, qkv_w.size() * sizeof(float));
    ggml_backend_tensor_set(qkv_b_tensor, qkv_b.data(), 0, qkv_b.size() * sizeof(float));
    ggml_backend_tensor_set(proj_w_tensor, proj_w.data(), 0, proj_w.size() * sizeof(float));
    ggml_backend_tensor_set(proj_b_tensor, proj_b.data(), 0, proj_b.size() * sizeof(float));
    ggml_backend_tensor_set(mlp_w1_tensor, mlp_w1.data(), 0, mlp_w1.size() * sizeof(float));
    ggml_backend_tensor_set(mlp_b1_tensor, mlp_b1.data(), 0, mlp_b1.size() * sizeof(float));
    ggml_backend_tensor_set(mlp_w2_tensor, mlp_w2.data(), 0, mlp_w2.size() * sizeof(float));
    ggml_backend_tensor_set(mlp_b2_tensor, mlp_b2.data(), 0, mlp_b2.size() * sizeof(float));
    ggml_backend_tensor_set(norm1_w_tensor, norm1_w.data(), 0, norm1_w.size() * sizeof(float));
    ggml_backend_tensor_set(norm1_b_tensor, norm1_b.data(), 0, norm1_b.size() * sizeof(float));
    ggml_backend_tensor_set(norm2_w_tensor, norm2_w.data(), 0, norm2_w.size() * sizeof(float));
    ggml_backend_tensor_set(norm2_b_tensor, norm2_b.data(), 0, norm2_b.size() * sizeof(float));
    ggml_backend_tensor_set(freqs_tensor, freqs.data(), 0, freqs.size() * sizeof(float));

    // 计算
    auto t0 = std::chrono::high_resolution_clock::now();
    if (ggml_backend_is_cpu(backend)) {
        ggml_backend_cpu_set_n_threads(backend, 4);
    }
    auto status = ggml_backend_graph_compute(backend, graph);
    auto t1 = std::chrono::high_resolution_clock::now();

    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "Graph compute failed\n");
        ggml_gallocr_free(galloc);
        ggml_free(ctx);
        return false;
    }

    // 读取输出
    output.resize(x_input.size());
    ggml_backend_tensor_get(x, output.data(), 0, output.size() * sizeof(float));

    ggml_gallocr_free(galloc);
    ggml_free(ctx);

    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    fprintf(stdout, "[%s] Block %d compute time: %lld ms\n",
            backend_name.c_str(), block_idx, ms);

    return true;
}

bool init_backend(const std::string& backend_name, ggml_backend_t& backend) {
    if (backend_name == "cpu") {
        backend = ggml_backend_cpu_init();
    }
#ifdef GGML_USE_METAL
    else if (backend_name == "metal") {
        backend = ggml_backend_metal_init();
    }
#endif
#ifdef GGML_USE_VULKAN
    else if (backend_name == "vulkan") {
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            enum ggml_backend_dev_type dev_type = ggml_backend_dev_type(dev);
            if (dev_type == GGML_BACKEND_DEVICE_TYPE_GPU ||
                dev_type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                backend = ggml_backend_vk_init(i);
                break;
            }
        }
    }
#endif
    else {
        fprintf(stderr, "Unknown backend: %s\n", backend_name.c_str());
        return false;
    }

    if (!backend) {
        fprintf(stderr, "Failed to initialize backend: %s\n", backend_name.c_str());
        return false;
    }

    return true;
}

void print_help(const char* argv0) {
    fprintf(stdout, "Usage: %s [options]\n", argv0);
    fprintf(stdout, "Options:\n");
    fprintf(stdout, "  --backend <type>   Backend to use (cpu|metal|vulkan)\n");
    fprintf(stdout, "  --block <idx>      ViT block index to test (0-31)\n");
    fprintf(stdout, "  --compare <b1> <b2> Compare two backends\n");
    fprintf(stdout, "  --help             Show this help message\n");
}

int main(int argc, char* argv[]) {
    std::string backend1 = "cpu";
    std::string backend2;
    int block_idx = 15;  // 默认测试第15个block（global attention）
    bool compare_mode = false;

    // 解析命令行参数
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_help(argv[0]);
            return 0;
        } else if (arg == "--backend" && i + 1 < argc) {
            backend1 = argv[++i];
        } else if (arg == "--block" && i + 1 < argc) {
            block_idx = std::atoi(argv[++i]);
        } else if (arg == "--compare" && i + 2 < argc) {
            compare_mode = true;
            backend1 = argv[++i];
            backend2 = argv[++i];
        }
    }

    fprintf(stdout, "=== SAM3 ViT Block Vulkan vs CPU Test ===\n");
    fprintf(stdout, "Block index: %d\n", block_idx);
    fprintf(stdout, "Backend 1: %s\n", backend1.c_str());
    if (compare_mode) {
        fprintf(stdout, "Backend 2: %s\n", backend2.c_str());
    }
    fprintf(stdout, "\n");

    // 参数
    const int E = 1024;
    const int W = 72;
    const int H = 72;
    const int B = 1;
    const int total = E * W * H * B;

    // 生成测试数据
    std::vector<float> x_input(total);
    for (size_t i = 0; i < x_input.size(); ++i) {
        x_input[i] = (float)(rand() % 100) / 100.0f;
    }

    // 生成随机权重
    std::vector<float> qkv_w(3 * E * E);
    std::vector<float> qkv_b(3 * E);
    std::vector<float> proj_w(E * E);
    std::vector<float> proj_b(E);
    std::vector<float> mlp_w1(4 * E * E);
    std::vector<float> mlp_b1(4 * E);
    std::vector<float> mlp_w2(E * 4 * E);
    std::vector<float> mlp_b2(E);
    std::vector<float> norm1_w(E);
    std::vector<float> norm1_b(E);
    std::vector<float> norm2_w(E);
    std::vector<float> norm2_b(E);

    for (auto& v : qkv_w) v = (float)(rand() % 100) / 1000.0f;
    for (auto& v : qkv_b) v = (float)(rand() % 100) / 1000.0f;
    for (auto& v : proj_w) v = (float)(rand() % 100) / 1000.0f;
    for (auto& v : proj_b) v = (float)(rand() % 100) / 1000.0f;
    for (auto& v : mlp_w1) v = (float)(rand() % 100) / 1000.0f;
    for (auto& v : mlp_b1) v = (float)(rand() % 100) / 1000.0f;
    for (auto& v : mlp_w2) v = (float)(rand() % 100) / 1000.0f;
    for (auto& v : mlp_b2) v = (float)(rand() % 100) / 1000.0f;
    for (auto& v : norm1_w) v = 1.0f;
    for (auto& v : norm1_b) v = 0.0f;
    for (auto& v : norm2_w) v = 1.0f;
    for (auto& v : norm2_b) v = 0.0f;

    // 生成频率
    const int half = 32;
    std::vector<float> freqs(2 * half * W * H);
    const float theta = 10000.0f;
    const int half_dim = half / 2;

    for (int idx = 0; idx < W * H; ++idx) {
        float t_x = (float)(idx % W);
        float t_y = (float)(idx / W);
        for (int i = 0; i < half_dim; ++i) {
            float freq = 1.0f / std::pow(theta, (float)(i * 4) / half);
            freqs[idx * half * 2 + i * 2 + 0] = std::cos(t_x * freq);
            freqs[idx * half * 2 + i * 2 + 1] = std::sin(t_x * freq);
        }
        for (int i = 0; i < half_dim; ++i) {
            float freq = 1.0f / std::pow(theta, (float)(i * 4) / half);
            freqs[W * H * half * 2 + idx * half * 2 + i * 2 + 0] = std::cos(t_y * freq);
            freqs[W * H * half * 2 + idx * half * 2 + i * 2 + 1] = std::sin(t_y * freq);
        }
    }

    // 初始化后端
    ggml_backend_t backend1_ptr = nullptr;
    ggml_backend_t backend2_ptr = nullptr;

    if (!init_backend(backend1, backend1_ptr)) {
        return 1;
    }

    if (compare_mode && !init_backend(backend2, backend2_ptr)) {
        return 1;
    }

    // 运行测试
    std::vector<float> out1, out2;

    fprintf(stdout, "Running ViT block %d test with %s backend...\n", block_idx, backend1.c_str());
    if (!test_vit_block(backend1_ptr, backend1, x_input, qkv_w, qkv_b, proj_w, proj_b,
                      mlp_w1, mlp_b1, mlp_w2, mlp_b2,
                      norm1_w, norm1_b, norm2_w, norm2_b,
                      freqs, block_idx, out1)) {
        return 1;
    }

    TensorStats stats;

    if (compare_mode) {
        fprintf(stdout, "Running ViT block %d test with %s backend...\n", block_idx, backend2.c_str());
        if (!test_vit_block(backend2_ptr, backend2, x_input, qkv_w, qkv_b, proj_w, proj_b,
                          mlp_w1, mlp_b1, mlp_w2, mlp_b2,
                          norm1_w, norm1_b, norm2_w, norm2_b,
                          freqs, block_idx, out2)) {
            return 1;
        }

        // 比较结果
        fprintf(stdout, "\n=== Comparing %s vs %s ===\n", backend1.c_str(), backend2.c_str());
        fprintf(stdout, "\n");

        stats = compute_stats(out1, out2);
        print_stats(stats, "ViT Block Output Error");

        // 判断是否通过
        const float tolerance = 1e-3f;
        bool pass = stats.max_error < tolerance;

        fprintf(stdout, "\n=== Test Result ===\n");
        fprintf(stdout, "Block %d: %s (max_error = %.6e, tolerance = %.6e)\n",
                block_idx, pass ? "PASS" : "FAIL", stats.max_error, tolerance);

        if (pass) {
            fprintf(stdout, "\n✓ ViT block operations are consistent between backends\n");
            return 0;
        } else {
            fprintf(stdout, "\n✗ ViT block has significant differences between backends\n");
            fprintf(stdout, "\nThis indicates a problem with one of the operations:\n");
            fprintf(stdout, "  - LayerNorm\n");
            fprintf(stdout, "  - QKV projection\n");
            fprintf(stdout, "  - RoPE application\n");
            fprintf(stdout, "  - Attention computation\n");
            fprintf(stdout, "  - MLP layers\n");
            fprintf(stdout, "  - Residual connections\n");
            return 1;
        }
    } else {
        fprintf(stdout, "\n=== ViT Block Output Statistics ===\n");
        fprintf(stdout, "Output: %zu elements\n", out1.size());
        fprintf(stdout, "First 10 values: ");
        for (int i = 0; i < 10 && i < (int)out1.size(); ++i) {
            fprintf(stdout, "%.6f ", out1[i]);
        }
        fprintf(stdout, "\n");

        if (backend1 == "vulkan") {
            fprintf(stdout, "\n=== Vulkan Backend Diagnostics ===\n");
            fprintf(stdout, "Check for:\n");
            fprintf(stdout, "  - Memory layout issues\n");
            fprintf(stdout, "  - Numerical precision in attention\n");
            fprintf(stdout, "  - RoPE computation correctness\n");
        }
    }

    return 0;
}
