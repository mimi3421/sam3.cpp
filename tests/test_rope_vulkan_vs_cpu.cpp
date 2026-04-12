/**
 * SAM3 RoPE Vulkan vs CPU 对比测试
 *
 * 目的：测试 RoPE（Rotary Positional Embedding）操作在不同后端下的正确性
 *
 * 使用方法：
 *   ./test_rope_vulkan_vs_cpu --backend [cpu|metal|vulkan] [--output file.txt]
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

// 计算两个张量的误差统计
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

// 输出统计信息
void print_stats(const TensorStats& stats, const char* name) {
    fprintf(stdout, "=== %s ===\n", name);
    fprintf(stdout, "Count: %zu\n", stats.count);
    fprintf(stdout, "Max error: %.6e\n", stats.max_error);
    fprintf(stdout, "Min error: %.6e\n", stats.min_error);
    fprintf(stdout, "Mean error: %.6e\n", stats.mean_error);
    fprintf(stdout, "Std error: %.6e\n", stats.std_error);
    fprintf(stdout, "\n");
}

// 模拟 RoPE 频率计算（与 sam3_compute_axial_cis 相同）
void compute_rope_freqs(std::vector<float>& out, int dim, int N, float theta = 10000.0f, float scale_pos = 1.0f) {
    const int half_dim = dim / 4;  // 16 for dim=64
    
    out.resize(N * dim);
    
    // 计算频率基
    std::vector<float> freqs(half_dim);
    for (int i = 0; i < half_dim; ++i) {
        freqs[i] = 1.0f / std::pow(theta, (float)(i * 4) / (float)dim);
    }
    
    // 对每个空间位置计算轴向频率
    for (int idx = 0; idx < N; ++idx) {
        float t_x = (float)(idx % (int)std::sqrt(N)) * scale_pos;
        float t_y = (float)(idx / (int)std::sqrt(N)) * scale_pos;
        
        // X 频率 -> 前 16 个复数值
        for (int i = 0; i < half_dim; ++i) {
            float angle_x = t_x * freqs[i];
            out[idx * dim + i * 2 + 0] = std::cos(angle_x);
            out[idx * dim + i * 2 + 1] = std::sin(angle_x);
        }
        // Y 频率 -> 后 16 个复数值
        for (int i = 0; i < half_dim; ++i) {
            float angle_y = t_y * freqs[i];
            out[idx * dim + half_dim * 2 + i * 2 + 0] = std::cos(angle_y);
            out[idx * dim + half_dim * 2 + i * 2 + 1] = std::sin(angle_y);
        }
    }
}

// 测试 RoPE 操作
bool test_rope(ggml_backend_t backend, const std::string& backend_name, 
               const std::vector<float>& Q_input, const std::vector<float>& K_input,
               const std::vector<float>& freqs, 
               std::vector<float>& Q_out, std::vector<float>& K_out) {
    const int head_dim = 64;
    const int N = 5184;  // 72 * 72
    const int nheads_B = 16;
    const int half = head_dim / 2;
    
    // 创建 ggml context
    const size_t buf_size = ggml_tensor_overhead() * 512 + ggml_graph_overhead();
    struct ggml_init_params gparams = {buf_size, nullptr, true};
    auto* ctx = ggml_init(gparams);
    if (!ctx) {
        fprintf(stderr, "Failed to init ggml context\n");
        return false;
    }
    
    // 创建输入张量
    auto* Q_tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, N, nheads_B);
    ggml_set_name(Q_tensor, "Q_input");
    ggml_set_input(Q_tensor);
    
    auto* K_tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, N, nheads_B);
    ggml_set_name(K_tensor, "K_input");
    ggml_set_input(K_tensor);
    
    auto* freqs_tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, half, N);
    ggml_set_name(freqs_tensor, "freqs_cis");
    ggml_set_input(freqs_tensor);
    
    // 模拟 sam3_apply_rope 函数的实现
    // Reshape x 到 [2, half, N, nheads_B] 以分离实部和虚部
    auto* Q_pairs = ggml_reshape_4d(ctx, Q_tensor, 2, half, N, nheads_B);
    auto* K_pairs = ggml_reshape_4d(ctx, K_tensor, 2, half, N, nheads_B);
    
    // freqs_cis: [2, 32, N] -> [2, half, N, 1] for broadcast
    auto* fc = ggml_reshape_4d(ctx, freqs_tensor, 2, half, N, 1);
    
    // 提取 cos 和 sin
    auto* cos_f = ggml_view_4d(ctx, fc, 1, half, N, 1,
                                fc->nb[1], fc->nb[2], fc->nb[3], 0);
    auto* sin_f = ggml_view_4d(ctx, fc, 1, half, N, 1,
                                fc->nb[1], fc->nb[2], fc->nb[3], fc->nb[0]);
    
    // 提取实部和虚部
    auto* Q_re = ggml_view_4d(ctx, Q_pairs, 1, half, N, nheads_B,
                               Q_pairs->nb[1], Q_pairs->nb[2], Q_pairs->nb[3], 0);
    auto* Q_im = ggml_view_4d(ctx, Q_pairs, 1, half, N, nheads_B,
                               Q_pairs->nb[1], Q_pairs->nb[2], Q_pairs->nb[3], Q_pairs->nb[0]);
    
    auto* K_re = ggml_view_4d(ctx, K_pairs, 1, half, N, nheads_B,
                               K_pairs->nb[1], K_pairs->nb[2], K_pairs->nb[3], 0);
    auto* K_im = ggml_view_4d(ctx, K_pairs, 1, half, N, nheads_B,
                               K_pairs->nb[1], K_pairs->nb[2], K_pairs->nb[3], K_pairs->nb[0]);
    
    // 复数乘法: (x_re + j*x_im) * (cos + j*sin)
    // out_re = x_re*cos - x_im*sin
    // out_im = x_re*sin + x_im*cos
    auto* Q_out_re = ggml_sub(ctx, ggml_mul(ctx, Q_re, cos_f), 
                                   ggml_mul(ctx, Q_im, sin_f));
    auto* Q_out_im = ggml_add(ctx, ggml_mul(ctx, Q_re, sin_f), 
                                   ggml_mul(ctx, Q_im, cos_f));
    
    auto* K_out_re = ggml_sub(ctx, ggml_mul(ctx, K_re, cos_f), 
                                   ggml_mul(ctx, K_im, sin_f));
    auto* K_out_im = ggml_add(ctx, ggml_mul(ctx, K_re, sin_f), 
                                   ggml_mul(ctx, K_im, cos_f));
    
    // 交错合并
    auto* Q_out_temp = ggml_concat(ctx, Q_out_re, Q_out_im, 0);
    auto* K_out_temp = ggml_concat(ctx, K_out_re, K_out_im, 0);
    
    // Reshape 回 [head_dim, N, nheads_B]
    Q_out_temp = ggml_reshape_3d(ctx, Q_out_temp, head_dim, N, nheads_B);
    K_out_temp = ggml_reshape_3d(ctx, K_out_temp, head_dim, N, nheads_B);
    
    Q_out_temp = ggml_cont(ctx, Q_out_temp);
    K_out_temp = ggml_cont(ctx, K_out_temp);
    
    // 设置输出张量
    ggml_set_name(Q_out_temp, "Q_output");
    ggml_set_output(Q_out_temp);
    
    ggml_set_name(K_out_temp, "K_output");
    ggml_set_output(K_out_temp);
    
    // 构建和分配 graph
    auto* graph = ggml_new_graph_custom(ctx, 4096, false);
    ggml_build_forward_expand(graph, Q_out_temp);
    ggml_build_forward_expand(graph, K_out_temp);
    
    auto* galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_reserve(galloc, graph) ||
        !ggml_gallocr_alloc_graph(galloc, graph)) {
        fprintf(stderr, "Failed to allocate graph\n");
        ggml_gallocr_free(galloc);
        ggml_free(ctx);
        return false;
    }
    
    // 设置输入
    ggml_backend_tensor_set(Q_tensor, Q_input.data(), 
                             (ggml_backend_buffer_type)0, Q_input.size() * sizeof(float));
    ggml_backend_tensor_set(K_tensor, K_input.data(), 
                             (ggml_backend_buffer_type)0, K_input.size() * sizeof(float));
    ggml_backend_tensor_set(freqs_tensor, freqs.data(), 
                             (ggml_backend_buffer_type)0, freqs.size() * sizeof(float));
    
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
    Q_out.resize(Q_input.size());
    K_out.resize(K_input.size());
    
    ggml_backend_tensor_get(Q_out_temp, Q_out.data(), 
                             (ggml_backend_buffer_type)0, Q_out.size() * sizeof(float));
    ggml_backend_tensor_get(K_out_temp, K_out.data(), 
                             (ggml_backend_buffer_type)0, K_out.size() * sizeof(float));
    
    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    fprintf(stdout, "[%s] RoPE compute time: %lld ms\n", backend_name.c_str(), ms);
    
    return true;
}

// 初始化后端
bool init_backend(const std::string& backend_name, ggml_backend_t& backend) {
    if (backend_name == "cpu") {
        backend = ggml_backend_cpu_init();
    }
#ifdef GGML_USE_METAL
    else if (backend_name == "metal") {
        fprintf(stdout, "Initializing Metal backend...\n");
        backend = ggml_backend_metal_init();
    }
#endif
#ifdef GGML_USE_VULKAN
    else if (backend_name == "vulkan") {
        fprintf(stdout, "Initializing Vulkan backend...\n");
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            enum ggml_backend_dev_type dev_type = ggml_backend_dev_type(dev);
            if (dev_type == GGML_BACKEND_DEVICE_TYPE_GPU || 
                dev_type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                const char* dev_name = ggml_backend_dev_name(dev);
                fprintf(stdout, "Using device: %s\n", dev_name);
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

// 打印帮助信息
void print_help(const char* argv0) {
    fprintf(stdout, "Usage: %s [options]\n", argv0);
    fprintf(stdout, "Options:\n");
    fprintf(stdout, "  --backend <type>  Backend to use (cpu|metal|vulkan)\n");
    fprintf(stdout, "  --output <file>   Output file for results\n");
    fprintf(stdout, "  --compare <b1> <b2> Compare two backends\n");
    fprintf(stdout, "  --help             Show this help message\n");
}

int main(int argc, char* argv[]) {
    std::string backend1 = "cpu";
    std::string backend2;
    std::string output_file;
    bool compare_mode = false;
    
    // 解析命令行参数
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_help(argv[0]);
            return 0;
        } else if (arg == "--backend" && i + 1 < argc) {
            backend1 = argv[++i];
        } else if (arg == "--compare" && i + 2 < argc) {
            compare_mode = true;
            backend1 = argv[++i];
            backend2 = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_file = argv[++i];
        }
    }
    
    fprintf(stdout, "=== SAM3 RoPE Vulkan vs CPU Test ===\n");
    fprintf(stdout, "Backend 1: %s\n", backend1.c_str());
    if (compare_mode) {
        fprintf(stdout, "Backend 2: %s\n", backend2.c_str());
    }
    fprintf(stdout, "\n");
    
    // 生成测试数据
    const int head_dim = 64;
    const int N = 5184;  // 72 * 72
    const int nheads_B = 16;
    const int half = head_dim / 2;
    
    std::vector<float> Q_input(head_dim * N * nheads_B);
    std::vector<float> K_input(head_dim * N * nheads_B);
    std::vector<float> freqs(N * head_dim);
    
    // 填充随机数据
    for (size_t i = 0; i < Q_input.size(); ++i) {
        Q_input[i] = (float)(rand() % 100) / 100.0f;
        K_input[i] = (float)(rand() % 100) / 100.0f;
    }
    
    // 计算频率
    compute_rope_freqs(freqs, head_dim, N);
    
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
    std::vector<float> Q_out1, K_out1, Q_out2, K_out2;
    
    fprintf(stdout, "Running RoPE test with %s backend...\n", backend1.c_str());
    if (!test_rope(backend1_ptr, backend1, Q_input, K_input, freqs, Q_out1, K_out1)) {
        return 1;
    }
    
    TensorStats Q_stats, K_stats;
    
    if (compare_mode) {
        fprintf(stdout, "Running RoPE test with %s backend...\n", backend2.c_str());
        if (!test_rope(backend2_ptr, backend2, Q_input, K_input, freqs, Q_out2, K_out2)) {
            return 1;
        }
        
        // 比较结果
        fprintf(stdout, "\n=== Comparing %s vs %s ===\n", backend1.c_str(), backend2.c_str());
        fprintf(stdout, "\n");
        
        Q_stats = compute_stats(Q_out1, Q_out2);
        print_stats(Q_stats, "Q Output Error");
        
        K_stats = compute_stats(K_out1, K_out2);
        print_stats(K_stats, "K Output Error");
        
        // 判断是否通过
        const float tolerance = 1e-3f;
        bool q_pass = Q_stats.max_error < tolerance;
        bool k_pass = K_stats.max_error < tolerance;
        
        fprintf(stdout, "\n=== Test Result ===\n");
        fprintf(stdout, "Q: %s (max_error = %.6e, tolerance = %.6e)\n",
                q_pass ? "PASS" : "FAIL", Q_stats.max_error, tolerance);
        fprintf(stdout, "K: %s (max_error = %.6e, tolerance = %.6e)\n",
                k_pass ? "PASS" : "FAIL", K_stats.max_error, tolerance);
        
        if (q_pass && k_pass) {
            fprintf(stdout, "\n✓ RoPE operations are consistent between backends\n");
            return 0;
        } else {
            fprintf(stdout, "\n✗ RoPE operations have significant differences between backends\n");
            fprintf(stdout, "\nThis indicates a problem with the ggml implementation on Vulkan backend.\n");
            fprintf(stdout, "The issue is likely in one of these operations:\n");
            fprintf(stdout, "  - ggml_reshape_4d\n");
            fprintf(stdout, "  - ggml_view_4d (stride calculation)\n");
            fprintf(stdout, "  - ggml_concat\n");
            fprintf(stdout, "  - ggml_mul / ggml_add / ggml_sub\n");
            return 1;
        }
    } else {
        // 单后端模式，只输出统计
        fprintf(stdout, "\n=== RoPE Output Statistics ===\n");
        fprintf(stdout, "Q output: %zu elements\n", Q_out1.size());
        fprintf(stdout, "K output: %zu elements\n", K_out1.size());
        fprintf(stdout, "\nFirst 10 Q values: ");
        for (int i = 0; i < 10 && i < (int)Q_out1.size(); ++i) {
            fprintf(stdout, "%.6f ", Q_out1[i]);
        }
        fprintf(stdout, "\n");
        
        // 如果是 Vulkan，输出一些诊断信息
        if (backend1 == "vulkan") {
            fprintf(stdout, "\n=== Vulkan Backend Diagnostics ===\n");
            fprintf(stdout, "Check for:\n");
            fprintf(stdout, "  - GPU memory allocation issues\n");
            fprintf(stdout, "  - Numerical precision problems\n");
            fprintf(stdout, "  - Kernel launch failures (check with validation layers if available)\n");
        }
    }
    
    return 0;
}
