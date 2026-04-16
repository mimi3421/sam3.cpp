/**
 * SAM3 ViT Block Vulkan vs CPU 对比测试
 *
 * 目的：测试单个 ViT Block 在不同后端下的正确性（除 RoPE 外的其他环节）
 *
 * 使用方法：
 *   ./test_vit_block_vulkan_vs_cpu --backend [cpu|metal|vulkan] --block <idx> [--output file.txt]
 *   ./test_vit_block_vulkan_vs_cpu --compare [cpu|metal|vulkan] [cpu|metal|vulkan] --block <idx>
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

// 保存张量到文件
void save_tensor(const std::vector<float>& tensor, const std::string& filename) {
    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open()) {
        fprintf(stderr, "Failed to open file: %s\n", filename.c_str());
        return;
    }
    size_t n = tensor.size();
    out.write(reinterpret_cast<const char*>(&n), sizeof(size_t));
    out.write(reinterpret_cast<const char*>(tensor.data()), n * sizeof(float));
    out.close();
}

// 从文件加载张量
bool load_tensor(std::vector<float>& tensor, const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        return false;
    }
    size_t n;
    in.read(reinterpret_cast<char*>(&n), sizeof(size_t));
    tensor.resize(n);
    in.read(reinterpret_cast<char*>(tensor.data()), n * sizeof(float));
    in.close();
    return true;
}

// 模拟 SAM3 的 LayerNorm
struct ggml_tensor* sam3_layer_norm_test(ggml_context* ctx, ggml_tensor* x,
                                         ggml_tensor* w, ggml_tensor* b) {
    auto* normed = ggml_norm(ctx, x, 1e-5f);
    normed = ggml_mul(ctx, normed, w);
    normed = ggml_add(ctx, normed, b);
    return normed;
}

// 模拟 RoPE 频率计算（与 sam3_compute_axial_cis 相同）
void compute_rope_freqs(std::vector<float>& out, int dim, int N, float theta = 10000.0f, float scale_pos = 1.0f) {
    const int half_dim = dim / 4;

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

// 模拟 SAM3 的 RoPE 应用
struct ggml_tensor* sam3_apply_rope_test(ggml_context* ctx,
                                         ggml_tensor* x,
                                         ggml_tensor* freqs_cis) {
    const int head_dim = 64;
    const int N = x->ne[1];
    const int nheads_B = x->ne[2];
    const int half = head_dim / 2;

    // Reshape x 到 [2, half, N, nheads_B] 以分离实部和虚部
    auto* x_pairs = ggml_reshape_4d(ctx, x, 2, half, N, nheads_B);

    // freqs_cis: [2, 32, N] -> [2, half, N, 1] for broadcast
    auto* fc = ggml_reshape_4d(ctx, freqs_cis, 2, half, N, 1);

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

    // 复数乘法: (x_re + j*x_im) * (cos + j*sin)
    auto* out_re = ggml_sub(ctx, ggml_mul(ctx, x_re, cos_f),
                                ggml_mul(ctx, x_im, sin_f));
    auto* out_im = ggml_add(ctx, ggml_mul(ctx, x_re, sin_f),
                                ggml_mul(ctx, x_im, cos_f));

    // 交错合并
    auto* out_temp = ggml_concat(ctx, out_re, out_im, 0);

    // Reshape 回 [head_dim, N, nheads_B]
    out_temp = ggml_reshape_3d(ctx, out_temp, head_dim, N, nheads_B);
    out_temp = ggml_cont(ctx, out_temp);

    return out_temp;
}

// ViT Block 测试结构
struct VitBlockWeights {
    std::vector<float> norm1_w, norm1_b;
    std::vector<float> qkv_w, qkv_b;
    std::vector<float> proj_w, proj_b;
    std::vector<float> norm2_w, norm2_b;
    std::vector<float> mlp_fc1_w, mlp_fc1_b;
    std::vector<float> mlp_fc2_w, mlp_fc2_b;
    std::vector<float> freqs_cis;
};

// 生成随机权重
void generate_random_weights(VitBlockWeights& weights, int embed_dim, int mlp_dim,
                             int window_size, bool use_rope) {
    int n_windows = (72 / window_size) * (72 / window_size);
    int N = use_rope ? 72 * 72 : window_size * window_size * n_windows;

    // LayerNorm weights
    weights.norm1_w.resize(embed_dim);
    weights.norm1_b.resize(embed_dim);
    weights.norm2_w.resize(embed_dim);
    weights.norm2_b.resize(embed_dim);

    for (int i = 0; i < embed_dim; ++i) {
        weights.norm1_w[i] = 1.0f + (rand() % 100) / 1000.0f;
        weights.norm1_b[i] = (rand() % 100) / 1000.0f;
        weights.norm2_w[i] = 1.0f + (rand() % 100) / 1000.0f;
        weights.norm2_b[i] = (rand() % 100) / 1000.0f;
    }

    // QKV projection
    weights.qkv_w.resize(3 * embed_dim * embed_dim);
    weights.qkv_b.resize(3 * embed_dim);

    for (size_t i = 0; i < weights.qkv_w.size(); ++i) {
        weights.qkv_w[i] = (rand() % 1000 - 500) / 10000.0f;
    }
    for (size_t i = 0; i < weights.qkv_b.size(); ++i) {
        weights.qkv_b[i] = (rand() % 100) / 1000.0f;
    }

    // Output projection
    weights.proj_w.resize(embed_dim * embed_dim);
    weights.proj_b.resize(embed_dim);

    for (size_t i = 0; i < weights.proj_w.size(); ++i) {
        weights.proj_w[i] = (rand() % 1000 - 500) / 10000.0f;
    }
    for (size_t i = 0; i < weights.proj_b.size(); ++i) {
        weights.proj_b[i] = (rand() % 100) / 1000.0f;
    }

    // MLP
    weights.mlp_fc1_w.resize(mlp_dim * embed_dim);
    weights.mlp_fc1_b.resize(mlp_dim);
    weights.mlp_fc2_w.resize(embed_dim * mlp_dim);
    weights.mlp_fc2_b.resize(embed_dim);

    for (size_t i = 0; i < weights.mlp_fc1_w.size(); ++i) {
        weights.mlp_fc1_w[i] = (rand() % 1000 - 500) / 10000.0f;
    }
    for (size_t i = 0; i < weights.mlp_fc1_b.size(); ++i) {
        weights.mlp_fc1_b[i] = (rand() % 100) / 1000.0f;
    }
    for (size_t i = 0; i < weights.mlp_fc2_w.size(); ++i) {
        weights.mlp_fc2_w[i] = (rand() % 1000 - 500) / 10000.0f;
    }
    for (size_t i = 0; i < weights.mlp_fc2_b.size(); ++i) {
        weights.mlp_fc2_b[i] = (rand() % 100) / 1000.0f;
    }

    // RoPE frequencies
    if (use_rope) {
        compute_rope_freqs(weights.freqs_cis, 64, N);
    }
}

// 测试 ViT Block（单个环节）
bool test_vit_block_component(ggml_backend_t backend, const std::string& backend_name,
                              const std::vector<float>& input,
                              const VitBlockWeights& weights,
                              const std::string& component,
                              std::vector<float>& output,
                              bool is_global, int embed_dim, int mlp_dim, int window_size) {
    const int E = embed_dim;       // 1024
    const int NH = 16;             // num heads
    const int HD = E / NH;         // 64
    const int WS = window_size;    // 24
    const int N = 72 * 72;         // 5184

    // 创建 ggml context
    const size_t buf_size = ggml_tensor_overhead() * 1024 + ggml_graph_overhead();
    struct ggml_init_params gparams = {buf_size, nullptr, true};
    auto* ctx = ggml_init(gparams);
    if (!ctx) {
        fprintf(stderr, "Failed to init ggml context\n");
        return false;
    }

    // 创建输入张量
    auto* x_tensor = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, E, 72, 72, 1);
    ggml_set_name(x_tensor, "input");
    ggml_set_input(x_tensor);

    auto* x = x_tensor;
    auto* shortcut = x;

    // LayerNorm
    auto* norm1_w_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, E);
    auto* norm1_b_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, E);
    ggml_set_input(norm1_w_tensor);
    ggml_set_input(norm1_b_tensor);

    // 声明所有可能的tensor变量（在函数开头，确保在所有分支中可用）
    ggml_tensor* qkv_w_tensor = nullptr;
    ggml_tensor* qkv_b_tensor = nullptr;
    ggml_tensor* freqs_cis_tensor = nullptr;
    ggml_tensor* proj_w_tensor = nullptr;
    ggml_tensor* proj_b_tensor = nullptr;
    ggml_tensor* norm2_w_tensor = nullptr;
    ggml_tensor* norm2_b_tensor = nullptr;
    ggml_tensor* mlp_fc1_w_tensor = nullptr;
    ggml_tensor* mlp_fc1_b_tensor = nullptr;
    ggml_tensor* mlp_fc2_w_tensor = nullptr;
    ggml_tensor* mlp_fc2_b_tensor = nullptr;

    if (component == "layernorm") {
        x = sam3_layer_norm_test(ctx, x, norm1_w_tensor, norm1_b_tensor);
        ggml_set_name(x, "layernorm_output");
        ggml_set_output(x);
    } else {
        x = sam3_layer_norm_test(ctx, x, norm1_w_tensor, norm1_b_tensor);

        // Window partition
        const int64_t w0 = x->ne[1];
        const int64_t h0 = x->ne[2];

        if (!is_global) {
            x = ggml_win_part(ctx, x, WS);
        }

        const int64_t W_cur = x->ne[1];
        const int64_t H_cur = x->ne[2];
        const int64_t B_cur = x->ne[3];

        // QKV projection
        // Note: sam3.cpp uses blk.qkv_w with shape [3*E, E] directly with x
        // For ggml_mul_mat, we need to ensure dimensions are compatible
        qkv_w_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, E, 3*E);  // Transposed
        qkv_b_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3*E);
        ggml_set_input(qkv_w_tensor);
        ggml_set_input(qkv_b_tensor);

        if (component == "qkv") {
			// 1. 强制将 x 转为连续内存布局（Vulkan 等 GPU 后端必需）
			x = ggml_cont(ctx, x); 
            // Reshape x to 3D: [E, W_cur*H_cur*B_cur, 1] for proper matrix multiplication
            auto* x_reshaped = ggml_reshape_3d(ctx, x, E, W_cur * H_cur * B_cur, 1);
            auto* cur = ggml_mul_mat(ctx, qkv_w_tensor, x_reshaped);
            cur = ggml_add(ctx, cur, qkv_b_tensor);
            ggml_set_name(cur, "qkv_output");
            ggml_set_output(cur);
            x = cur;
        } else {
            // Reshape x to 3D: [E, W_cur*H_cur*B_cur, 1] for proper matrix multiplication
            auto* x_reshaped = ggml_reshape_3d(ctx, x, E, W_cur * H_cur * B_cur, 1);
            auto* cur = ggml_mul_mat(ctx, qkv_w_tensor, x_reshaped);
            cur = ggml_add(ctx, cur, qkv_b_tensor);

            // Reshape and permute for Q, K, V
            // cur is now [3*E, W_cur*H_cur*B_cur, 1], need to reshape to [E, 3, W_cur*H_cur, B_cur]
            cur = ggml_reshape_4d(ctx, cur, E, 3, W_cur * H_cur, B_cur);
            cur = ggml_cont(ctx, ggml_permute(ctx, cur, 0, 3, 1, 2));
            // cur: [E, W*H, B_cur, 3]  (ne[3]=3 separates Q/K/V)

            auto* Q = ggml_view_3d(ctx, cur, E, W_cur * H_cur, B_cur,
                                   cur->nb[1], cur->nb[2], 0);
            auto* K = ggml_view_3d(ctx, cur, E, W_cur * H_cur, B_cur,
                                   cur->nb[1], cur->nb[2], 1 * cur->nb[3]);
            auto* V = ggml_view_3d(ctx, cur, E, W_cur * H_cur, B_cur,
                                   cur->nb[1], cur->nb[2], 2 * cur->nb[3]);

            Q = ggml_reshape_4d(ctx, Q, HD, NH, W_cur * H_cur, B_cur);
            Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));
            Q = ggml_reshape_3d(ctx, Q, HD, W_cur * H_cur, NH * B_cur);

            K = ggml_reshape_4d(ctx, K, HD, NH, W_cur * H_cur, B_cur);
            K = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));
            K = ggml_reshape_3d(ctx, K, HD, W_cur * H_cur, NH * B_cur);

            V = ggml_reshape_4d(ctx, V, HD, NH, W_cur * H_cur, B_cur);
            V = ggml_permute(ctx, V, 0, 2, 1, 3);

            // RoPE
            freqs_cis_tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, HD/2, N);
            ggml_set_input(freqs_cis_tensor);

            if (component == "rope") {
                Q = sam3_apply_rope_test(ctx, Q, freqs_cis_tensor);
                K = sam3_apply_rope_test(ctx, K, freqs_cis_tensor);

                Q = ggml_reshape_4d(ctx, Q, HD, W_cur * H_cur, NH, B_cur);
                K = ggml_reshape_4d(ctx, K, HD, W_cur * H_cur, NH, B_cur);

                ggml_set_name(Q, "rope_Q_output");
                ggml_set_name(K, "rope_K_output");
                ggml_set_output(Q);
                ggml_set_output(K);
                x = Q;
            } else {
                if (is_global) {
                    Q = sam3_apply_rope_test(ctx, Q, freqs_cis_tensor);
                    K = sam3_apply_rope_test(ctx, K, freqs_cis_tensor);
                }
                Q = ggml_reshape_4d(ctx, Q, HD, W_cur * H_cur, NH, B_cur);
                K = ggml_reshape_4d(ctx, K, HD, W_cur * H_cur, NH, B_cur);

                // Flash Attention
                if (component == "flash_attn") {
                    float scale = 1.0f / sqrtf((float)HD);
                    auto* attn_out = ggml_flash_attn_ext(ctx, Q, K, V, nullptr, scale, 0.0f, 0.0f);
                    x = ggml_reshape_4d(ctx, attn_out, E, W_cur, H_cur, B_cur);
                    ggml_set_name(x, "flash_attn_output");
                    ggml_set_output(x);
                } else {
                    float scale = 1.0f / sqrtf((float)HD);
                    auto* attn_out = ggml_flash_attn_ext(ctx, Q, K, V, nullptr, scale, 0.0f, 0.0f);
                    x = ggml_reshape_4d(ctx, attn_out, E, W_cur, H_cur, B_cur);

                    // Output projection
                    proj_w_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, E, E);
                    proj_b_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, E);
                    ggml_set_input(proj_w_tensor);
                    ggml_set_input(proj_b_tensor);

                    if (component == "proj") {
                        // Reshape x to 3D: [E, W_cur*H_cur*B_cur, 1] for proper matrix multiplication
                        auto* x_reshaped = ggml_reshape_3d(ctx, x, E, W_cur * H_cur * B_cur, 1);
                        x = ggml_mul_mat(ctx, proj_w_tensor, x_reshaped);
                        x = ggml_add(ctx, x, proj_b_tensor);
                        // Reshape back to 4D
                        x = ggml_reshape_4d(ctx, x, E, W_cur, H_cur, B_cur);
                        ggml_set_name(x, "proj_output");
                        ggml_set_output(x);
                    } else {
                        // Reshape x to 3D: [E, W_cur*H_cur*B_cur, 1] for proper matrix multiplication
                        auto* x_reshaped = ggml_reshape_3d(ctx, x, E, W_cur * H_cur * B_cur, 1);
                        x = ggml_mul_mat(ctx, proj_w_tensor, x_reshaped);
                        x = ggml_add(ctx, x, proj_b_tensor);
                        // Reshape back to 4D
                        x = ggml_reshape_4d(ctx, x, E, W_cur, H_cur, B_cur);

                        if (!is_global) {
                            x = ggml_win_unpart(ctx, x, w0, h0, WS);
                        }

                        x = ggml_add(ctx, shortcut, x);

                        shortcut = x;

                        // Second LayerNorm
                        norm2_w_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, E);
                        norm2_b_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, E);
                        ggml_set_input(norm2_w_tensor);
                        ggml_set_input(norm2_b_tensor);

                        if (component == "norm2") {
                            x = sam3_layer_norm_test(ctx, x, norm2_w_tensor, norm2_b_tensor);
                            ggml_set_name(x, "norm2_output");
                            ggml_set_output(x);
                        } else {
                            x = sam3_layer_norm_test(ctx, x, norm2_w_tensor, norm2_b_tensor);

                            // MLP
                            mlp_fc1_w_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, mlp_dim, E);
                            mlp_fc1_b_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, mlp_dim);
                            mlp_fc2_w_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, E, mlp_dim);
                            mlp_fc2_b_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, E);
                            ggml_set_input(mlp_fc1_w_tensor);
                            ggml_set_input(mlp_fc1_b_tensor);
                            ggml_set_input(mlp_fc2_w_tensor);
                            ggml_set_input(mlp_fc2_b_tensor);

                            if (component == "mlp") {
                                // Reshape x to 3D: [E, W_cur*H_cur*B_cur, 1] for proper matrix multiplication
                                auto* x_reshaped = ggml_reshape_3d(ctx, x, E, W_cur * H_cur * B_cur, 1);
                                x = ggml_mul_mat(ctx, mlp_fc1_w_tensor, x_reshaped);
                                x = ggml_add(ctx, x, mlp_fc1_b_tensor);
                                x = ggml_gelu_erf(ctx, x);
                                // Reshape back to 3D for second layer: [mlp_dim, W_cur*H_cur*B_cur, 1]
                                x = ggml_reshape_3d(ctx, x, mlp_dim, W_cur * H_cur * B_cur, 1);
                                x = ggml_mul_mat(ctx, mlp_fc2_w_tensor, x);
                                x = ggml_add(ctx, x, mlp_fc2_b_tensor);
                                // Reshape back to 4D
                                x = ggml_reshape_4d(ctx, x, E, W_cur, H_cur, B_cur);
                                ggml_set_name(x, "mlp_output");
                                ggml_set_output(x);
                            } else {
                                // Reshape x to 3D: [E, W_cur*H_cur*B_cur, 1] for proper matrix multiplication
                                auto* x_reshaped = ggml_reshape_3d(ctx, x, E, W_cur * H_cur * B_cur, 1);
                                x = ggml_mul_mat(ctx, mlp_fc1_w_tensor, x_reshaped);
                                x = ggml_add(ctx, x, mlp_fc1_b_tensor);
                                x = ggml_gelu_erf(ctx, x);
                                // Reshape back to 3D for second layer: [mlp_dim, W_cur*H_cur*B_cur, 1]
                                x = ggml_reshape_3d(ctx, x, mlp_dim, W_cur * H_cur * B_cur, 1);
                                x = ggml_mul_mat(ctx, mlp_fc2_w_tensor, x);
                                x = ggml_add(ctx, x, mlp_fc2_b_tensor);
                                // Reshape back to 4D
                                x = ggml_reshape_4d(ctx, x, E, W_cur, H_cur, B_cur);
                                x = ggml_add(ctx, shortcut, x);
                                ggml_set_name(x, "block_output");
                                ggml_set_output(x);
                            }
                        }
                    }
                }
            }
        }
    }

    // 构建和分配 graph
    auto* graph = ggml_new_graph_custom(ctx, 4096, false);
    ggml_build_forward_expand(graph, x);

    auto* galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_reserve(galloc, graph) ||
        !ggml_gallocr_alloc_graph(galloc, graph)) {
        fprintf(stderr, "Failed to allocate graph\n");
        ggml_gallocr_free(galloc);
        ggml_free(ctx);
        return false;
    }

    // 设置输入
    ggml_backend_tensor_set(x_tensor, input.data(), 0, input.size() * sizeof(float));
    ggml_backend_tensor_set(norm1_w_tensor, weights.norm1_w.data(), 0, weights.norm1_w.size() * sizeof(float));
    ggml_backend_tensor_set(norm1_b_tensor, weights.norm1_b.data(), 0, weights.norm1_b.size() * sizeof(float));

    if (component != "layernorm") {
        ggml_backend_tensor_set(qkv_w_tensor, weights.qkv_w.data(), 0, weights.qkv_w.size() * sizeof(float));
        ggml_backend_tensor_set(qkv_b_tensor, weights.qkv_b.data(), 0, weights.qkv_b.size() * sizeof(float));
    }

    if ((component == "rope" || (is_global && component != "layernorm" && component != "qkv")) && freqs_cis_tensor) {
        ggml_backend_tensor_set(freqs_cis_tensor, weights.freqs_cis.data(), 0, weights.freqs_cis.size() * sizeof(float));
    }

    if ((component == "proj" || component == "norm2" || component == "mlp" || component == "full") && proj_w_tensor && proj_b_tensor) {
        ggml_backend_tensor_set(proj_w_tensor, weights.proj_w.data(), 0, weights.proj_w.size() * sizeof(float));
        ggml_backend_tensor_set(proj_b_tensor, weights.proj_b.data(), 0, weights.proj_b.size() * sizeof(float));
    }

    if ((component == "norm2" || component == "mlp" || component == "full") && norm2_w_tensor && norm2_b_tensor) {
        ggml_backend_tensor_set(norm2_w_tensor, weights.norm2_w.data(), 0, weights.norm2_w.size() * sizeof(float));
        ggml_backend_tensor_set(norm2_b_tensor, weights.norm2_b.data(), 0, weights.norm2_b.size() * sizeof(float));
    }

    if ((component == "mlp" || component == "full") && mlp_fc1_w_tensor && mlp_fc1_b_tensor && mlp_fc2_w_tensor && mlp_fc2_b_tensor) {
        ggml_backend_tensor_set(mlp_fc1_w_tensor, weights.mlp_fc1_w.data(), 0, weights.mlp_fc1_w.size() * sizeof(float));
        ggml_backend_tensor_set(mlp_fc1_b_tensor, weights.mlp_fc1_b.data(), 0, weights.mlp_fc1_b.size() * sizeof(float));
        ggml_backend_tensor_set(mlp_fc2_w_tensor, weights.mlp_fc2_w.data(), 0, weights.mlp_fc2_w.size() * sizeof(float));
        ggml_backend_tensor_set(mlp_fc2_b_tensor, weights.mlp_fc2_b.data(), 0, weights.mlp_fc2_b.size() * sizeof(float));
    }

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
    output.clear();
    size_t output_size = x->ne[0] * x->ne[1] * x->ne[2] * x->ne[3];
    output.resize(output_size);
    ggml_backend_tensor_get(x, output.data(), 0, output.size() * sizeof(float));

    ggml_gallocr_free(galloc);
    ggml_free(ctx);

    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    fprintf(stdout, "[%s] %s compute time: %lld ms\n", backend_name.c_str(), component.c_str(), ms);

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
    fprintf(stdout, "  --block <idx>     Block index to test (0-31)\n");
    fprintf(stdout, "  --component <name> Component to test:\n");
    fprintf(stdout, "                    layernorm, qkv, rope, flash_attn, proj, norm2, mlp, full\n");
    fprintf(stdout, "  --output <file>   Output file for results\n");
    fprintf(stdout, "  --compare <b1> <b2> Compare two backends\n");
    fprintf(stdout, "  --help             Show this help message\n");
}

int main(int argc, char* argv[]) {
    std::string backend1 = "cpu";
    std::string backend2;
    std::string output_file;
    bool compare_mode = false;
    int block_idx = 15;
    std::string component = "full";

    // 解析命令行参数
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_help(argv[0]);
            return 0;
        } else if (arg == "--backend" && i + 1 < argc) {
            backend1 = argv[++i];
        } else if (arg == "--block" && i + 1 < argc) {
            block_idx = atoi(argv[++i]);
        } else if (arg == "--component" && i + 1 < argc) {
            component = argv[++i];
        } else if (arg == "--compare" && i + 2 < argc) {
            compare_mode = true;
            backend1 = argv[++i];
            backend2 = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_file = argv[++i];
        }
    }

    // SAM3 global attention blocks: 7, 15, 23, 31
    bool is_global = (block_idx == 7 || block_idx == 15 || block_idx == 23 || block_idx == 31);
    int window_size = 24;
    int embed_dim = 1024;
    int mlp_dim = 4736;

    fprintf(stdout, "=== SAM3 ViT Block %d (%s Attention) Vulkan vs CPU Test ===\n", block_idx, is_global ? "Global" : "Window");
    fprintf(stdout, "Testing component: %s\n", component.c_str());
    fprintf(stdout, "Backend 1: %s\n", backend1.c_str());
    if (compare_mode) {
        fprintf(stdout, "Backend 2: %s\n", backend2.c_str());
    }
    fprintf(stdout, "\n");

    // 生成测试数据
    int N = 72 * 72;
    std::vector<float> input(embed_dim * N);

    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = (float)(rand() % 100) / 100.0f;
    }

    // 生成权重
    VitBlockWeights weights;
    generate_random_weights(weights, embed_dim, mlp_dim, window_size, is_global && component != "full");

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
    std::vector<float> output1, output2;

    fprintf(stdout, "Running %s test with %s backend...\n", component.c_str(), backend1.c_str());
    if (!test_vit_block_component(backend1_ptr, backend1, input, weights, component,
                                    output1, is_global, embed_dim, mlp_dim, window_size)) {
        return 1;
    }

    TensorStats stats;

    if (compare_mode) {
        fprintf(stdout, "Running %s test with %s backend...\n", component.c_str(), backend2.c_str());
        if (!test_vit_block_component(backend2_ptr, backend2, input, weights, component,
                                        output2, is_global, embed_dim, mlp_dim, window_size)) {
            return 1;
        }

        // 比较结果
        fprintf(stdout, "\n=== Comparing %s vs %s ===\n", backend1.c_str(), backend2.c_str());
        fprintf(stdout, "\n");

        stats = compute_stats(output1, output2);
        print_stats(stats, component.c_str());

        // 判断是否通过
        const float tolerance = 1e-3f;
        bool pass = stats.max_error < tolerance;

        fprintf(stdout, "\n=== Test Result ===\n");
        fprintf(stdout, "%s: %s (max_error = %.6e, tolerance = %.6e)\n",
                component.c_str(), pass ? "PASS" : "FAIL", stats.max_error, tolerance);

        if (pass) {
            fprintf(stdout, "\n✓ %s operations are consistent between backends\n", component.c_str());
            fprintf(stdout, "\nThis component works correctly on Vulkan backend.\n");
        } else {
            fprintf(stdout, "\n✗ %s operations have significant differences between backends\n", component.c_str());
            fprintf(stdout, "\nThe issue is likely in one of these ggml operations:\n");

            if (component == "layernorm") {
                fprintf(stdout, "  - ggml_norm\n");
                fprintf(stdout, "  - ggml_mul\n");
                fprintf(stdout, "  - ggml_add\n");
            } else if (component == "qkv") {
                fprintf(stdout, "  - ggml_mul_mat\n");
                fprintf(stdout, "  - ggml_add\n");
                fprintf(stdout, "  - ggml_reshape_4d\n");
                fprintf(stdout, "  - ggml_cont\n");
                fprintf(stdout, "  - ggml_permute\n");
            } else if (component == "rope") {
                fprintf(stdout, "  - ggml_reshape_4d\n");
                fprintf(stdout, "  - ggml_view_4d (stride calculation)\n");
                fprintf(stdout, "  - ggml_concat\n");
                fprintf(stdout, "  - ggml_mul / ggml_add / ggml_sub\n");
            } else if (component == "flash_attn") {
                fprintf(stdout, "  - ggml_flash_attn_ext\n");
                fprintf(stdout, "  - ggml_reshape_4d\n");
            } else if (component == "proj") {
                fprintf(stdout, "  - ggml_mul_mat\n");
                fprintf(stdout, "  - ggml_add\n");
                fprintf(stdout, "  - ggml_win_unpart (for window attention)\n");
                fprintf(stdout, "  - ggml_add (residual)\n");
            } else if (component == "norm2") {
                fprintf(stdout, "  - ggml_norm\n");
                fprintf(stdout, "  - ggml_mul\n");
                fprintf(stdout, "  - ggml_add\n");
            } else if (component == "mlp") {
                fprintf(stdout, "  - ggml_mul_mat\n");
                fprintf(stdout, "  - ggml_add\n");
                fprintf(stdout, "  - ggml_gelu_erf\n");
            } else if (component == "full") {
                fprintf(stdout, "  - Any of the above operations\n");
                fprintf(stdout, "  - Cumulative numerical errors\n");
            }

            fprintf(stdout, "\nThis indicates a problem with ggml implementation on Vulkan backend.\n");
        }

        return pass ? 0 : 1;
    } else {
        // 单后端模式，只输出统计
        fprintf(stdout, "\n=== %s Output Statistics ===\n", component.c_str());
        fprintf(stdout, "Output: %zu elements\n", output1.size());
        fprintf(stdout, "\nFirst 10 values: ");
        for (int i = 0; i < 10 && i < (int)output1.size(); ++i) {
            fprintf(stdout, "%.6f ", output1[i]);
        }
        fprintf(stdout, "\n");

        // 保存到文件
        if (!output_file.empty()) {
            fprintf(stdout, "\nSaving output to %s\n", output_file.c_str());
            save_tensor(output1, output_file + "_" + component + ".bin");
        }

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
