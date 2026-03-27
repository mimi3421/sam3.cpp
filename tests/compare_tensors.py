#!/usr/bin/env python3
"""Compare tensors from Python reference vs C++ output.

Handles layout differences between PyTorch (NCHW / [B, H, W, C]) and
ggml (column-major [ne0, ne1, ne2, ne3]) formats.

Usage:
    uv run python tests/compare_tensors.py <ref_dir> <cpp_dir> [atol]
    uv run python tests/compare_tensors.py tests/ref_phase3 tests/cpp_out 1e-4
"""
import numpy as np
import sys, os


def load_tensor(path):
    with open(path + ".shape") as f:
        shape = [int(x) for x in f.read().strip().split(",") if x]
    data = np.fromfile(path + ".bin", dtype=np.float32)
    return data, shape


def compare(name, ref_path, cpp_path, atol=1e-4, transpose_mode=None):
    """Compare two tensors with optional layout transposition.

    transpose_mode:
        None        - compare as-is (same layout)
        "ggml2nchw" - C++ is [C, W, H] (ggml), ref is [1, C, H, W] (PyTorch NCHW)
        "ggml2nhwc" - C++ is [E, W, H] (ggml), ref is [1, H, W, E] (PyTorch NHWC)
    """
    ref_data, ref_shape = load_tensor(ref_path)
    cpp_data, cpp_shape = load_tensor(cpp_path)

    # Reshape both to flat arrays for comparison
    ref = ref_data
    cpp = cpp_data

    if transpose_mode == "ggml2nchw":
        # ggml [C, W, H] → PyTorch [1, C, H, W]
        # ggml stores: c + w*C + h*C*W
        # PyTorch stores: c*H*W + h*W + w
        C, W, H = cpp_shape[0], cpp_shape[1], cpp_shape[2] if len(cpp_shape) > 2 else 1
        cpp_3d = cpp.reshape(H, W, C)  # reading ggml flat as [H, W, C] in C-order
        # Actually ggml is column-major: index = c + w*C + h*C*W
        # In numpy (row-major), this is shape (H, W, C) with strides matching
        # So cpp.reshape(H, W, C) gives us [h, w, c] indexing
        # PyTorch NCHW: ref.reshape(1, C, H, W) or ref.reshape(C, H, W)
        # Strip batch dim from ref if present
        if ref_shape[0] == 1 and len(ref_shape) == 4:
            ref = ref.reshape(ref_shape[1], ref_shape[2], ref_shape[3])  # [C, H, W]
        elif len(ref_shape) == 3:
            ref = ref.reshape(ref_shape[0], ref_shape[1], ref_shape[2])
        # Transpose cpp from [H, W, C] to [C, H, W]
        cpp = cpp_3d.transpose(2, 0, 1).flatten()
        ref = ref.flatten()
    elif transpose_mode == "ggml2nhwc":
        # ggml [E, W, H] → PyTorch [1, H, W, E] (NHWC)
        E, W, H = cpp_shape[0], cpp_shape[1], cpp_shape[2] if len(cpp_shape) > 2 else 1
        cpp_3d = cpp.reshape(H, W, E)  # ggml column-major → [H, W, E]
        # ref is [1, H, W, E] or [H, W, E]
        if ref_shape[0] == 1 and len(ref_shape) == 4:
            ref = ref.reshape(ref_shape[1], ref_shape[2], ref_shape[3])
        ref = ref.flatten()
        cpp = cpp_3d.flatten()

    if ref.size != cpp.size:
        print(f"  FAIL {name:40s}  SIZE MISMATCH ref={ref.size} cpp={cpp.size} "
              f"ref_shape={ref_shape} cpp_shape={cpp_shape}")
        return False

    diff = np.abs(ref - cpp)
    max_d = diff.max()
    mean_d = diff.mean()
    cos = np.dot(ref, cpp) / (np.linalg.norm(ref) * np.linalg.norm(cpp) + 1e-12)
    status = "PASS" if max_d < atol else "FAIL"
    print(f"  {status} {name:40s}  max={max_d:.6e}  mean={mean_d:.6e}  cos={cos:.8f}  "
          f"ref_shape={ref_shape}  cpp_shape={cpp_shape}")
    if status == "FAIL":
        worst = np.argmax(diff)
        print(f"         worst flat_idx={worst}  ref={ref[worst]:.6e}  cpp={cpp[worst]:.6e}")
        # Print some stats
        print(f"         ref: min={ref.min():.6e} max={ref.max():.6e} mean={ref.mean():.6e}")
        print(f"         cpp: min={cpp.min():.6e} max={cpp.max():.6e} mean={cpp.mean():.6e}")
    return status == "PASS"


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <ref_dir> <cpp_dir> [atol]")
        sys.exit(1)

    ref_dir = sys.argv[1]
    cpp_dir = sys.argv[2]
    atol = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-4

    # Compare all tensors that exist in both directories
    ref_tensors = {f[:-4] for f in os.listdir(ref_dir) if f.endswith(".bin")}
    cpp_tensors = {f[:-4] for f in os.listdir(cpp_dir) if f.endswith(".bin")}
    common = sorted(ref_tensors & cpp_tensors)
    if not common:
        print("No common tensors found!")
        print(f"  ref has: {sorted(ref_tensors)[:10]}")
        print(f"  cpp has: {sorted(cpp_tensors)[:10]}")
        sys.exit(1)

    n_pass = 0
    for name in common:
        if compare(name, os.path.join(ref_dir, name), os.path.join(cpp_dir, name), atol):
            n_pass += 1
    print(f"\n{n_pass}/{len(common)} passed (atol={atol})")
