# 100x Video Tracking Speedup Plan for SAM2 Tiny

## Context

SAM2 Tiny with quantized weights currently takes ~350ms per frame for video object tracking on Apple Silicon with Metal via ggml. That's ~3 FPS -- unusable for real-time applications. The goal is a **100x speedup** to ~3.5ms/frame (~285 FPS), or at minimum a path to real-time 30 FPS (~33ms/frame).

### Current Per-Frame Pipeline (single object)

```
sam3_propagate_frame() [sam3.cpp:9999]
  |
  [1] sam3_encode_image()         [sam3.cpp:4139 -> sam2_encode_image_hiera:3972]
  |     Preprocess (CPU resize 1008x1008 + normalize)
  |     Build Hiera graph (12 blocks for Tiny)
  |     Build FPN neck graph (3 levels, scalp=1)
  |     gallocr_new -> reserve -> alloc -> set inputs -> compute -> tensor_get/set -> free
  |     Copy FPN outputs GPU->CPU->GPU (state.neck_trk[0..2])
  |     Compute sinusoidal PE on CPU, upload to GPU
  |
  [2] sam3_propagate_single()     [sam3.cpp:9152]  (per instance)
  |     Read memory bank tensors GPU->CPU
  |     Build prompt tensors on CPU
  |     Build memory attention graph (4 layers, self-attn + cross-attn + FFN)
  |     Build SAM decoder graph (2 two-way blocks + upscaling)
  |     gallocr_new -> reserve -> alloc -> set 10+ inputs -> compute -> tensor_get -> free
  |
  [3] sam3_encode_memory()        [sam3.cpp:9446]  (per instance)
  |     CPU bilinear: mask 288x288 -> 1008x1008 -> sigmoid -> 1152x1152
  |     Build memory encoder graph (4 conv stages + pixel proj + 2 CxBlocks)
  |     gallocr_new -> reserve -> alloc -> set inputs -> compute -> tensor_get -> free
  |     Allocate new GPU buffer for memory slot, upload
  |
  [4] sam3_extract_obj_ptr_cpu()  [sam3.cpp:7348]  (per instance)
  |     3-layer MLP on CPU
  |     Allocate new GPU buffer for pointer, upload
  |
  [5] Post-processing (CPU)
        Bilinear interpolate 288x288 -> original resolution
        fill_holes, remove_sprinkles, resolve_overlaps
```

### Bottleneck Analysis

| Stage | Estimated Time | % of Total | Notes |
|-------|---------------|------------|-------|
| Image encode (Hiera+FPN) | ~150-200ms | ~50% | 12 Hiera blocks + FPN, run every frame |
| Graph overhead (3 graphs) | ~50-80ms | ~20% | init/build/reserve/alloc/free x3 |
| CPU-GPU copies | ~30-50ms | ~12% | FPN readback, memory bank reads, input copies |
| Memory attention | ~30-40ms | ~10% | 4 layers, 5184 tokens x M_total keys |
| SAM decoder | ~15-25ms | ~6% | 2 two-way blocks + upscaling |
| Memory encoder | ~20-30ms | ~7% | 4 conv stages + fusion |
| Pre/post processing (CPU) | ~10-20ms | ~5% | Resize, normalize, interpolate, BFS |

---

## 100 Ideas for 100x Speedup

### A. Skip Image Encoder on Most Frames (Impact: 5-30x)

1. **Keyframe-only backbone**: Run Hiera+FPN every Nth frame (N=10-30); reuse features for intermediate frames
2. **Optical flow feature warping**: Use Lucas-Kanade/Farneback optical flow to warp cached feature maps to current frame
3. **Feature delta prediction**: Train a tiny CNN (3 layers) to predict the feature delta between frames
4. **Motion-compensated feature reuse**: Use block matching to shift feature map blocks; only re-encode new regions
5. **Conditional re-encoding**: Monitor mask IoU across frames; re-encode only when predicted quality drops
6. **Temporal feature interpolation**: Between keyframes, linearly interpolate backbone features
7. **Background subtraction gating**: Skip backbone when frame-to-frame pixel difference < threshold
8. **Two-speed pipeline**: Full backbone at 1 FPS for accurate features, lightweight propagation at 30 FPS
9. **Feature bank recycling**: Store features from recent keyframes; select most similar for current frame
10. **Deformable feature propagation**: Learned lightweight deformation network warps cached features (~2ms)

### B. Reduce Input Resolution / Spatial Processing (Impact: 2-10x)

11. **Halve input resolution**: 1008x1008 -> 504x504 (4x fewer pixels)
12. **ROI cropping**: Crop 2x bounding box around object -> encode ~300x300 instead of 1008x1008
13. **Pyramid encoding**: Encode at 252x252 globally, refine only near object boundary at 504x504
14. **Adaptive resolution**: Large objects -> low res; small objects -> higher res ROI
15. **Increased patch stride**: Hiera uses 4x4 patches; use 8x8 to quarter spatial tokens
16. **Attention token pruning**: After first Hiera stage, drop tokens far from tracked object
17. **Progressive resolution**: Start at 256x256, increase only if object is small
18. **Spatial attention masking**: Zero out attention for tokens far from predicted object position
19. **Multi-scale tracking**: 256x256 for position estimation, higher only for boundary refinement
20. **Reduced FPN scales**: Use only 2 of 3 FPN levels instead of all 3

### C. Model Architecture Simplification (Impact: 2-5x)

21. **Early exit from Hiera**: Use 6 of 12 blocks (exit after stage 2)
22. **Reduce memory attention**: 4 layers -> 1 layer (with fine-tuning)
23. **Reduce SAM decoder**: 2 layers -> 1 layer
24. **Skip memory encoder**: Store raw backbone features as memory instead of encoding
25. **Replace Hiera with MobileNetV3**: Mobile-grade backbone (~10x faster)
26. **Replace with depthwise separable convolutions**: Avoid transformer overhead
27. **Knowledge distillation**: Distill to a 1M param model specialized for tracking
28. **Halve embedding dimension**: 96 -> 48 in backbone, 256 -> 128 in neck/decoder
29. **Share weights across Hiera stages**: Reduce cache pressure
30. **Remove FPN neck entirely**: Use single-scale backbone output directly

### D. Eliminate CPU-GPU Round-trips (Impact: 1.5-3x)

31. **Merge all sub-graphs into one**: Single ggml graph for entire per-frame pipeline
32. **Keep FPN outputs on GPU**: Don't copy to CPU and back; pass directly to next stage
33. **GPU-resident state management**: Redesign state to avoid ggml dependency chain issue without CPU copies
34. **Unified allocator**: Single gallocr managing buffers across all pipeline stages
35. **Pre-allocated reusable graph buffers**: Allocate once at init, reuse every frame
36. **Batch CPU-GPU transfers**: Pack all input tensors into a single upload command
37. **Apple Silicon unified memory**: Use `ggml_backend_metal_buffer_from_ptr` for zero-copy
38. **Persistent Metal command buffer**: Avoid re-encoding Metal commands each frame
39. **Async tensor uploads**: Upload frame N+1 data while frame N computes
40. **Eliminate PE recomputation**: Cache sinusoidal PE on GPU permanently; never recompute

### E. ggml / Metal Backend Optimization (Impact: 1.5-4x)

41. **Fused LayerNorm+Residual+Activation kernel**: Reduce from 5 ops to 1
42. **Metal Performance Shaders for matmul**: Replace ggml matmul with MPS equivalent
43. **Custom Metal attention kernel**: Single kernel for the complete memory attention pattern
44. **Persistent graph execution**: Build graph once at init, re-execute without rebuild
45. **Pre-compiled Metal pipeline states**: Cache PSOs across frames
46. **Cache gallocr_reserve result**: Skip graph traversal after first frame
47. **Reduce graph node count via fusion**: Fewer Metal dispatches = less overhead
48. **Tune Metal threadgroup sizes**: Optimize for SAM2 Tiny's specific tensor dimensions
49. **Apple Neural Engine (ANE)**: Offload backbone to ANE via CoreML (15+ TOPS)
50. **Metal indirect command buffers**: Batch kernel launches to reduce CPU overhead

### F. Alternative Runtimes (Impact: 5-20x)

51. **MLX framework**: Apple's ML framework, optimized for Apple Silicon unified memory
52. **MPSGraph**: Apple's graph-based Metal compute framework with automatic optimization

### G. Quantization and Precision (Impact: 1.5-3x)

53. **INT8 activations**: Quantize not just weights but intermediate activations
54. **Binary attention masks**: 1-bit attention weights for memory attention
55. **FP16 for all activations**: Currently some computations use FP32
56. **Per-channel quantization**: Better accuracy than per-tensor at same speed
57. **Dynamic quantization**: Quantize on-the-fly based on tensor range
58. **Q4_0 for attention projections**: More aggressive quantization for Q/K/V weights
59. **Sparse quantization**: Skip near-zero weights entirely

### H. Pipeline Parallelism (Impact: 2-4x effective throughput)

60. **Double-buffer frames**: Encode frame N+1 while processing frame N's tracking
61. **Overlap decode and encode**: Start ffmpeg decode during GPU compute
62. **Async mask output**: Display previous mask while computing current
63. **GPU-CPU overlap**: CPU does post-processing while GPU runs next stage
64. **Dual compute**: Use both Metal GPU and ANE simultaneously for different stages
65. **Pipeline Hiera stages**: Stage 1 of frame N+1 starts while stage 4 of frame N finishes
66. **Speculative tracking**: Start memory attention with approximate features before backbone completes
67. **Frame skip + interpolation**: Process every Kth frame, linear-interpolate masks for skipped

### I. Video Decode Optimization (Impact: 1.2-2x)

68. **VideoToolbox**: Hardware H.264/HEVC decode instead of ffmpeg pipe
69. **Decode to Metal texture**: Skip CPU-side RGB conversion
70. **Batch decode**: Decode multiple frames into ring buffer
71. **Process YUV directly**: Skip YUV->RGB conversion; learn on YUV
72. **Pre-decode entire video**: Decode all frames upfront into memory-mapped file
73. **Streaming decode**: Use AVFoundation for zero-copy frame access

### J. Pre/Post-processing Optimization (Impact: 1.1-1.5x)

74. **GPU-side image resize**: Metal compute shader for bilinear resize
75. **GPU-side normalization**: Fold normalize into compute graph as first op
76. **Skip fill_holes/remove_sprinkles**: Cosmetic; skip for speed
77. **Reduce mask output resolution**: 72x72 or 144x144 instead of full resolution
78. **Skip full-res interpolation**: Only upsample for display, not for tracking logic
79. **Bake normalization into patch embedding**: Absorb mean/std into first conv weights

### K. Algorithmic Innovations (Impact: 10-100x for cheap frames)

80. **Affine mask transform**: For rigid objects, predict 2D affine warp of previous mask (~0.5ms)
81. **Bbox tracker + SAM decoder**: Use MOSSE/KCF for bbox, run SAM decoder only inside bbox
82. **Sparse point tracking**: Track 8-16 boundary points, reconstruct mask via contour (~1ms)
83. **Template matching**: NCC on raw pixels for position, warp previous mask (~2ms)
84. **Contour tracking**: Active contours/snakes for boundary evolution (~1ms)
85. **Single-point prompt per frame**: Use object centroid as point prompt, skip memory attention
86. **Feature matching**: BRIEF/ORB descriptors between frames; skip backbone when matched
87. **Mask validity prediction**: Tiny network predicts IoU of warped mask; skip SAM when high

### L. Caching and Memoization (Impact: 1.3-2x)

88. **Graph template caching**: Build ggml graph structure once, stamp new tensors each frame
89. **Allocator reuse**: Don't free gallocr between frames; reuse with same graph
90. **Context pool**: Pool ggml_context objects instead of init/free each frame
91. **Attention KV cache**: Cache K,V from self-attention across frames (like LLM KV cache)
92. **Pre-build prompt tensors**: Cache prompt tensors for common memory bank sizes
93. **Weight prefetch**: Ensure model weights are in L2 cache before graph compute

### M. Application-Level (Impact: 1.2-3x)

94. **Temporal mask smoothing**: If mask IoU > 0.95 with previous, reuse without SAM
95. **Confidence-gated skip**: High-confidence objects skip memory attention, just warp
96. **Object-size shortcuts**: Very large objects -> aggressive downscale (they're easy to track)
97. **Amortized memory encoding**: Encode memory every 5th frame instead of every frame
98. **Lazy memory bank updates**: Only update memory when mask changes significantly
99. **Skip object pointer extraction**: Reuse previous pointer when mask is stable
100. **Adaptive keyframe scheduling**: Use frame difference metrics to trigger re-encoding only when needed

---

## Top 4 Recommendations (Highest Likelihood of 100x)

To reach 100x, we need **multiplicative gains from complementary optimizations**. No single idea gets 100x alone. The following 4, when combined, multiply to 100x+:

---

### #1: Keyframe Backbone + Optical Flow Mask Warping (Ideas 1, 2, 67, 80)

**Expected speedup: 10-30x (the single biggest lever)**

**Core insight**: The Hiera backbone + FPN takes ~50-60% of frame time (~175ms) and produces feature maps that change slowly between consecutive video frames. Running it every frame is pure waste for tracking.

**Implementation**:
- Run full `sam2_encode_image_hiera()` only every K frames (K=10-30, tunable)
- On non-keyframes, compute lightweight optical flow between frame N-1 and frame N using Lucas-Kanade (OpenCV-free implementation, ~1-2ms)
- Warp the previous frame's output mask using the flow field
- **Also skip memory attention + SAM decoder + memory encoder** on non-keyframes -- this is the key insight: if the mask barely changed, there's no need to run ANY neural network
- Monitor warped mask quality using simple heuristics (boundary smoothness, area change rate)
- Trigger a keyframe when quality degrades

**Per non-keyframe cost**: optical flow (~1ms) + mask warp (~0.5ms) + quality check (~0.1ms) = **~2ms**
**Per keyframe cost**: ~350ms (current full pipeline)
**Amortized at K=15**: (350 + 14*2) / 15 = **~25ms** = **14x speedup**

**Files to modify**:
- `sam3.cpp:9999` (`sam3_propagate_frame`): Add keyframe logic
- New function: `sam3_propagate_flow()` for optical flow mask warping
- `sam3.h`: Add keyframe interval to `sam3_tracker_params`

**Risk**: Fast camera motion or object deformation can degrade warped masks. Mitigated by adaptive keyframe triggering.

---

### #2: Persistent Graph + Allocator Reuse Across Frames (Ideas 35, 44, 46, 88-90)

**Expected speedup: 2-4x**

**Core insight**: Every frame, each sub-graph goes through the full lifecycle: `ggml_init` -> graph build -> `gallocr_new` -> `reserve` (full graph traversal) -> `alloc` -> compute -> `free`. The graph structure is **identical every frame** (same tensor shapes, same operations). This overhead is estimated at ~50-80ms per frame (3 graphs x 15-25ms each).

**Implementation**:
- At init time (first frame), build each graph once and cache:
  - The `ggml_context` with all tensor definitions
  - The `ggml_cgraph` structure 
  - The `ggml_gallocr` with completed reservation
- On subsequent frames, skip init/build/reserve/alloc/free -- just:
  - `ggml_backend_tensor_set()` for input tensors
  - `ggml_backend_graph_compute()` to execute
- Store cached graphs in the `sam3_tracker` struct
- Invalidate caches when model or graph dimensions change (rare)

**Caveat**: The memory attention graph's shape depends on `M_total` (number of memory tokens), which changes as the memory bank fills. Solution: pre-allocate for max capacity (`num_maskmem * N + max_obj_ptrs = 7*5184 + 16 = 36,304`), pad with zeros.

**Files to modify**:
- `sam3.cpp`: New struct `sam3_cached_graph` holding ctx, graph, galloc
- `sam3.cpp:3972` (`sam2_encode_image_hiera`): Cache backbone graph
- `sam3.cpp:9152` (`sam3_propagate_single`): Cache mem_attn + decoder graph
- `sam3.cpp:9446` (`sam3_encode_memory`): Cache memory encoder graph
- `sam3.h`: Add cached graph storage to `sam3_tracker`

---

### #3: Merge Sub-Graphs + Eliminate CPU-GPU Data Copies (Ideas 31-34, 37, 40)

**Expected speedup: 2-3x**

**Core insight**: The current architecture mandates 3+ separate ggml sub-graphs per frame with CPU-side data copying between them. Each `ggml_backend_tensor_get()` is a **GPU synchronization barrier** (~5-10ms) and each data copy through CPU wastes bandwidth. On Apple Silicon, CPU and GPU share unified memory -- these copies are completely unnecessary.

**Key waste points**:
1. `sam2_encode_image_hiera:4090-4094`: Reads FPN outputs GPU->CPU->GPU (state.neck_trk)
2. `sam3_propagate_single:9170-9172`: Reads memory bank tensors GPU->CPU
3. `sam3_propagate_single:9216-9218`: Copies state.neck_trk[2] to fresh input via CPU
4. `sam3_encode_memory:9522-9525`: Copies state.neck_trk[2] to fresh input via CPU
5. `sam3_encode_memory:9533`: Reads memory encoder output GPU->CPU for storage

**Implementation**:
- **Phase A**: Restructure state so `neck_trk[i]` tensors are allocated in a persistent backend buffer that survives across graphs (not owned by gallocr). This lets the backbone graph write directly to state tensors that the next graph reads.
- **Phase B**: Use `ggml_backend_tensor_copy()` (GPU-GPU copy) instead of get+set through CPU for memory bank reads.
- **Phase C**: For the "fresh input tensor" problem (CLAUDE.md graph isolation rule), explore: (a) the fresh tensor pattern but with GPU-GPU copy instead of CPU detour, or (b) a secondary persistent buffer pool that's not managed by any gallocr.

**Files to modify**:
- `sam3.cpp:3972-4131` (sam2_encode_image_hiera): Persistent state buffer
- `sam3.cpp:9152-9393` (sam3_propagate_single): GPU-GPU copies
- `sam3.cpp:9446-9593` (sam3_encode_memory): GPU-GPU copies
- `sam3.cpp:739-768` (sam3_state struct): Persistent buffer management

---

### #4: ROI-Based Encoding at Reduced Resolution (Ideas 11, 12, 14, 20)

**Expected speedup: 3-10x on backbone (the most expensive stage)**

**Core insight**: For tracking, the object typically occupies a small fraction of the full 1008x1008 frame. Encoding the entire frame wastes computation on irrelevant background. By cropping a region around the tracked object and encoding only that region, we can dramatically reduce the token count in the backbone.

**Implementation**:
- After first keyframe: compute bounding box of tracked mask
- Expand bbox by 2x (for context), clamp to image boundaries
- Crop the source frame to this ROI, resize to a smaller target (e.g., 504x504 or even 252x252)
- Run Hiera on the smaller crop
- Adjust FPN features to map back to full-frame coordinates (simple offset)
- Memory attention and decoder operate in the crop's coordinate space
- Final mask is placed back into full-frame coordinates

**Example**: Object bbox = 200x200 pixels in a 1008x1008 frame
- ROI = 400x400 (2x bbox) -> resize to 504x504
- Tokens: 504/4 = 126 per side = 15,876 tokens (vs 63,504 at 1008) = **4x fewer**
- But Hiera is hierarchical, so the reduction is even better for later stages

**Combining with keyframe approach**: On keyframes, encode full frame. On intermediate keyframes near the object, use ROI. On non-keyframes, use optical flow.

**Files to modify**:
- `sam3.cpp:9999` (sam3_propagate_frame): Add ROI computation
- `sam3.cpp:3972` (sam2_encode_image_hiera): Accept crop parameters
- `sam3.cpp:9152` (sam3_propagate_single): Coordinate transform for memory
- `sam3.h`: Add ROI params to tracker

---

## Combined Speedup Projection

| Optimization | Speedup | Amortized Frame Time |
|-------------|---------|---------------------|
| Baseline (current) | 1x | 350ms |
| + #1 Keyframe K=15 | 14x | 25ms |
| + #2 Persistent graph | 2x | 12.5ms |
| + #3 Eliminate CPU-GPU copies | 1.5x | 8.3ms |
| + #4 ROI encoding on keyframes | 3x on keyframes | ~5ms |
| **Combined** | **~126x** | **~3ms** |

Ideas #1-#4 yield **~14 x 2 x 1.5 x 3 = ~126x** on the amortized frame time, reaching the 100x target.

## Verification Plan

1. **Baseline measurement**: Add `std::chrono` timing to each pipeline stage in `sam3_propagate_frame`
2. **Per-idea measurement**: After each optimization, measure wall-clock time per frame over 100-frame video
3. **Quality metric**: Compare mask IoU between optimized and baseline over the same video
4. **Regression test**: Track a small object with fast motion to test worst case
5. **Memory usage**: Monitor GPU memory consumption to ensure no leaks from persistent graphs

## Recommended Implementation Order

1. **#2 Persistent graph** (lowest risk, moderate gain, enables #3)
2. **#3 Eliminate copies** (builds on #2's persistent buffers)
3. **#1 Keyframe + flow** (biggest single gain, independent of #2/#3)
4. **#4 ROI encoding** (complements #1 for keyframes)
