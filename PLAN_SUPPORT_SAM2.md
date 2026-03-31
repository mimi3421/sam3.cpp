# SAM2 Support in sam3.cpp — Complete Implementation Plan

> Extend sam3.cpp to simultaneously support SAM2 and SAM3 forward passes for
> image and video segmentation, reusing as much code as possible.

---

## Table of Contents

1. [Goal & Design Principles](#1-goal--design-principles)
2. [Architecture Comparison](#2-architecture-comparison)
3. [Component Reuse Analysis](#3-component-reuse-analysis)
4. [Binary Weight Format (SAM2)](#4-binary-weight-format-sam2)
5. [Python Weight Conversion Script](#5-python-weight-conversion-script)
6. [New & Modified Structs](#6-new--modified-structs)
7. [New & Modified Functions](#7-new--modified-functions)
8. [Model Loading](#8-model-loading)
9. [Image Encoder: Hiera Backbone](#9-image-encoder-hiera-backbone)
10. [Image Encoder: FPN Neck](#10-image-encoder-fpn-neck)
11. [SAM Decoder Path (Shared)](#11-sam-decoder-path-shared)
12. [Video Tracking Path (Shared)](#12-video-tracking-path-shared)
13. [Public API Changes](#13-public-api-changes)
14. [Tensor Name Mapping (SAM2)](#14-tensor-name-mapping-sam2)
15. [Implementation Order](#15-implementation-order)
16. [Appendix: SAM2 Tensor Shape Reference](#16-appendix-sam2-tensor-shape-reference)
17. [Appendix: SAM2 Hyperparameters by Variant](#17-appendix-sam2-hyperparameters-by-variant)

---

## 1. Goal & Design Principles

### What We're Adding

SAM2 (Segment Anything Model 2) is a predecessor to SAM3 with an almost
identical tracker path but a fundamentally different backbone (Hiera instead
of ViT) and no text/detector path.  SAM2 supports:

- **Image segmentation** via point/box prompts (PVS — identical to SAM3's PVS)
- **Video tracking** with memory bank + mask propagation (identical mechanism)

SAM2 does **not** have: text encoder, fusion encoder, DETR decoder, geometry
encoder, segmentation head, or text-prompted PCS.

### Design Principles

1. **No if-else divergence inside functions.**  When two implementations differ,
   use the `sam2_` prefix for SAM2-specific functions and `sam3_` for SAM3-specific
   ones.  Shared functions keep the existing `sam3_` prefix (they are library
   functions, not SAM3-specific).

2. **Maximum code reuse.**  The tracker path (SAM decoder, memory attention,
   memory encoder, object pointer) is architecturally identical between SAM2
   and SAM3.  These functions are reused without modification.

3. **Same file.**  Everything stays in `sam3.cpp` + `sam3.h`.  SAM2-specific
   structs and functions are added alongside existing SAM3 code.

4. **Model type auto-detection.**  `sam3_load_model()` reads the binary header
   magic number to distinguish SAM2 from SAM3 files.  The loaded model carries
   a `model_type` field.  Top-level API functions dispatch to the correct
   backbone encoder; everything downstream is shared.

5. **Separate conversion script.**  `convert_sam2_to_ggml.py` converts SAM2
   PyTorch checkpoints.  It uses a different magic number (`0x73616D32`, "sam2").

---

## 2. Architecture Comparison

```
┌────────────────────────────────────────────────────────────────────────┐
│                  SAM3 Architecture                                      │
│                                                                        │
│  ┌─────────────┐   ┌──────────┐   ┌────────────┐   ┌────────────┐    │
│  │ ViT Backbone │──▶│SimpleFPN │──▶│  Fusion    │──▶│   DETR     │    │
│  │ (32 blocks)  │   │ (det+trk)│   │  Encoder   │   │  Decoder   │    │
│  │ 1024-dim     │   │ 256-dim  │   │ 6 layers   │   │ 6 layers   │    │
│  └──────┬───────┘   └────┬─────┘   └────────────┘   └────────────┘    │
│         │                │                                              │
│         │           ┌────┴─────┐                                        │
│         │           │SAM2 Neck │─── ▶ [Tracker: MemAttn→SAM→MemEnc]    │
│         │           └──────────┘                                        │
└────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────┐
│                  SAM2 Architecture                                      │
│                                                                        │
│  ┌─────────────┐   ┌──────────┐                                        │
│  │   Hiera     │──▶│  FpnNeck │─── ▶ [Tracker: MemAttn→SAM→MemEnc]    │
│  │ (4 stages)  │   │ (256-dim)│                                        │
│  │ 96-1152 dim │   │ nearest  │                                        │
│  └─────────────┘   └──────────┘                                        │
│                                                                        │
│  No text encoder.  No fusion encoder.  No DETR decoder.                │
│  No geometry encoder.  No segmentation head.                           │
└────────────────────────────────────────────────────────────────────────┘
```

### Side-by-Side Component Table

| Component | SAM3 | SAM2 | Shared? |
|---|---|---|---|
| **Backbone** | ViT (32 blocks, 1024-dim, 14×14 patches) | Hiera (4 stages, variable dim, 7×7 patches stride 4) | ❌ |
| **Neck** | SimpleFPN (ConvTranspose upsample) | FpnNeck (nearest upsample, scalp=1) | ❌ |
| **Text Encoder** | CLIP-like (24 layers) | — | SAM3 only |
| **Fusion Encoder** | 6 layers | — | SAM3 only |
| **DETR Decoder** | 6 layers, 200 queries | — | SAM3 only |
| **Geometry Encoder** | 3 layers | — | SAM3 only |
| **Segmentation Head** | MaskFormer | — | SAM3 only |
| **Prompt Encoder** | Random Fourier PE, 4 point embeds | Identical | ✅ |
| **Mask Decoder** | 2-layer TwoWay, 4 masks | Identical | ✅ |
| **Memory Encoder** | MaskDownSampler + 2 CXBlock + out_proj | Identical | ✅ |
| **Memory Attention** | 4 layers, RoPE, kv_dim=64 | Identical | ✅ |
| **Object Pointer** | 3-layer MLP + temporal PE | Identical | ✅ |
| **Post-processing** | NMS, fill holes, etc. | Identical | ✅ |

---

## 3. Component Reuse Analysis

### Fully Reused (zero changes)

These functions work identically for both SAM2 and SAM3:

| Function | Purpose |
|---|---|
| `sam3_build_sam_pe()` | SAM prompt encoder (points/boxes/masks) |
| `sam3_build_mem_attn_graph()` | Memory attention (4-layer RoPE transformer) |
| `sam3_propagate_frame()` | Visual-only video propagation |
| `sam3_tracker_add_instance()` | Add instance from PVS prompts |
| `sam3_refine_instance()` | Interactive click refinement |
| `sam3_apply_rope()` | Rotary position embedding |
| `sam3_sinusoidal_pe_2d()` | 2D sinusoidal positional encoding |
| `sam3_get_1d_sine_pe()` | 1D sinusoidal PE for temporal encoding |
| `sam3_twoway_block_forward()` | TwoWay attention block (shared architecture) |
| `sam3_sam_attention()` | Scaled attention with optional downsampling |
| `sam3_nms()` | Non-maximum suppression |
| `sam3_fill_holes()` | Hole filling |
| `sam3_remove_sprinkles()` | Small component removal |
| `sam3_bilinear_interpolate()` | Mask upsampling |
| `sam3_stability_score()` | Dynamic mask selection |
| `sam3_layer_norm()` | LayerNorm op |
| `sam3_layer_norm_2d()` | LayerNorm2d op |
| `sam3_mlp()` / `sam3_mlp_3layer()` | MLP building blocks |
| `sam3_conv2d()` | Conv2d wrapper |
| `sam3_conv_transpose2d()` | ConvTranspose2d wrapper |
| `sam3_load_image()` / `sam3_save_mask()` | Image I/O |
| `sam3_decode_video_frame()` / `sam3_get_video_info()` | Video I/O |

### Fully Reused Structs (zero changes)

| Struct | Fields |
|---|---|
| `sam3_sam_prompt_enc` | pe_gaussian, point_embed[4], not_a_point_embed, no_mask_embed, mask_ds_* |
| `sam3_sam_attn` | q_w/b, k_w/b, v_w/b, out_w/b |
| `sam3_twoway_block` | self_attn, ca_tok2img, ca_img2tok, norms, mlp |
| `sam3_sam_mask_dec` | iou_token, mask_tokens, obj_score_token, twoway_blocks, final_attn, upscale, conv_s0/s1, hyper, iou_head, obj_head |
| `sam3_mem_attn_layer` | sa_q/k/v/out, ca_q/k/v/out, ffn, norms |
| `sam3_mem_attn` | layers vector |
| `sam3_mem_enc` | ds_conv/norm, pix_proj, fuser_dw/norm/fc1/fc2/gamma, out_proj, tpos |
| `sam3_memory_slot` | spatial_feats, spatial_pe, frame_index, is_cond_frame |
| `sam3_masklet` | instance_id, first_frame, last_seen, score, mask_logits, obj_ptr |

### Reused with Parameterization (minor changes)

| Function | Change needed |
|---|---|
| `sam3_build_sam_dec_graph()` | SAM2 variant needed for `pred_obj_scores=False` (5-token layout) and `iou_prediction_use_sigmoid` flag. For SAM2.1 configs where `pred_obj_scores=True`, existing code works. |
| `sam3_encode_memory()` | Parameterize sigmoid_scale/bias from hparams. Parameterize mask interpolation target: `feat_size × 16` (1152 for SAM3, 1024 for SAM2). Parameterize pix_feat source index. |
| `sam3_propagate_single()` | Read spatial size from tensor shapes (already done in current code). Verify empty-memory (first frame) handling. |
| `sam3_segment_pvs()` | Coordinate normalization must use correct image_size (1024 vs 1008). |
| `sam3_preprocess_image()` | SAM2 needs ImageNet normalization; create `sam2_preprocess_image()` instead. |

### New for SAM2 (backbone + neck only)

| Component | Purpose |
|---|---|
| `sam2_hiera_block` struct | MultiScaleBlock weights |
| `sam2_hiera` struct | Full Hiera backbone weights |
| `sam2_fpn_level` struct | Per-level FPN lateral conv |
| `sam2_fpn_neck` struct | FpnNeck weights |
| `sam2_build_hiera_graph()` | Build Hiera backbone ggml graph |
| `sam2_hiera_block_forward()` | Single MultiScaleBlock forward |
| `sam2_multiscale_attention()` | Attention with optional Q pooling |
| `sam2_window_partition()` | Window partition for variable window sizes |
| `sam2_window_unpartition()` | Window unpartition |
| `sam2_build_fpn_neck_graph()` | Build FPN neck ggml graph |
| `sam2_encode_image_hiera()` | Full SAM2 image encoding pipeline |
| `sam2_hiera_pos_embed()` | Hiera windowed+background positional embedding |

### SAM3-Only (existing, completely untouched)

All text encoder, fusion encoder, DETR decoder, geometry encoder, segmentation
head structs and functions remain SAM3-only.  They are never invoked when a
SAM2 model is loaded.

---

## 4. Binary Weight Format (SAM2)

### Magic & Header

SAM2 files use magic `0x73616D32` ("sam2") to distinguish from SAM3 (`0x73616D33`).

```
┌─────────────────────────────────────────────┐
│                  FILE HEADER                 │
├─────────────────────────────────────────────┤
│ [4 bytes]  magic: 0x73616D32 ("sam2")       │
│ [4 bytes]  version: 1                        │
│ [4 bytes]  ftype: 0=f32, 1=f16              │
│ [4 bytes]  n_tensors                         │
│                                              │
│ === Hyperparameters block ===                │
│ [4 bytes]  image_size: 1024                  │
│ [4 bytes]  backbone_type: 1 (hiera)          │
│                                              │
│ [4 bytes]  hiera_embed_dim: 144 (or 96/112)  │
│ [4 bytes]  hiera_num_heads: 2 (or 1)         │
│ [4 bytes]  hiera_num_stages: 4               │
│ [4 bytes]  hiera_stages[0]: 2                │
│ [4 bytes]  hiera_stages[1]: 6                │
│ [4 bytes]  hiera_stages[2]: 36               │
│ [4 bytes]  hiera_stages[3]: 4                │
│ [4 bytes]  hiera_global_att_n: 3             │
│ [4 bytes]  hiera_global_att_idx[0]: 23       │
│ [4 bytes]  hiera_global_att_idx[1]: 33       │
│ [4 bytes]  hiera_global_att_idx[2]: 43       │
│ [4 bytes]  hiera_q_pool: 3                   │
│ [4 bytes]  hiera_window_spec[0]: 8           │
│ [4 bytes]  hiera_window_spec[1]: 4           │
│ [4 bytes]  hiera_window_spec[2]: 16          │
│ [4 bytes]  hiera_window_spec[3]: 8           │
│ [4 bytes]  hiera_pos_embed_bkg_h: 7         │
│ [4 bytes]  hiera_pos_embed_bkg_w: 7         │
│ [4 bytes]  scalp: 1                          │
│                                              │
│ [4 bytes]  neck_dim: 256                     │
│ [4 bytes]  fpn_top_down_levels_n: 2          │
│ [4 bytes]  fpn_top_down_levels[0]: 2         │
│ [4 bytes]  fpn_top_down_levels[1]: 3         │
│                                              │
│ [4 bytes]  sam_embed_dim: 256                │
│ [4 bytes]  sam_dec_depth: 2                  │
│ [4 bytes]  sam_n_multimask: 3                │
│ [4 bytes]  sam_iou_head_depth: 3             │
│                                              │
│ [4 bytes]  mem_out_dim: 64                   │
│ [4 bytes]  mem_attn_layers: 4                │
│ [4 bytes]  num_maskmem: 7                    │
│ [4 bytes]  max_obj_ptrs: 16                  │
│                                              │
│ [4 bytes]  sigmoid_scale_for_mem_enc_x100:   │
│            2000 (= 20.0)                     │
│ [4 bytes]  sigmoid_bias_for_mem_enc_x100:    │
│            -1000 (= -10.0)                   │
│                                              │
│ [4 bytes]  use_high_res_features: 1          │
│ [4 bytes]  use_obj_ptrs_in_encoder: 1        │
│ [4 bytes]  pred_obj_scores: 1                │
│ [4 bytes]  use_multimask_token_for_obj_ptr: 1│
│ [4 bytes]  directly_add_no_mem_embed: 1      │
│ [4 bytes]  non_overlap_masks_for_mem_enc: 1  │
│ [4 bytes]  binarize_mask_from_pts: 0         │
│ [4 bytes]  multimask_output_for_tracking: 1  │
│ [4 bytes]  multimask_min_pt_num: 0           │
│ [4 bytes]  multimask_max_pt_num: 1           │
│ [4 bytes]  fixed_no_obj_ptr: 1               │
│ [4 bytes]  iou_prediction_use_sigmoid: 1     │
│ [4 bytes]  use_mask_input_as_output: 1       │
│ [4 bytes]  multimask_output_in_sam: 1        │
│                                              │
├─────────────────────────────────────────────┤
│              TENSOR RECORDS                  │
│  (same format as SAM3 — n_dims, name,       │
│   dtype, shape reversed, 32-byte aligned)    │
└─────────────────────────────────────────────┘
```

**Key differences from SAM3 header:**

- Different magic (`sam2` vs `sam3`)
- Backbone hyperparameters describe Hiera (stages, window_spec, q_pool) instead of ViT (depth, window_size)
- No text/fusion/DETR/geom hyperparameters
- Additional SAM2-specific flags (sigmoid_scale, use_high_res, pred_obj_scores)

---

## 5. Python Weight Conversion Script

### `convert_sam2_to_ggml.py`

```python
#!/usr/bin/env python3
"""Convert SAM2 PyTorch checkpoint to ggml binary format."""

MAGIC = 0x73616D32  # "sam2"
VERSION = 1
```

### Key Tensor Name Mappings (PyTorch → ggml)

```
# ── Hiera Backbone ──
image_encoder.trunk.patch_embed.proj.weight        → hiera.patch_embed.weight
image_encoder.trunk.patch_embed.proj.bias           → hiera.patch_embed.bias
image_encoder.trunk.pos_embed                       → hiera.pos_embed
image_encoder.trunk.pos_embed_window                → hiera.pos_embed_window

# Hiera blocks: flat ModuleList, indexed 0..total_blocks-1.
# QKV is a SINGLE fused projection: nn.Linear(dim_in, 3*dim_out).
# MLP uses nn.Linear with keys .fc1 and .fc2 (from sam2_utils.MLP .layers.0 / .layers.1).
image_encoder.trunk.blocks.{i}.norm1.weight         → hiera.blocks.{i}.norm1.weight
image_encoder.trunk.blocks.{i}.norm1.bias           → hiera.blocks.{i}.norm1.bias
image_encoder.trunk.blocks.{i}.attn.qkv.weight      → hiera.blocks.{i}.attn.qkv.weight
image_encoder.trunk.blocks.{i}.attn.qkv.bias        → hiera.blocks.{i}.attn.qkv.bias
image_encoder.trunk.blocks.{i}.attn.proj.weight     → hiera.blocks.{i}.attn.proj.weight
image_encoder.trunk.blocks.{i}.attn.proj.bias       → hiera.blocks.{i}.attn.proj.bias
image_encoder.trunk.blocks.{i}.norm2.weight         → hiera.blocks.{i}.norm2.weight
image_encoder.trunk.blocks.{i}.norm2.bias           → hiera.blocks.{i}.norm2.bias
image_encoder.trunk.blocks.{i}.mlp.layers.0.weight  → hiera.blocks.{i}.mlp.fc1.weight
image_encoder.trunk.blocks.{i}.mlp.layers.0.bias    → hiera.blocks.{i}.mlp.fc1.bias
image_encoder.trunk.blocks.{i}.mlp.layers.1.weight  → hiera.blocks.{i}.mlp.fc2.weight
image_encoder.trunk.blocks.{i}.mlp.layers.1.bias    → hiera.blocks.{i}.mlp.fc2.bias

# Stage transition blocks have an extra dim projection + the block's attention
# module gets a q_pool MaxPool2d (no weights — just an op).
# Projection: nn.Linear(dim_in, dim_out), applied to normed input.
image_encoder.trunk.blocks.{i}.proj.weight          → hiera.blocks.{i}.proj.weight
image_encoder.trunk.blocks.{i}.proj.bias            → hiera.blocks.{i}.proj.bias

# ── FPN Neck ──
# Each conv is wrapped in nn.Sequential with a named "conv" submodule.
# backbone_channel_list=[1152,576,288,144] → convs[0] maps 1152, convs[3] maps 144.
# BUT in FPN forward, convs are accessed as convs[n-i] (reversed).
image_encoder.neck.convs.{i}.conv.weight            → fpn.convs.{i}.weight
image_encoder.neck.convs.{i}.conv.bias              → fpn.convs.{i}.bias
# PositionEmbeddingSine has NO learnable parameters (pure computation).

# ── SAM Prompt Encoder ──
sam_prompt_encoder.point_embeddings.{i}.weight       → sam_pe.point_embeddings.{i}
sam_prompt_encoder.not_a_point_embed.weight           → sam_pe.not_a_point
sam_prompt_encoder.no_mask_embed.weight               → sam_pe.no_mask
sam_prompt_encoder.pe_layer.positional_encoding_gaussian_matrix
                                                     → sam_pe.pe_gaussian
sam_prompt_encoder.mask_downscaling.{j}.weight        → sam_pe.mask_ds.{j}.weight
sam_prompt_encoder.mask_downscaling.{j}.bias          → sam_pe.mask_ds.{j}.bias

# ── SAM Mask Decoder ──
sam_mask_decoder.iou_token.weight                     → sam_dec.iou_token
sam_mask_decoder.mask_tokens.weight                   → sam_dec.mask_tokens
sam_mask_decoder.obj_score_token.weight               → sam_dec.obj_score_token

# TwoWay transformer layers — each layer has separate q/k/v projections
# (NOT fused in_proj like standard PyTorch MultiheadAttention).
# Self-attention: full-dim (256→256), downsample_rate=1
sam_mask_decoder.transformer.layers.{i}.self_attn.q_proj.weight
    → sam_dec.twoway.{i}.self_attn.q.weight
sam_mask_decoder.transformer.layers.{i}.self_attn.k_proj.weight
    → sam_dec.twoway.{i}.self_attn.k.weight
sam_mask_decoder.transformer.layers.{i}.self_attn.v_proj.weight
    → sam_dec.twoway.{i}.self_attn.v.weight
sam_mask_decoder.transformer.layers.{i}.self_attn.out_proj.weight
    → sam_dec.twoway.{i}.self_attn.out.weight
# (same pattern for biases)

# Cross-attention token→image: downsample_rate=2, internal_dim=128
sam_mask_decoder.transformer.layers.{i}.cross_attn_token_to_image.q_proj.*
    → sam_dec.twoway.{i}.ca_tok2img.q.*
sam_mask_decoder.transformer.layers.{i}.cross_attn_token_to_image.k_proj.*
    → sam_dec.twoway.{i}.ca_tok2img.k.*
sam_mask_decoder.transformer.layers.{i}.cross_attn_token_to_image.v_proj.*
    → sam_dec.twoway.{i}.ca_tok2img.v.*
sam_mask_decoder.transformer.layers.{i}.cross_attn_token_to_image.out_proj.*
    → sam_dec.twoway.{i}.ca_tok2img.out.*

# Cross-attention image→token: downsample_rate=2
sam_mask_decoder.transformer.layers.{i}.cross_attn_image_to_token.q_proj.*
    → sam_dec.twoway.{i}.ca_img2tok.q.*
# ... same pattern

# MLP: 2-layer (256→2048→256), uses sam2_utils.MLP with .layers.{0,1}
sam_mask_decoder.transformer.layers.{i}.mlp.layers.0.*
    → sam_dec.twoway.{i}.mlp.fc1.*
sam_mask_decoder.transformer.layers.{i}.mlp.layers.1.*
    → sam_dec.twoway.{i}.mlp.fc2.*

# 4 norms per layer: norm1 (self-attn), norm2 (ca_tok2img), norm3 (mlp), norm4 (ca_img2tok)
sam_mask_decoder.transformer.layers.{i}.norm{j}.*     → sam_dec.twoway.{i}.norm{j}.*

# Final attention: separate q/k/v projections
sam_mask_decoder.transformer.final_attn_token_to_image.q_proj.*
    → sam_dec.final_attn.q.*
sam_mask_decoder.transformer.final_attn_token_to_image.k_proj.*
    → sam_dec.final_attn.k.*
sam_mask_decoder.transformer.final_attn_token_to_image.v_proj.*
    → sam_dec.final_attn.v.*
sam_mask_decoder.transformer.final_attn_token_to_image.out_proj.*
    → sam_dec.final_attn.out.*
sam_mask_decoder.transformer.norm_final_attn.*
    → sam_dec.final_norm.*

# Upscaling: nn.Sequential with indexed layers
# .0 = ConvTranspose2d(256→64), .1 = LayerNorm2d(64), .2 = GELU(no params),
# .3 = ConvTranspose2d(64→32), .4 = GELU(no params)
sam_mask_decoder.output_upscaling.{j}.*               → sam_dec.upscale.{j}.*

# High-res convs (applied ONCE in forward_image, BEFORE tracking)
sam_mask_decoder.conv_s0.*                             → sam_dec.conv_s0.*
sam_mask_decoder.conv_s1.*                             → sam_dec.conv_s1.*

# Hypernetwork MLPs: 4 × MLP(256→256→256→32), each with .layers.{0,1,2}
sam_mask_decoder.output_hypernetworks_mlps.{i}.layers.{j}.*
    → sam_dec.hyper.{i}.{j}.*

# IoU head: MLP(256→256→256→4) with .layers.{0,1,2}
sam_mask_decoder.iou_prediction_head.layers.{j}.*
    → sam_dec.iou_head.{j}.*

# Object score head: MLP(256→256→256→1) when pred_obj_scores_mlp=True
# OR nn.Linear(256→1) when pred_obj_scores_mlp=False
sam_mask_decoder.pred_obj_score_head.layers.{j}.*     → sam_dec.obj_head.{j}.*
# OR (if Linear):
sam_mask_decoder.pred_obj_score_head.*                → sam_dec.obj_head.*

# ── Memory Encoder ──
memory_encoder.mask_downsampler.encoder.{j}.*         → mem_enc.ds.{j}.*
memory_encoder.pix_feat_proj.*                        → mem_enc.pix_feat_proj.*
memory_encoder.fuser.layers.{i}.dwconv.*              → mem_enc.fuser.{i}.dw.*
memory_encoder.fuser.layers.{i}.norm.*                → mem_enc.fuser.{i}.norm.*
memory_encoder.fuser.layers.{i}.pwconv1.*             → mem_enc.fuser.{i}.fc1.*
memory_encoder.fuser.layers.{i}.pwconv2.*             → mem_enc.fuser.{i}.fc2.*
memory_encoder.fuser.layers.{i}.gamma                 → mem_enc.fuser.{i}.gamma
memory_encoder.out_proj.*                             → mem_enc.out_proj.*

# ── Memory Attention ──
# Memory attention layers — self_attn and cross_attn_image are both RoPEAttention
# which uses SEPARATE q/k/v projections (not fused in_proj).
# Self-attn: RoPEAttention(256, 1 head, downsample_rate=1, kv_in_dim=256)
memory_attention.layers.{i}.self_attn.q_proj.*        → mem_attn.layers.{i}.sa.q.*
memory_attention.layers.{i}.self_attn.k_proj.*        → mem_attn.layers.{i}.sa.k.*
memory_attention.layers.{i}.self_attn.v_proj.*        → mem_attn.layers.{i}.sa.v.*
memory_attention.layers.{i}.self_attn.out_proj.*      → mem_attn.layers.{i}.sa.out.*
# Cross-attn: RoPEAttention(256, 1 head, downsample_rate=1, kv_in_dim=64, rope_k_repeat=True)
memory_attention.layers.{i}.cross_attn_image.q_proj.* → mem_attn.layers.{i}.ca.q.*
memory_attention.layers.{i}.cross_attn_image.k_proj.* → mem_attn.layers.{i}.ca.k.*
memory_attention.layers.{i}.cross_attn_image.v_proj.* → mem_attn.layers.{i}.ca.v.*
memory_attention.layers.{i}.cross_attn_image.out_proj.*→ mem_attn.layers.{i}.ca.out.*
memory_attention.layers.{i}.linear1.*                  → mem_attn.layers.{i}.ffn.fc1.*
memory_attention.layers.{i}.linear2.*                  → mem_attn.layers.{i}.ffn.fc2.*
memory_attention.layers.{i}.norm1.*                    → mem_attn.layers.{i}.norm1.*
memory_attention.layers.{i}.norm2.*                    → mem_attn.layers.{i}.norm2.*
memory_attention.layers.{i}.norm3.*                    → mem_attn.layers.{i}.norm3.*
# Final norm: at MemoryAttention TOP level (not inside layers)
memory_attention.norm.*                                → mem_attn.norm.*

# ── SAM2Base top-level parameters ──
# maskmem_tpos_enc is a SINGLE stacked tensor [num_maskmem, 1, 1, mem_dim],
# NOT a ParameterList. Key is exactly "maskmem_tpos_enc" (no indices).
# Indexing: maskmem_tpos_enc[num_maskmem - t_pos - 1] for temporal slot t_pos.
maskmem_tpos_enc                                      → mem_enc.tpos_enc

no_mem_embed                                          → no_mem_embed
no_mem_pos_enc                                        → no_mem_pos_enc

# obj_ptr_proj: nn.Linear if use_mlp_for_obj_ptr_proj=False,
# MLP(.layers.{0,1,2}) if use_mlp_for_obj_ptr_proj=True
obj_ptr_proj.weight / obj_ptr_proj.bias               → obj_ptr_proj.weight / .bias  (Linear)
obj_ptr_proj.layers.{j}.*                             → obj_ptr_proj.{j}.*  (MLP)

no_obj_ptr                                            → no_obj_ptr

# obj_ptr_tpos_proj: Linear projection for temporal PE (if proj_tpos_enc_in_obj_ptrs=True)
obj_ptr_tpos_proj.*                                   → obj_ptr_tpos_proj.*

# mask_downsample: Conv2d(1, 1, 4, 4) for downsampling masks to get obj pointers
mask_downsample.*                                     → trk_mask_ds.*

no_obj_embed_spatial                                  → no_obj_embed_spatial
```

### Tensors to Skip

```
- Any key containing "loss", "criterion", "_dn_", "label_enc"
- Training-only buffers
- Optimizer states
```

---

## 6. New & Modified Structs

### 6.1 Model Type Enum (NEW)

```cpp
enum sam3_model_type {
    SAM3_MODEL_SAM3        = 0,  // Full SAM3 (ViT + detector + tracker)
    SAM3_MODEL_SAM3_VISUAL = 1,  // SAM3 visual-only (ViT + tracker)
    SAM3_MODEL_SAM2        = 2,  // SAM2 (Hiera + tracker)
};
```

### 6.2 SAM2 Hyperparameters (NEW — added to sam3_hparams)

```cpp
struct sam3_hparams {
    // ── Existing SAM3 fields (unchanged) ──
    int32_t img_size;
    int32_t patch_size;       // SAM3: 14, unused for SAM2
    int32_t vit_embed_dim;    // SAM3: 1024, unused for SAM2
    int32_t vit_depth;        // SAM3: 32, unused for SAM2
    int32_t vit_num_heads;    // SAM3: 16, unused for SAM2
    int32_t vit_mlp_dim;      // SAM3: 4736, unused for SAM2
    int32_t vit_window_size;  // SAM3: 24, unused for SAM2
    int32_t n_global_attn;
    int32_t global_attn_idx[8];
    // ... all existing SAM3 fields ...

    // ── NEW: SAM2-specific Hiera backbone fields ──
    sam3_model_type model_type = SAM3_MODEL_SAM3;

    int32_t hiera_embed_dim   = 144;    // Initial embedding dimension
    int32_t hiera_num_heads   = 2;      // Initial attention heads
    int32_t hiera_num_stages  = 4;
    int32_t hiera_stages[4]   = {2, 6, 36, 4};  // Blocks per stage
    int32_t hiera_q_pool      = 3;      // Stages with Q pooling (first 3)
    int32_t hiera_window_spec[4] = {8, 4, 16, 8}; // Window size per stage
    int32_t hiera_global_n    = 3;      // Number of global attention blocks
    int32_t hiera_global_idx[8] = {23, 33, 43}; // Global attention block indices
    int32_t hiera_pos_embed_bkg_h = 7;  // Background PE spatial size (config: [7,7])
    int32_t hiera_pos_embed_bkg_w = 7;

    int32_t fpn_top_down_n    = 2;      // How many levels get top-down fusion
    int32_t fpn_top_down_levels[4] = {2, 3};
    int32_t scalp             = 1;      // Discard N lowest-res FPN levels

    // ── NEW: SAM2-specific memory/tracking flags ──
    int32_t sigmoid_scale_x100   = 2000;  // 20.0 for HieraL/T
    int32_t sigmoid_bias_x100    = -1000; // -10.0 for HieraL/T
    int32_t use_high_res_features = 1;
    int32_t use_obj_ptrs_in_encoder = 1;
    int32_t pred_obj_scores       = 1;
    int32_t use_multimask_token_for_obj_ptr = 1;  // SAM2.1: true
    int32_t directly_add_no_mem_embed       = 1;  // SAM2.1: true
    int32_t non_overlap_masks_for_mem_enc   = 1;  // SAM2.1: true
    int32_t binarize_mask_from_pts          = 0;  // default: false
    int32_t multimask_output_for_tracking   = 1;  // SAM2.1: true
    int32_t multimask_min_pt_num            = 0;  // SAM2.1: 0 (default: 1)
    int32_t multimask_max_pt_num            = 1;  // SAM2.1: 1
    int32_t fixed_no_obj_ptr               = 1;  // SAM2.1: true
    int32_t iou_prediction_use_sigmoid      = 1;  // SAM2.1: true (matches SAM3)
    int32_t use_mask_input_as_output        = 1;  // SAM2.1: true (skip SAM on cond frames)
    int32_t multimask_output_in_sam         = 1;  // SAM2.1: true (default: false). Master switch for multimask.

    // ── Derived helpers ──
    // SAM3:
    int32_t n_img_embd()   const { return img_size / patch_size; }
    int32_t n_img_tokens() const { return n_img_embd() * n_img_embd(); }
    int32_t vit_head_dim() const { return vit_embed_dim / vit_num_heads; }

    // SAM2:
    int32_t hiera_total_blocks() const {
        int s = 0; for (int i = 0; i < hiera_num_stages; ++i) s += hiera_stages[i]; return s;
    }
    // Backbone feature map size: 1024 / 4 (patch stride) / 2^(num_q_pool_stages)
    // For Hiera, after PatchEmbed: 1024/4=256, then Q-pool stages reduce by 2x each.
    // Final backbone spatial size = 1024 / (4 * 2^3) = 1024/32 = 32 ... wait
    // Actually q_pool only applies to the first q_pool stages (stages 0..q_pool-1).
    // Spatial size after stage i = initial_size / (2^min(i+1, q_pool))
    // Stage 0 out: 256/2=128 (if q_pool >= 1)
    // Stage 1 out: 128/2=64
    // Stage 2 out: 64/2=32 ... no wait.
    // q_pool=3 means Q pooling happens at the END of stages 0,1,2
    // After PatchEmbed: 256x256
    // After stage 0 (q_pool at boundary): 128x128
    // Q-pooling in Hiera:
    //   q_pool_blocks = [stage_ends[i]+1 for i in 0..num_stages-2][:q_pool]
    //   For stages=[2,6,36,4]: stage_ends=[1,7,43,47], q_pool_blocks=[2,8,44]
    //   Block 44 (first of stage 3) HAS q_stride=(2,2).
    //
    //   Stage 0 (blocks 0-1):   no pooling,       output 256×256, dim=embed_dim
    //   Stage 1 (blocks 2-7):   Q-pool at block 2, output 128×128, dim=embed_dim*2
    //   Stage 2 (blocks 8-43):  Q-pool at block 8, output 64×64,   dim=embed_dim*4
    //   Stage 3 (blocks 44-47): Q-pool at block 44,output 32×32,   dim=embed_dim*8
    //
    //   backbone_channel_list=[1152,576,288,144] = [stage3,stage2,stage1,stage0] (reversed)
    //   Final backbone spatial = 1024 / 4 / 2^3 = 32.
    int32_t hiera_feat_size() const {
        // After PatchEmbed (stride=4), Q-pool reduces spatial at each stage transition.
        // q_pool_blocks are at start of stages 1..q_pool (blocks sum(stages[:i]) for i=1..q_pool).
        // But with scalp, the lowest-res level is discarded.
        // The effective feature size is the spatial size of the KEPT lowest-res level.
        //
        // Full backbone: after PatchEmbed→256, then Q-pool at stages 1..q_pool
        //   Stage 0: 256×256 (no pool)
        //   Stage 1: 128×128 (pool)
        //   Stage 2: 64×64 (pool)
        //   Stage 3: 32×32 (pool, if q_pool>=3)
        //
        // With scalp=1: discard stage 3 → effective = stage 2 output = 64×64
        int s = img_size / 4;
        // Pool at stages 1..min(q_pool, num_stages-1)
        int n_pools = (hiera_q_pool < hiera_num_stages) ? hiera_q_pool : hiera_num_stages - 1;
        for (int i = 0; i < n_pools; ++i) s /= 2;
        // With scalp: each scalp level "undoes" one pooling step
        for (int i = 0; i < scalp; ++i) s *= 2;
        return s;
    }
    // HieraL (q_pool=3, scalp=1): 1024/4=256, pool 3x: 256→128→64→32, scalp +1: 32→64. Result: 64 ✓
    // HieraT (q_pool=3, scalp=1): same. Result: 64 ✓
    // If scalp=0: Result: 32

    int32_t hiera_stage_dim(int stage) const {
        int d = hiera_embed_dim;
        for (int i = 0; i < stage; ++i) d *= 2;
        return d;
    }
    // Stage 0: embed_dim, Stage 1: embed_dim*2, Stage 2: embed_dim*4, Stage 3: embed_dim*8

    int32_t hiera_stage_heads(int stage) const {
        int h = hiera_num_heads;
        for (int i = 0; i < stage; ++i) h *= 2;
        return h;
    }

    int32_t hiera_stage_spatial(int stage) const {
        int s = img_size / 4; // After PatchEmbed
        for (int i = 1; i <= stage && i <= hiera_q_pool; ++i) s /= 2;
        return s;
    }
    // Stage 0: 256, Stage 1: 128, Stage 2: 64, Stage 3: 32

    float sigmoid_scale() const { return sigmoid_scale_x100 / 100.0f; }
    float sigmoid_bias()  const { return sigmoid_bias_x100  / 100.0f; }

    bool is_sam2() const { return model_type == SAM3_MODEL_SAM2; }
};
```

### 6.3 Hiera Block Weights (NEW)

```cpp
struct sam2_hiera_block {
    // LayerNorm1
    ggml_tensor * norm1_w = nullptr;  // [dim]
    ggml_tensor * norm1_b = nullptr;  // [dim]

    // MultiScaleAttention: fused QKV
    ggml_tensor * qkv_w   = nullptr;  // [3*dim_out, dim]
    ggml_tensor * qkv_b   = nullptr;  // [3*dim_out]
    ggml_tensor * proj_w   = nullptr; // [dim_out, dim_out]
    ggml_tensor * proj_b   = nullptr; // [dim_out]

    // LayerNorm2
    ggml_tensor * norm2_w = nullptr;  // [dim_out]
    ggml_tensor * norm2_b = nullptr;  // [dim_out]

    // MLP
    ggml_tensor * mlp_fc1_w = nullptr; // [dim_out*4, dim_out]
    ggml_tensor * mlp_fc1_b = nullptr; // [dim_out*4]
    ggml_tensor * mlp_fc2_w = nullptr; // [dim_out, dim_out*4]
    ggml_tensor * mlp_fc2_b = nullptr; // [dim_out]

    // Projection (only for stage transition blocks where dim != dim_out)
    ggml_tensor * dim_proj_w = nullptr; // [dim_out, dim]  (nullptr if dim==dim_out)
    ggml_tensor * dim_proj_b = nullptr; // [dim_out]

    // Block metadata (set during loading, not tensors)
    int stage_idx       = -1;    // Which stage this block belongs to
    int dim_in          = 0;     // Input dimension
    int dim_out         = 0;     // Output dimension
    int num_heads       = 0;     // Attention heads
    int window_size     = 0;     // 0 = global attention
    bool has_q_stride   = false; // Q-pooling (first block of stages 1..q_pool)
};
```

### 6.4 Hiera Backbone (NEW)

```cpp
struct sam2_hiera {
    // PatchEmbed: Conv2d(3, embed_dim, kernel=7, stride=4, padding=3)
    // NO LayerNorm after conv (just proj + permute BCHW→BHWC)
    ggml_tensor * patch_embed_w = nullptr; // [embed_dim, 3, 7, 7]
    ggml_tensor * patch_embed_b = nullptr; // [embed_dim]

    // Positional embeddings (background + window)
    // pos_embed shape depends on config window_pos_embed_bkg_spatial_size:
    //   HieraL: [1, 144, 7, 7]   (config: [7,7])
    //   HieraT: [1, 96, 7, 7]    (config: [7,7])
    // pos_embed_window shape: [1, embed_dim, window_spec[0], window_spec[0]]
    //   HieraL: [1, 144, 8, 8]   (window_spec[0]=8)
    //   HieraT: [1, 96, 8, 8]    (window_spec[0]=8)
    ggml_tensor * pos_embed        = nullptr;
    ggml_tensor * pos_embed_window = nullptr;

    // All blocks (flattened across stages)
    std::vector<sam2_hiera_block> blocks;

    // Stage end indices (cumulative sum of stages[])
    int stage_ends[4] = {};
};
```

### 6.5 FPN Neck (NEW)

```cpp
struct sam2_fpn_level {
    ggml_tensor * conv_w = nullptr;  // Conv2d(backbone_ch, d_model, k=1) — inside nn.Sequential
    ggml_tensor * conv_b = nullptr;  // [d_model]
};

struct sam2_fpn_neck {
    sam2_fpn_level levels[4];  // One per backbone output level (4 levels from Hiera)
    // Note: no additional 3x3 conv (SAM2 FPN only has 1x1 lateral convs)
    // Top-down fusion uses NEAREST interpolation (not bilinear) — from config
    // scalp=1 discards the lowest-resolution level (32×32), leaving 3 output levels
};
```

**CRITICAL: scalp=1**

All SAM2 configs set `scalp=1` in the ImageEncoder.  This discards the
lowest-resolution FPN level (32×32) after the FPN processes all 4 backbone
levels.  The result is **3 FPN output levels**, not 4:

```
Hiera backbone → 4 outputs: [256×256, 128×128, 64×64, 32×32]
FPN neck processes all 4 (with top-down fusion on levels 2,3)
scalp=1 → discard last: [256×256, 128×128, 64×64]  (3 levels only)
```

The effective backbone spatial size for all downstream processing is **64×64**
(not 32×32), matching `feat_sizes=[64,64]` in the memory attention config.
```

### 6.6 Modified sam3_model

```cpp
struct sam3_model {
    sam3_hparams hparams;
    ggml_type    weight_type = GGML_TYPE_F16;

    // ── SAM3-specific (loaded only when model_type == SAM3) ──
    sam3_vit             vit;
    sam3_neck            neck_det;
    sam3_neck            neck_trk;
    sam3_text_encoder    text_enc;
    sam3_fusion_encoder  fenc;
    sam3_detr_decoder    ddec;
    sam3_geom_encoder    geom_enc;
    sam3_seg_head        seg_head;

    // ── SAM2-specific (loaded only when model_type == SAM2) ──
    sam2_hiera           hiera;
    sam2_fpn_neck        fpn_neck;

    // ── Shared (loaded for both SAM2 and SAM3) ──
    sam3_sam_prompt_enc  sam_pe;
    sam3_sam_mask_dec    sam_dec;
    sam3_mem_enc         mem_enc;
    sam3_mem_attn        mem_attn;

    ggml_tensor *        obj_ptr_proj_w[3] = {};
    ggml_tensor *        obj_ptr_proj_b[3] = {};
    ggml_tensor *        no_obj_ptr        = nullptr;
    ggml_tensor *        obj_ptr_tpos_w    = nullptr;
    ggml_tensor *        obj_ptr_tpos_b    = nullptr;

    // ── SAM2 additional top-level tensors ──
    ggml_tensor *        no_mem_embed      = nullptr;  // [1, 1, 256]
    ggml_tensor *        no_mem_pos_enc    = nullptr;  // [1, 1, 256]
    ggml_tensor *        no_obj_embed_spatial = nullptr; // [1, 64]
    ggml_tensor *        mem_attn_norm_w   = nullptr;  // Final norm in memory attention
    ggml_tensor *        mem_attn_norm_b   = nullptr;

    // ggml backend (unchanged)
    ggml_context *       ctx     = nullptr;
    ggml_backend_t       backend = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    std::map<std::string, ggml_tensor *> tensors;

    // Tokenizer (SAM3 only)
    sam3_bpe_tokenizer   tokenizer;
};
```

### 6.7 Modified sam3_state

```cpp
struct sam3_state {
    // ── Backbone outputs ──
    // For SAM3: neck_det[4] + neck_trk[4]
    // For SAM2: neck_trk[4] only (no detector path)
    ggml_tensor * vit_output    = nullptr;
    ggml_tensor * neck_det[4]   = {};
    ggml_tensor * neck_trk[4]   = {};
    ggml_tensor * neck_det_pe[4] = {};
    ggml_tensor * neck_trk_pe[4] = {};

    // pix_feat for memory encoder = backbone_fpn[-1] = neck_trk[2] for BOTH models.
    // SAM2: neck_trk[2] = 64×64 (after scalp=1, 3 levels remain: indices 0,1,2).
    // SAM3: neck_trk[2] = 72×72. Same index — no model_type dispatch needed.

    int orig_width  = 0;
    int orig_height = 0;
    int n_threads   = 4;

    // ggml resources (unchanged)
    ggml_context  * ctx     = nullptr;
    ggml_backend_t  backend = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_gallocr  * galloc  = nullptr;
    ggml_backend_sched_t sched = nullptr;
    ggml_backend_t  aux_backend = nullptr;

    ggml_context  * pe_ctx  = nullptr;
    ggml_backend_buffer_t pe_buf = nullptr;

    bool pe_cache_valid = false;
    std::vector<float> pe_gauss_cache;
    float point_emb_cache[4][256] = {};
    float not_a_point_cache[256]  = {};
    float no_mask_emb_cache[256]  = {};
    std::vector<float> dense_pe_cache;
    std::vector<float> dense_nomask_cache;
};
```

---

## 7. New & Modified Functions

### 7.1 New Functions (SAM2-specific, `sam2_` prefix)

| Function | Signature | Purpose |
|---|---|---|
| `sam2_build_hiera_graph` | `(ctx, input, model) → ggml_tensor*[4]` | Build full Hiera backbone graph, return 4 stage outputs |
| `sam2_hiera_block_forward` | `(ctx, x, block, hp) → ggml_tensor*` | Single MultiScaleBlock forward pass |
| `sam2_multiscale_attention` | `(ctx, x, block) → ggml_tensor*` | Multi-head attention with optional Q-pooling (MaxPool2d) |
| `sam2_window_partition` | `(ctx, x, window_size) → (windows, pad_hw)` | Partition [B,H,W,C] into windows with padding |
| `sam2_window_unpartition` | `(ctx, windows, window_size, pad_hw, orig_hw) → ggml_tensor*` | Reverse window partition, remove padding |
| `sam2_hiera_pos_embed` | `(ctx, model, h, w) → ggml_tensor*` | Interpolate background PE + tile window PE |
| `sam2_build_fpn_neck_graph` | `(ctx, backbone_outs[4], model, out[3]) → void` | Build FPN with nearest top-down + scalp=1 |
| `sam2_encode_image_hiera` | `(state, model, image) → bool` | Full SAM2 image encoding: preprocess → Hiera → FPN → state |
| `sam2_load_hparams` | `(fin, hp) → bool` | Read SAM2-specific hyperparameters from file |
| `sam2_register_tensors` | `(model) → void` | Register all SAM2 tensor names + create ggml tensors |
| `sam2_precompute_hiera_metadata` | `(model) → void` | Set stage_idx, dim_in/out, window_size, has_q_stride on blocks |

### 7.2 Modified Functions (dispatch based on model_type)

| Function | Change |
|---|---|
| `sam3_load_model()` | Read magic → if `0x73616D32` call `sam2_load_hparams` + `sam2_register_tensors`, else existing SAM3 path |
| `sam3_encode_image()` | If `model.hparams.is_sam2()` call `sam2_encode_image_hiera()`, else existing `sam3_encode_image_vit()` |
| `sam3_create_state()` | Allocate PE buffers sized for the active backbone's spatial resolution |
| `sam3_encode_memory()` | Use `hparams.sigmoid_scale()` / `hparams.sigmoid_bias()` instead of hardcoded 20.0 / -10.0 |
| `sam3_propagate_single()` | Use `hparams.hiera_feat_size()` or `hparams.n_img_embd()` for spatial dimensions |
| `sam3_segment_pvs()` | Feature map size from hparams (64×64 for SAM2 vs 72×72 for SAM3) |

### 7.3 Unchanged Functions (used by both)

All functions listed in Section 3 "Fully Reused" — no modifications needed.

---

## 8. Model Loading

### 8.1 Magic-Based Dispatch

```cpp
std::shared_ptr<sam3_model> sam3_load_model(const sam3_params & params) {
    FILE * fin = fopen(params.model_path.c_str(), "rb");

    uint32_t magic;
    fread(&magic, sizeof(magic), 1, fin);

    if (magic == 0x73616D33) {
        // SAM3 loading (existing code)
        return sam3_load_model_sam3(fin, params);
    } else if (magic == 0x73616D32) {
        // SAM2 loading (new code)
        return sam2_load_model_sam2(fin, params);
    } else {
        fprintf(stderr, "%s: unknown magic 0x%08x\n", __func__, magic);
        return nullptr;
    }
}
```

### 8.2 SAM2 Loading Pipeline

```
sam2_load_model_sam2(fin, params):
  1. sam2_load_hparams(fin, model.hparams)
  2. sam2_precompute_hiera_metadata(model)
  3. sam3_init_backend(model, params)     // Shared: Metal or CPU
  4. sam2_register_tensors(model)          // Create all ggml tensors
  5. sam3_register_shared_tensors(model)   // SAM PE, SAM dec, mem_enc, mem_attn, obj_ptr
  6. sam3_load_tensor_data(fin, model)     // Shared: read binary data into tensors
  7. return model
```

### 8.3 Tensor Registration for SAM2

`sam2_register_tensors()` creates ggml tensors for:

1. **Hiera backbone** — PatchEmbed (7×7 conv + bias), pos_embed, pos_embed_window, all blocks (variable dim per stage)
2. **FPN neck** — 4 × Conv2d(backbone_ch[i], 256, k=1)
3. **Shared tensors** (delegated to `sam3_register_shared_tensors()`) — sam_pe, sam_dec, mem_enc, mem_attn, obj_ptr, no_mem_embed, etc.

The key difference is that Hiera blocks have variable dimensions per stage:
- Stage 0 blocks: dim_in = embed_dim, dim_out = embed_dim (except first block of stage 1)
- First block of each subsequent stage: dim_in = prev_dim, dim_out = prev_dim * 2

---

## 9. Image Encoder: Hiera Backbone

### 9.1 PatchEmbed

```
Input:  [B, 3, 1024, 1024]
Output: [B, 256, 256, embed_dim]   (H/4, W/4, C)

Conv2d(3 → embed_dim, kernel=7, stride=4, padding=3)
Permute: BCHW → BHWC
```

In ggml:
```cpp
ggml_tensor * x = ggml_conv_2d(ctx, model.hiera.patch_embed_w, input, 4, 4, 3, 3);
x = ggml_add(ctx, x, ggml_reshape_4d(ctx, model.hiera.patch_embed_b, 1, 1, embed_dim, 1));
// x: [embed_dim, 256, 256, B] in ggml (column-major)
```

### 9.2 Positional Embedding

```
pos_embed:        [1, embed_dim, bkg_H, bkg_W]  (background, bicubic interpolated to HxW)
                  HieraL: [1, 144, 7, 7]   (config: window_pos_embed_bkg_spatial_size=[7,7])
                  HieraT: [1, 96, 7, 7]
pos_embed_window: [1, embed_dim, W0, W0]   (window, tiled to fill HxW)
                  HieraL: [1, 144, 8, 8]   (window_spec[0]=8)
                  HieraT: [1, 96, 8, 8]

_get_pos_embed(h, w):
  1. bicubic_interpolate(pos_embed, (h, w))   // [1, E, 7, 7] → [1, E, 256, 256]
  2. tile(pos_embed_window, [h/W0, w/W0])     // [1, E, 8, 8] → [1, E, 256, 256]
     tile_factors = [pos.shape[i] // win.shape[i] for i in 0..3]
     For h=w=256, W0=8: factor = 256/8 = 32
  3. PE = interpolated + tiled
  4. permute [1, E, H, W] → [1, H, W, E]
  5. x = x + PE
```

In ggml:
- Precompute the interpolated+tiled PE on CPU at image encoding time (H=W=256 after PatchEmbed)
- Upload as a fresh input tensor
- The bicubic interpolation uses `F.interpolate(mode="bicubic")` — in C++ use
  `ggml_interpolate()` with `GGML_SCALE_MODE_BICUBIC` or precompute on CPU

### 9.3 Attention: NO RoPE in Hiera (Critical Difference from SAM3 ViT)

Hiera uses **plain scaled dot-product attention** with NO rotary position embeddings.
Position information comes solely from the additive PE applied once after PatchEmbed
(Section 9.2). After window partitioning, tokens lose global positional context —
this is by design (the window PE component provides local position awareness).

**Comparison with SAM3 ViT:** SAM3's ViT applies 2D axial RoPE to Q and K at
EVERY layer, with different frequencies for global (72×72) vs windowed (24×24)
attention.  Hiera has none of this — no `freqs_cis` tensors in the checkpoint,
no rotary encoding functions needed.

**This simplifies the Hiera implementation:** Each attention block is just
QKV projection → optional Q-pool → `F.scaled_dot_product_attention` → output
projection.  No RoPE application code needed in `sam2_multiscale_attention()`.

Note: RoPE IS used in SAM2's **memory attention** module (Section 12.1) — that
is the same as SAM3 and already implemented.  The distinction is backbone (no
RoPE) vs memory attention (yes RoPE).

### 9.4 Multi-Stage Block Processing

```
For flat_block_idx = 0 .. total_blocks-1:
    block = model.hiera.blocks[flat_block_idx]

    // 1. Pre-norm + shortcut
    shortcut = x                      // [B, H, W, C_in]
    x = LayerNorm(x, block.norm1_w/b)

    // 2. Dimension projection on shortcut (at stage transitions)
    //    IMPORTANT: proj is applied to NORMED x, then MaxPooled for shortcut.
    //    This happens BEFORE attention.
    if block.dim_in != block.dim_out:
        shortcut = Linear(x, block.dim_proj_w/b)     // [B, H, W, C_out]
        shortcut = MaxPool2d(shortcut, k=2, s=2)      // [B, H/2, W/2, C_out]

    // 3. Window partition (if not global attention)
    //    Applied AFTER norm1, BEFORE attention.
    if block.window_size > 0:
        (x, pad_hw) = sam2_window_partition(x, block.window_size)
        // x: [B*nW, ws, ws, C_in]

    // 4. Multi-scale attention
    //    QKV: nn.Linear(C_in, 3*C_out) — input is old dim, output is 3× new dim
    //    If block.has_q_stride: Q is MaxPooled (k=2,s=2) → spatial halved
    //    Output projection: nn.Linear(C_out, C_out)
    x = sam2_multiscale_attention(ctx, x, block)

    // 5. Window unpartition
    if block.window_size > 0:
        effective_ws = block.has_q_stride ? (block.window_size / 2) : block.window_size
        x = sam2_window_unpartition(x, effective_ws, pad_hw, target_hw)

    // 6. Residual + DropPath
    x = shortcut + DropPath(x)

    // 7. MLP + DropPath
    //    MLP uses dim_out for all dimensions: Linear(C_out, C_out*4) → GELU → Linear(C_out*4, C_out)
    //    GELU is standard (not approximate)
    x = x + DropPath(MLP(LayerNorm(x, block.norm2_w/b)))

    // 8. Collect intermediate output at stage ends
    //    NO final LayerNorm — raw block output is used directly
    if flat_block_idx == stage_end[s]:
        intermediates[s] = permute(x, BHWC → BCHW)
```

### 9.5 MultiScale Attention with Q-Pooling

```cpp
static ggml_tensor * sam2_multiscale_attention(
    ggml_context * ctx,
    ggml_tensor  * x,       // [B, H, W, C]
    const sam2_hiera_block & block)
{
    int B = x->ne[3], H = x->ne[1], W = x->ne[0]; // ggml reversed dims
    int C = block.dim_in;
    int C_out = block.dim_out;
    int n_heads = block.num_heads;

    // Flatten spatial: [B, H*W, C]
    // QKV projection: [B, H*W, 3*C_out]
    auto * qkv = ggml_mul_mat(ctx, block.qkv_w, x_flat);
    qkv = ggml_add(ctx, qkv, block.qkv_b);

    // Split Q, K, V: each [B, H*W, C_out]
    // Reshape: [B, H*W, n_heads, head_dim]

    // Q-pooling (if has_q_stride):
    if (block.has_q_stride) {
        // Reshape Q to [B, H, W, C_out], apply MaxPool2d(k=2, s=2), reshape back
        // Q: [B, H/2, W/2, C_out] → [B, H*W/4, n_heads, head_dim]
    }

    // Scaled dot-product attention
    // Q: [B, n_heads, N_q, head_dim]
    // K: [B, n_heads, N_kv, head_dim]
    // V: [B, n_heads, N_kv, head_dim]
    // out: [B, n_heads, N_q, head_dim]

    // Recombine heads + output projection
    auto * out = ggml_mul_mat(ctx, block.proj_w, recombined);
    out = ggml_add(ctx, out, block.proj_b);

    // Reshape back to spatial: [B, H_out, W_out, C_out]
    return out;
}
```

### 9.6 Window Partition / Unpartition

```cpp
// Partition: [B, H, W, C] → [B*num_windows, ws, ws, C]
// Pad H, W to be divisible by ws
// Reshape: [B, H/ws, ws, W/ws, ws, C] → [B*H/ws*W/ws, ws, ws, C]

// Unpartition: reverse + crop to original H, W
```

In ggml this is implemented via `ggml_view` + `ggml_permute` + `ggml_cont` operations,
similar to the existing `sam3_window_partition` but parameterized for variable window sizes.

---

## 10. Image Encoder: FPN Neck

### 10.1 Architecture

SAM2's FPN is simpler than SAM3's SimpleFPN:
- **Lateral connections:** 1×1 Conv per backbone level (project to d_model=256)
- **Top-down path:** **Nearest** 2× upsample + element-wise sum (for levels in `fpn_top_down_levels`)
- **No 3×3 refinement conv** (unlike SAM3)
- **No ConvTranspose** (nearest-neighbor interpolation instead)
- **scalp=1:** Discards lowest-resolution FPN output after processing

### 10.2 Forward Pass

```
Hiera backbone outputs (4 stages, forward order):
  Stage 0: [B, 144, 256, 256]   → convs[3] = Conv1x1(144→256) → [B, 256, 256, 256]
  Stage 1: [B, 288, 128, 128]   → convs[2] = Conv1x1(288→256) → [B, 256, 128, 128]
  Stage 2: [B, 576, 64, 64]     → convs[1] = Conv1x1(576→256) → [B, 256, 64, 64]
  Stage 3: [B, 1152, 32, 32]    → convs[0] = Conv1x1(1152→256)→ [B, 256, 32, 32]

NOTE: convs are indexed as convs[n-i] where n=3, so convs[0] maps the
LAST backbone level (1152ch) and convs[3] maps the FIRST (144ch).

FPN processes in reverse order (low to high resolution), i = 3→0:
  i=3: lateral = convs[0](stage_3) = 32×32.  3 ∈ [2,3], prev=None → prev=lateral
  i=2: lateral = convs[1](stage_2) = 64×64.  2 ∈ [2,3], prev≠None →
       top_down = nearest_upsample(32×32, 2×) = 64×64
       prev = lateral + top_down = 64×64 (top-down fused!)
  i=1: lateral = convs[2](stage_1) = 128×128.  1 ∉ [2,3] → prev=lateral
  i=0: lateral = convs[3](stage_0) = 256×256.  0 ∉ [2,3] → prev=lateral

FPN raw output (4 levels):
  out[0]: [B, 256, 256, 256]  — stride 4
  out[1]: [B, 256, 128, 128]  — stride 8
  out[2]: [B, 256, 64, 64]    — stride 16 (with top-down from 32×32)
  out[3]: [B, 256, 32, 32]    — stride 32

AFTER scalp=1 (discard last):
  backbone_fpn[0]: [B, 256, 256, 256]  — stride 4   (high-res s0)
  backbone_fpn[1]: [B, 256, 128, 128]  — stride 8   (high-res s1)
  backbone_fpn[2]: [B, 256, 64, 64]    — stride 16  (image embedding)

Only 3 levels remain.  backbone_fpn[-1] = 64×64 (the main feature).

Position encoding: PositionEmbeddingSine(256) applied to each level.
  Formula: y = arange(1,H+1)/H * 2π, x = arange(1,W+1)/W * 2π,
  then sin/cos with temperature=10000. Output [B, 256, H, W].
  This MATCHES SAM3's sam3_sinusoidal_pe_2d ((i+1)/H * 2π). No learnable params.
```

### 10.3 Mapping to sam3_state

After FPN + scalp, we store outputs into `state.neck_trk[]` and `state.neck_trk_pe[]`.
SAM2 has 3 levels (after scalp=1); SAM3 has 4 levels (no scalp).

For SAM2 (3 levels after scalp=1):
- `state.neck_trk[0]` = FPN level 0: [256, 256, 256] — high-res for conv_s0
- `state.neck_trk[1]` = FPN level 1: [256, 128, 128] — high-res for conv_s1
- `state.neck_trk[2]` = FPN level 2: [256, 64, 64]   — image embedding + pix_feat
- `state.neck_trk[3]` = nullptr (not used)

For SAM3 (4 levels, no scalp):
- `state.neck_trk[0]` = scale 0: [256, 288, 288]
- `state.neck_trk[1]` = scale 1: [256, 144, 144]
- `state.neck_trk[2]` = scale 2: [256, 72, 72]
- `state.neck_trk[3]` = scale 3: [256, 36, 36]

**Unified convention (both models):**
- `neck_trk[-1]` (last non-null) = image embedding for SAM decoder + pix_feat for memory encoder
- `neck_trk[0]` = high-res level for conv_s0
- `neck_trk[1]` = high-res level for conv_s1

**Accessing the right index:** Add `hp.n_neck_levels()` helper (3 for SAM2, 4 for SAM3)
and `hp.img_embed_idx()` returning `n_neck_levels - 1` (2 for SAM2, 2 for SAM3).
For SAM3 the pix_feat is neck_trk[2] (72×72); for SAM2 it's also neck_trk[2] (64×64).
This happens to be the same index for both, simplifying the code.

---

## 11. SAM Decoder Path (Mostly Shared — See Caveats)

The SAM decoder architecture is structurally identical between SAM2 and SAM3,
but several config-driven behavioral differences exist.

**SAM2 image segmentation (PVS) path — NO memory attention:**
When using `sam3_segment_pvs()` with a SAM2 model, the path is:
1. `sam3_encode_image()` → Hiera backbone + FPN → state
2. During image encoding, `no_mem_embed` is added directly to the image embedding
   (same as first-frame behavior with `directly_add_no_mem_embed=True`)
3. `sam3_segment_pvs()` → prompt encoder + mask decoder (NO memory attention)
This matches SAM2ImagePredictor, which calls `sam_prompt_encoder` and
`sam_mask_decoder` directly without going through `track_step`.

**Post-processing:** Mask logits are upsampled to original image dimensions via
`F.interpolate(mode="bilinear", align_corners=False)`, then thresholded at 0.0
on logits for binary output.

### 11.1 Prompt Encoder — Fully Shared

`sam3_build_sam_pe()` is reused without changes.  The Gaussian PE matrix,
point embeddings, mask downscaling, and dense PE generation are all
architecturally identical.  Spatial dimensions adapt automatically from
tensor shapes.

Key shared behaviors:
- Point coordinates normalized: `coords / input_image_size` → [0,1] → `2*x - 1` → [-1,+1]
- Box prompts shifted: `boxes += 0.5` (center-of-pixel) before PE, using point_embeddings[2] (TL) and [3] (BR)
- Padding points: label=-1 → replaced with `not_a_point_embed` (discards PE)
- Dense PE: `PositionEmbeddingRandom.forward(embedding_size)` normalizes by embedding dims (not image dims)

**mask_input_size** = 4 × image_embedding_size:
- SAM2 (64×64 backbone): mask_input_size = 256×256
- SAM3 (72×72 backbone): mask_input_size = 288×288
- The prompt encoder's mask downscaling (Conv k=2,s=2 × 2 + Conv k=1) reduces
  mask_input_size by 4× to match backbone spatial size. This adapts automatically.

**Mask prompt handling in video:** When a prior mask is fed as input (e.g., from
a previous frame), SAM2 resizes it to mask_input_size using bilinear interpolation
with `antialias=True` and NO re-binarization before the prompt encoder. The mask
is treated as float logits.

**Binarization threshold:** SAM2 uses 0.0 on logits (matching SAM3).

### 11.2 TwoWay Transformer — Fully Shared

The 2-layer TwoWayTransformer is identical:
- `skip_first_layer_pe` on layer 0
- Attention `downsample_rate=2` → `internal_dim=128`, `head_dim=16`
- Q/K swap in image-to-token cross-attention
- **ReLU** activation in TwoWay block MLPs (2-layer: 256→2048→256)
  Note: TwoWayTransformer has its own `activation=nn.ReLU` default, separate
  from MaskDecoder's `activation=nn.GELU`. The GELU only applies to the
  output_upscaling ConvTranspose layers, NOT to the transformer's internal MLPs.
- LayerNorm + residual connections

### 11.3 Upscaling + High-Res Features — Shared With Caveat

- ConvTranspose2d(256→64, k=2, s=2) + LayerNorm2d + GELU
- conv_s1(256→64) skip added before LayerNorm
- ConvTranspose2d(64→32, k=2, s=2) + GELU
- conv_s0(256→32) skip added before GELU
- 4 hypernetwork MLPs: 256→256→256→32 (ReLU intermediate, linear final)
- Final mask: `masks = (hyper_in @ upscaled_embedding.view(B, 32, -1)).view(B, -1, 4H, 4W)`
  where hyper_in=[B, num_masks, 32] from hypernetwork, upscaled_embedding=[B, 32, 4H, 4W]

**Mask slice selection (multimask vs single-mask):**
- `multimask_output=False`: `mask_slice = [s+0 : s+1]` → 1 mask from hypernetwork MLP[0]
- `multimask_output=True`: `mask_slice = [s+1 : s+4]` → 3 masks from hypernetwork MLPs[1-3]
  (s is the pred_obj_scores offset: 0 or 1)
  Best mask selected by `argmax(ious)` after the decoder returns.

**CRITICAL: conv_s0/conv_s1 pre-projection.**
In SAM2, `forward_image()` pre-applies conv_s0 and conv_s1 to backbone_fpn[0]
and backbone_fpn[1] **ONCE** during image encoding (AFTER scalp).  The
projected features are stored in `state.neck_trk[0]` and `state.neck_trk[1]`.
During tracking, the mask decoder receives these already-projected features
and adds them directly (no conv_s0/conv_s1 inside the decoder).

In SAM3, conv_s0/conv_s1 are applied inside `sam3_build_sam_dec_graph()` every
time the decoder runs.

**Action for SAM2:** Apply conv_s0 and conv_s1 during `sam2_encode_image_hiera()`
and store the projected features in `state.neck_trk[0/1]`.  The downstream
decoder code must know whether features are already projected (SAM2) or need
projection (SAM3).  Simplest approach: SAM2 stores projected features; SAM3
stores raw features + applies conv in decoder.  The decoder checks a flag or
the channel count (32/64 vs 256) to decide.

### 11.4 Token Layout — **Needs SAM2 Variant**

SAM3 always uses 6 output tokens: `[obj_score, iou, mask×4]`.

SAM2 with `pred_obj_scores=True` (SAM2.1 default): same 6-token layout.
SAM2 with `pred_obj_scores=False` (older configs): 5-token layout
`[iou, mask×4]`, no object score prediction.

**When `pred_obj_scores=False`:**
- Token extraction indices shift: iou at 0, masks at 1-4
- Object score defaults to `sigmoid(10.0) ≈ 1.0` (always present)
- Object score head weights don't exist in checkpoint

For SAM2.1 configs, the 6-token layout matches SAM3 exactly.  Only
older SAM2 checkpoints need the 5-token path.  **Decision:** implement
`sam2_build_sam_dec_graph()` that checks `pred_obj_scores` flag and
handles both token layouts, rather than modifying the SAM3 function.

### 11.5 IoU Prediction Head — Minor Difference

SAM3 always applies sigmoid to IoU predictions.
SAM2 has `iou_prediction_use_sigmoid` flag.  SAM2.1 configs set this to
`True`, matching SAM3.  Older SAM2 configs may set it to `False`.

**Impact:** With `iou_prediction_use_sigmoid=False`, IoU outputs are raw
logits, not probabilities.  The post-processing (mask selection) must
handle raw logits vs probabilities correctly.

**Action:** Add flag to hparams.  For SAM2.1 (True), behavior matches SAM3
exactly.  The SAM2 decoder variant skips sigmoid on IoU head only when
flag is `0` (older configs).

### 11.6 Automatic Spatial Adaptation

The key dimensions adapt automatically from tensor shapes:
- `feat_size` = width of `neck_trk[2]` tensor (64 for SAM2, 72 for SAM3)
- Number of spatial tokens = `feat_size²` (4096 for SAM2, 5184 for SAM3)
- High-res sizes from `neck_trk[0]` and `neck_trk[1]` tensors
- Upscaled mask size = feat_size × 4 (256 for SAM2, 288 for SAM3)

---

## 12. Video Tracking Path (Shared)

### 12.1 Memory Attention

Identical between SAM2 and SAM3:
- 4 layers of self-attention (RoPE) + cross-attention (RoPE, kv_dim=64) + FFN
- Pre-norm architecture (LayerNorm applied before each sub-layer)
- Order per layer: self-attention → cross-attention → FFN
- Current features from `state.neck_trk[2]` (flattened to tokens)
- Memory bank entries from stored `sam3_memory_slot` structs
- Object pointers from stored pointer bank

**Feature format for memory attention (critical for ggml graph building):**
Vision features are reshaped from `B×C×H×W` → `HW×B×C` (sequence-first format)
before entering the memory attention transformer.  After memory attention, the
output is reshaped back: `HW×B×C` → `B×C×H×W`.  This is done by
`_prepare_backbone_features()` in Python; in C++ the equivalent is a
`ggml_permute` + `ggml_cont` + `ggml_reshape` sequence.

**Wrapper-level details:**
- `pos_enc_at_input`: When True, add `0.1 * curr_pos` to current features BEFORE
  the first layer.  This 0.1 scaling is hardcoded in MemoryAttention.forward().
- Final LayerNorm applied AFTER all 4 layers (at wrapper level, not inside layers).

The only parameterized difference: spatial token count (4096 vs 5184) and
RoPE frequency table dimensions — both derived from `feat_size`.

**RoPE exclusion for object pointers (`num_k_exclude_rope`) — CRITICAL:**
In cross-attention, memory tokens include both spatial memories (with spatial
positions → need RoPE) and object pointer tokens (temporal-only → no RoPE).
The `num_k_exclude_rope` mechanism handles this: the last N key/value tokens
(object pointers) are excluded from RoPE application.

Implementation in RoPEAttention.forward():
```python
num_k_rope = k.size(-2) - num_k_exclude_rope
q, k[:, :, :num_k_rope] = apply_rotary_enc(q, k[:, :, :num_k_rope], ...)
# k[:, :, num_k_rope:] (object pointers) remain unrotated
```

**`rope_k_repeat` for cross-attention sequence length mismatch:**
In cross-attention, Q has 4096 spatial tokens but K has N×4096+M tokens
(spatial memories + object pointers).  RoPE frequencies are computed for
the query spatial grid (64×64) and repeated to cover all K spatial tokens:
`repeat_factor = num_k_rope / q_seq_len`.  This ensures spatial tokens at
the same position across different memory frames get the same RoPE encoding.

The ggml implementation must:
1. Concatenate spatial memory tokens BEFORE object pointer tokens
2. Apply RoPE only to the first `num_k_rope` key tokens (spatial memories)
3. Leave the last `num_k_exclude_rope` tokens (object pointers) unrotated
4. Use repeated RoPE frequencies for K when K is longer than Q

**Changes needed in `sam3_propagate_single()`:**
- Use `hp.hiera_feat_size()` instead of `hp.n_img_embd()` when `hp.is_sam2()`
- Or better: read from the tensor shape directly (already done in current code)

### 12.2 Data Flow: Original vs Memory-Conditioned Features — **CRITICAL**

During video tracking, two DIFFERENT versions of the backbone features exist:

| Tensor | Source | Goes to |
|---|---|---|
| **Original features** | `backbone_fpn[-1]` (from image encoder) | Memory encoder (`_encode_new_memory`) |
| **Memory-conditioned features** | Output of `memory_attention()` | SAM mask decoder (`_forward_sam_heads`) |
| **High-res features** | `backbone_fpn[0], backbone_fpn[1]` (original, pre-projected by conv_s0/s1) | SAM mask decoder (skip connections) |

**The memory encoder MUST receive the ORIGINAL backbone features, NOT the
memory-conditioned output.**  This ensures that stored memories represent
the raw visual content of each frame, not a recursive self-reference.

**Implementation in `sam3_propagate_single()`:**
1. Copy original `state.neck_trk[2]` to a CPU buffer (`orig_feat`)
2. Run memory attention on `orig_feat` → produces `conditioned_feat`
3. Pass `conditioned_feat` to SAM decoder → produces mask
4. Pass `orig_feat` (NOT conditioned_feat) + mask to memory encoder → stored in memory bank

SAM3's current implementation already follows this pattern: `sam3_propagate_single`
copies `state.neck_trk[2]` to a fresh input tensor, runs memory attention to produce
conditioned features, then `sam3_encode_memory` separately copies `state.neck_trk[2]`
again for the memory encoder.  This is correct for SAM2 as well.

### 12.3 Memory Encoder — Shared Architecture, Different Mask Sizing

Architecture is identical:
1. Mask sigmoid + scale/bias
2. MaskDownSampler (4 Conv2d stages k=3,s=2,p=1 + final 1×1)
3. Pixel feature projection (Conv2d 256→256, k=1) — applied BEFORE adding downsampled mask
4. Fuser (2 CXBlock layers: dwconv k=7 p=3, LayerNorm2d, FC 256→1024→256, gamma=1e-6)
5. Output projection (Conv2d 256→64, k=1)
6. Sinusoidal PE (64-dim)

**Pre-memory-encoding steps (in _encode_new_memory, BEFORE calling memory_encoder):**

1. **Non-overlapping constraint** (multi-object tracking):
   When `non_overlap_masks_for_mem_enc=True` and in eval mode, apply per-spatial-location
   winner-take-all: keep highest-scoring object's mask, clamp all others to -10.0.
   ```python
   max_obj_inds = argmax(pred_masks, dim=0, keepdim=True)  # [1,H,W]
   keep = (max_obj_inds == batch_obj_inds)                  # [B,1,H,W]
   pred_masks = where(keep, pred_masks, clamp(pred_masks, max=-10.0))
   ```
   This prevents spatial overlap in stored memories.

2. **Mask binarization** (optional, `binarize_mask_from_pts_for_mem_enc` flag):
   When True AND the mask came from point prompts (not prior mask input):
   `mask_for_mem = (pred_masks_high_res > 0).float()` (hard threshold).
   Otherwise: `mask_for_mem = sigmoid(pred_masks_high_res)`.

3. **Sigmoid + scale + bias** (applied to sigmoid output, NOT re-applied inside memory_encoder):
   `mask_for_mem = mask_for_mem * sigmoid_scale + sigmoid_bias`
   Memory encoder is called with `skip_mask_sigmoid=True` since sigmoid was already applied.

**Changes needed:**

1. **sigmoid_scale/bias:** Use `hp.sigmoid_scale()` / `hp.sigmoid_bias()` instead of
   hardcoded 20.0 / -10.0.

2. **Mask interpolation target size (CRITICAL):**
   - SAM3: mask logits 288×288 → interpolate to 1008 → interpolate to **1152** → downsample to 72
   - SAM2: mask logits 256×256 → interpolate to **1024** → downsample to 64
   - Formula: `INTERPOL = feat_size × 16` (SAM3: 72×16=1152, SAM2: 64×16=1024)
   - The first interpolation (to HIGH_RES=image_size) stays the same
   - The second interpolation target changes with backbone spatial size

3. **pix_feat source:** Both SAM2 and SAM3 use `backbone_fpn[-1]` = `neck_trk[2]`.
   SAM2: neck_trk[2] = 64×64 (after scalp=1 removes level 3, only 3 levels remain).
   SAM3: neck_trk[2] = 72×72.
   Same index for both — no dispatch needed for pix_feat source.

4. **no_obj_embed_spatial modulation** (after memory encoder, in `_encode_new_memory`):
   When `no_obj_embed_spatial` is not None (SAM2.1: shape [1, 64]):
   ```
   is_obj_appearing = (object_score_logits > 0).float()
   maskmem_features += (1 - is_obj_appearing) * no_obj_embed_spatial
   ```
   This adds a learned spatial embedding to memory features when the object is NOT
   present in the frame, marking the memory as "no object here".  Applied AFTER the
   memory encoder's out_proj (on the 64-dim output), not inside the encoder itself.

### 12.4 Object Pointer — Shared Architecture, Different Token Selection

Architecture is identical: MLP projection → temporal PE → no_obj_ptr fallback.

**Differences in token selection (SAM2.1 HieraL config):**

When `use_multimask_token_for_obj_ptr=True` and `multimask_output=True`:
1. Mask decoder returns `sam_output_tokens` with shape [B, 3, C] (3 multimask tokens)
2. Best mask index: `best_idx = argmax(ious, dim=-1)` over the 3 candidates
3. Object pointer token: `sam_output_tokens[batch, best_idx]` → [B, C]

When `use_multimask_token_for_obj_ptr=False` (SAM3 behavior):
1. Mask decoder returns `sam_output_tokens` with shape [B, 1, C] (first mask token)
2. Object pointer token: `sam_output_tokens[:, 0]` → [B, C]

**Object presence modulation (SAM2.1 with `fixed_no_obj_ptr=True`):**
```
λ = sigmoid(object_score_logits)     if soft_no_obj_ptr=True
λ = (object_score_logits > 0).float  if soft_no_obj_ptr=False  (SAM2.1 default: False)

obj_ptr = λ * obj_ptr_proj(token)    // Scale by presence
obj_ptr = obj_ptr + (1 - λ) * no_obj_ptr  // Mix with learned no-object vector
```

SAM3 uses the same formula but always selects the first mask token.

### 12.5 Memory Frame Selection Algorithm

The exact algorithm for building the memory attention input (from SAM2Base._prepare_memory_conditioned_features):

**Step 1: Select conditioning frames** (user-annotated frames)
- All conditioning frames get `t_pos = 0`
- Indexed in `maskmem_tpos_enc` at position `num_maskmem - 1` (= 6 for num_maskmem=7)
- Limited by `max_cond_frames_in_attn` (default: -1 = unlimited), sorted by temporal closeness

**Step 2: Select non-conditioning frames** (t_pos = 1 to num_maskmem-1)
- `t_pos=1` → immediately previous frame (ALWAYS included)
- `t_pos≥2` → frames at stride `memory_temporal_stride_for_eval` (default: 1)
  - Forward: `prev = ((frame_idx-2) // stride) * stride - (t_rel-2) * stride`
  - If frame output doesn't exist: skip (no padding)
- Each gets `maskmem_tpos_enc[num_maskmem - t_pos - 1]` as temporal PE

**Step 3: Select object pointers** (up to max_obj_ptrs_in_encoder=16)
- From conditioning frames: temporal distance = `(frame_idx - t) * tpos_sign_mul`
  (signed when `use_signed_tpos_enc_to_obj_ptrs=True`)
- From non-conditioning frames: distance = 1, 2, 3, ... (walking backwards)
- When `only_obj_ptrs_in_the_past_for_eval=True`: skip future frames
- Temporal PE: `get_1d_sine_pe(pos / (max_obj_ptrs - 1), dim=tpos_dim)` → projected via `obj_ptr_tpos_proj`
- Each pointer split 256→4×64, PE replicated for each sub-token

**Step 4: Concatenate** → `[spatial_memories, obj_ptrs]` along sequence dim

**First-frame handling** (when `directly_add_no_mem_embed=True`):
- No memory attention is run
- Instead: `features = features + no_mem_embed` (broadcast addition)
- This **bypasses the entire memory attention transformer**

### 12.6 Video Tracker State

`sam3_tracker` struct is unchanged.  The video tracking API works for both:
- `sam3_create_visual_tracker()` — for SAM2 (no detection)
- `sam3_propagate_frame()` — propagate all instances
- `sam3_tracker_add_instance()` — add from PVS prompts
- `sam3_refine_instance()` — interactive clicks

SAM2 never uses `sam3_create_tracker()` (text-prompted) or `sam3_track_frame()` (detection + tracking)
since those require the text encoder and DETR decoder.

---

## 13. Public API Changes

### 13.1 New Query Function

```cpp
/* Returns the model type (SAM2 or SAM3). */
sam3_model_type sam3_get_model_type(const sam3_model & model);
```

### 13.2 Existing Functions That Now Work for Both

| Function | SAM2 | SAM3 |
|---|---|---|
| `sam3_load_model()` | ✅ Auto-detects from magic | ✅ |
| `sam3_create_state()` | ✅ | ✅ |
| `sam3_encode_image()` | ✅ Dispatches to Hiera | ✅ Dispatches to ViT |
| `sam3_segment_pvs()` | ✅ | ✅ |
| `sam3_create_visual_tracker()` | ✅ | ✅ |
| `sam3_propagate_frame()` | ✅ | ✅ |
| `sam3_tracker_add_instance()` | ✅ | ✅ |
| `sam3_refine_instance()` | ✅ | ✅ |

### 13.3 SAM3-Only Functions (unchanged, return error for SAM2)

| Function | Behavior with SAM2 model |
|---|---|
| `sam3_segment_pcs()` | Returns empty result + warning |
| `sam3_create_tracker()` | Returns nullptr + warning (requires text encoder) |
| `sam3_track_frame()` | Returns empty result if text tracker |

### 13.4 No New Public API Functions

SAM2 is fully served by the existing public API.  The only addition is
`sam3_get_model_type()` for callers that need to know whether PCS is available.

---

## 14. Tensor Name Mapping (SAM2)

Complete list of all ggml tensor names for a SAM2 model, with shapes for Hiera Large:

### 14.1 Hiera Backbone (~300M params)

```
hiera.patch_embed.weight         [144, 3, 7, 7]
hiera.patch_embed.bias           [144]
hiera.pos_embed                  [1, 144, 7, 7]    # bkg_spatial_size from config
hiera.pos_embed_window           [1, 144, 8, 8]    # window_spec[0] from config

# Per-block (48 blocks for HieraL: stages [2,6,36,4])
# Dimensions: stage 0→144, stage 1→288, stage 2→576, stage 3→1152

hiera.blocks.0.norm1.weight      [144]
hiera.blocks.0.norm1.bias        [144]
hiera.blocks.0.attn.qkv.weight   [432, 144]     # 3*144=432
hiera.blocks.0.attn.qkv.bias     [432]
hiera.blocks.0.attn.proj.weight  [144, 144]
hiera.blocks.0.attn.proj.bias    [144]
hiera.blocks.0.norm2.weight      [144]
hiera.blocks.0.norm2.bias        [144]
hiera.blocks.0.mlp.fc1.weight    [576, 144]     # dim_out*4=576 (MLP uses dim_out)
hiera.blocks.0.mlp.fc1.bias      [576]
hiera.blocks.0.mlp.fc2.weight    [144, 576]     # PyTorch key: .mlp.layers.0/1
hiera.blocks.0.mlp.fc2.bias      [144]
# ... repeat for blocks 0-1 (stage 0)

# Block 2 (first of stage 1): has dim_proj and different dims
hiera.blocks.2.norm1.weight      [144]           # pre-norm uses input dim
hiera.blocks.2.norm1.bias        [144]
hiera.blocks.2.attn.qkv.weight   [864, 144]     # 3*288=864, but input is 144
hiera.blocks.2.attn.qkv.bias     [864]
hiera.blocks.2.attn.proj.weight  [288, 288]
hiera.blocks.2.attn.proj.bias    [288]
hiera.blocks.2.proj.weight       [288, 144]      # Dimension change projection
hiera.blocks.2.proj.bias         [288]
hiera.blocks.2.norm2.weight      [288]
hiera.blocks.2.norm2.bias        [288]
hiera.blocks.2.mlp.fc1.weight    [1152, 288]
hiera.blocks.2.mlp.fc1.bias      [1152]
hiera.blocks.2.mlp.fc2.weight    [288, 1152]
hiera.blocks.2.mlp.fc2.bias      [288]
# ... continue for all 48 blocks
```

### 14.2 FPN Neck (~0.5M params)

```
fpn.convs.0.weight               [256, 1152, 1, 1]   # backbone_channel_list[0]=1152 → 256
fpn.convs.0.bias                 [256]
fpn.convs.1.weight               [256, 576, 1, 1]    # backbone_channel_list[1]=576 → 256
fpn.convs.1.bias                 [256]
fpn.convs.2.weight               [256, 288, 1, 1]    # backbone_channel_list[2]=288 → 256
fpn.convs.2.bias                 [256]
fpn.convs.3.weight               [256, 144, 1, 1]    # backbone_channel_list[3]=144 → 256
fpn.convs.3.bias                 [256]
```

### 14.3 Shared Tensors (same names as SAM3)

```
# SAM Prompt Encoder
sam_pe.pe_gaussian               [2, 128]
sam_pe.point_embeddings.{0-3}    [256, 1] × 4
sam_pe.not_a_point               [256, 1]
sam_pe.no_mask                   [256, 1]
sam_pe.mask_ds.{0-6}.*           (mask downscaling convs)

# SAM Mask Decoder
sam_dec.iou_token                [1, 256]
sam_dec.mask_tokens              [4, 256]
sam_dec.obj_score_token          [1, 256]
sam_dec.twoway.{0-1}.*           (TwoWay blocks)
sam_dec.final_attn.*
sam_dec.final_norm.*
sam_dec.upscale.*
sam_dec.conv_s0.*                (high-res feature convs)
sam_dec.conv_s1.*
sam_dec.hyper.{0-3}.{0-2}.*     (hypernetwork MLPs)
sam_dec.iou_head.{0-2}.*
sam_dec.obj_score_head.*

# Memory Encoder
mem_enc.ds.{j}.*                (mask downsampler: 4 Conv2d stages k=3,s=2,p=1 + final 1×1.
#                                PyTorch indices are non-contiguous: j∈{0,1,3,4,6,7,9,10,12}
#                                because GELU layers at indices 2,5,8,11 have no params.
#                                9 parameter groups total: 5 Conv2d + 4 LayerNorm2d)
mem_enc.pix_feat_proj.*
mem_enc.fuser.{0-1}.*           (2 CXBlocks)
mem_enc.out_proj.*
mem_enc.tpos_enc                [7, 1, 1, 64]

# Memory Attention
mem_attn.layers.{0-3}.*
mem_attn.norm.*                 (final norm)

# Object Pointer
obj_ptr_proj.*
no_obj_ptr                      [1, 256]
obj_ptr_tpos_proj.*

# SAM2-specific top-level
no_mem_embed                    [1, 1, 256]
no_mem_pos_enc                  [1, 1, 256]
no_obj_embed_spatial            [1, 64]
trk_mask_ds.*                   [1, 1, 4, 4]
```

---

## 15. Implementation Order

### Phase 0: Weight Conversion

**Step 0.1: Inspect SAM2 checkpoint**
- Load SAM2 checkpoint in Python
- List all keys + shapes
- Identify mapping to ggml tensor names
- Count total tensors (expect ~250-350 depending on model size)

**Step 0.2: Write `convert_sam2_to_ggml.py`**
- Implement header writing with SAM2 magic + hyperparameters
- Implement tensor renaming (PyTorch → ggml names, Section 14)
- Handle Hiera block flattening (stages → flat index)
- Handle dimension projection tensors at stage transitions
- Skip training-only tensors
- **Verify:** Convert SAM2-Large, print summary (tensor count, file size ~0.8 GB f16)

**Step 0.3: SAM2 hyperparameter loading in C++**
- Implement `sam2_load_hparams()`: read all SAM2-specific fields from header
- Add `sam3_model_type` dispatch in `sam3_load_model()`
- Print loaded hyperparameters for verification
- **Verify:** Load SAM2 .ggml file, print all hparams correctly

### Phase 1: Hiera Backbone

**Step 1.1: Hiera metadata precomputation**
- Implement `sam2_precompute_hiera_metadata()`: compute stage_ends, per-block dim_in/dim_out, window_size, has_q_stride
- **Verify:** Print block table (block_idx, stage, dim_in, dim_out, window_size, q_stride)

**Step 1.2: Tensor registration**
- Implement `sam2_register_tensors()`: create all Hiera + FPN ggml tensors with correct shapes
- Implement `sam3_register_shared_tensors()`: factor out shared tensor registration (sam_pe, sam_dec, mem_enc, mem_attn, obj_ptr)
- **Verify:** Load model, verify all tensors present and shapes match checkpoint

**Step 1.3: PatchEmbed**
- Implement as `ggml_conv_2d(3 → embed_dim, k=7, s=4, p=3)`
- **Verify:** Compare output [B, 256, 256, embed_dim] against Python

**Step 1.4: Positional embedding**
- Implement `sam2_hiera_pos_embed()`: bicubic interpolate background PE + tile window PE
- Precompute on CPU, upload as input tensor
- **Verify:** Compare PE tensor against Python `_get_pos_embed()`

**Step 1.5: Window partition/unpartition**
- Implement `sam2_window_partition()` and `sam2_window_unpartition()` using ggml view/permute ops
- Handle padding for non-divisible spatial sizes
- **Verify:** Round-trip test (partition → unpartition = identity)

**Step 1.6: Single Hiera block**
- Implement `sam2_hiera_block_forward()`: norm → attn → residual → norm → MLP → residual
- Handle Q-pooling case (MaxPool2d on Q before attention)
- Handle dimension projection case (Linear + MaxPool on shortcut)
- **Verify:** Compare block 0 output against Python

**Step 1.7: Multi-scale attention**
- Implement `sam2_multiscale_attention()`: QKV projection → optional Q pool → SDPA → output projection
- **Verify:** Compare attention output for window and global blocks

**Step 1.8: Full Hiera forward**
- Implement `sam2_build_hiera_graph()`: PatchEmbed → PE → all blocks → collect intermediates
- Return 4 intermediate feature tensors (one per stage end)
- **Verify:** Compare all 4 intermediate outputs against Python backbone output

### Phase 2: FPN Neck

**Step 2.1: FPN lateral + top-down**
- Implement `sam2_build_fpn_neck_graph()`: 4 lateral 1×1 convs, nearest 2× upsample for top-down
- Apply PositionEmbeddingSine to each output level
- For nearest 2× upsample: use `ggml_upscale(ctx, tensor, 2, GGML_SCALE_MODE_NEAREST)`
  (already used by SAM3 in its pixel decoder, with CPU/Metal/CUDA support)
- Store outputs into `state.neck_trk[0..3]` and `state.neck_trk_pe[0..3]`
- **Verify:** Compare each FPN level output against Python

**Step 2.2: Full image encoding**
- Implement `sam2_encode_image_hiera()`: preprocess → Hiera → FPN → state
- Wire into `sam3_encode_image()` dispatch
- **Verify:** Encode a test image, compare state.neck_trk tensors against Python

### Phase 3: Shared Tensor Registration Refactor

**Step 3.1: Factor out shared registration**
- Extract SAM PE, SAM decoder, memory encoder, memory attention, and object pointer
  tensor registration into `sam3_register_shared_tensors()` callable from both SAM2 and SAM3 loaders
- **Verify:** Both SAM2 and SAM3 models load correctly after refactoring

**Step 3.2: Parameterize spatial dimensions**
- Audit all downstream functions that assume 72×72 spatial size
- Replace with dimension reads from tensor shapes or `hp.hiera_feat_size()` / `hp.n_img_embd()`
- Key functions: `sam3_propagate_single`, `sam3_encode_memory`, `sam3_segment_pvs`
- **Verify:** SAM3 still works identically after parameterization

### Phase 4: SAM2 Image Segmentation (PVS)

**Step 4.1: SAM2 image preprocessing**
- Implement `sam2_preprocess_image()` with ImageNet normalization:
  mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- Target size: 1024×1024 (from hparams.img_size)
- **Verify:** Compare preprocessed tensor against Python SAM2Transforms output

**Step 4.2: SAM2 mask decoder variant (if needed)**
- If `pred_obj_scores=False` in checkpoint: implement 5-token layout in `sam2_build_sam_dec_graph()`
- If `iou_prediction_use_sigmoid=False`: skip IoU sigmoid in decoder
- For SAM2.1 configs (pred_obj_scores=True, pred_obj_scores_mlp=True): existing SAM3 decoder works
- **Verify:** Compare mask decoder output for both token layouts

**Step 4.3: PVS with SAM2 backbone**
- Load SAM2 model → encode image → run `sam3_segment_pvs()` with point/box prompts
- Verify coordinate normalization uses 1024 (not 1008)
- **Verify:** Compare masks against Python SAM2ImagePredictor output

### Phase 5: SAM2 Video Tracking

**Step 5.1: Memory encoder parameterization**
- Parameterize mask interpolation target: `feat_size × 16` (1024 for SAM2, 1152 for SAM3)
- Verify pix_feat reads from `neck_trk[2]` (same index for both SAM2 and SAM3)
- Parameterize sigmoid_scale/bias from hparams
- **Verify:** Compare memory encoder output [64, 64, 64] against Python

**Step 5.2: First-frame handling**
- Verify `sam3_propagate_single()` handles empty memory correctly
- SAM2 uses `no_mem_embed` as dummy token when no prior memory exists
- Ensure the conditioning frame path (SAM decode without memory attention) works
- **Verify:** First frame mask matches Python

**Step 5.3: Visual tracker with SAM2**
- Create visual tracker → add instance → propagate frames
- **Verify:** Track a 10-frame sequence, compare masks against Python SAM2VideoPredictor

**Step 5.4: End-to-end video test**
- Process 30+ frames with multiple objects
- Compare all propagated masks per frame against Python
- Test instance addition and refinement mid-video
- Test non-overlapping constraint with 2+ objects

### Phase 6: Polish

**Step 6.1: Guard SAM3-only functions**
- `sam3_segment_pcs()` → check model_type, warn if SAM2
- `sam3_create_tracker()` (text-prompted) → check model_type

**Step 6.2: Documentation**
- Update `sam3.h` comments to note SAM2 compatibility
- Add `sam3_get_model_type()` declaration

**Step 6.3: Test both model types**
- Run full test suite with both SAM2 and SAM3 models
- Verify SAM3 is completely unaffected by the changes

---

## 16. Appendix: SAM2 Tensor Shape Reference

### Input image 1024×1024, Hiera Large (embed_dim=144, stages=[2,6,36,4], q_pool=3, scalp=1)

```
Input:                     [1, 3, 1024, 1024]

After PatchEmbed (k=7,s=4,p=3):
                           [1, 256, 256, 144]     (BHWC format in Hiera)

Hiera backbone stage outputs (q_pool at blocks 2, 8, 44):
  Stage 0 (blocks 0-1):   [1, 144, 256, 256]     (no pooling)
  Stage 1 (blocks 2-7):   [1, 288, 128, 128]     (Q-pool at block 2: 256→128)
  Stage 2 (blocks 8-43):  [1, 576, 64, 64]       (Q-pool at block 8: 128→64)
  Stage 3 (blocks 44-47): [1, 1152, 32, 32]      (Q-pool at block 44: 64→32)

FPN raw outputs (4 levels, NEAREST interp for top-down):
  Level 0:                 [1, 256, 256, 256]     (stride 4, lateral only)
  Level 1:                 [1, 256, 128, 128]     (stride 8, lateral only)
  Level 2:                 [1, 256, 64, 64]       (stride 16, lateral + top-down)
  Level 3:                 [1, 256, 32, 32]       (stride 32, lateral only — first iteration, prev=None)

After scalp=1 (discard level 3 → 3 output levels):
  backbone_fpn[0]:         [1, 256, 256, 256]     (stride 4)
  backbone_fpn[1]:         [1, 256, 128, 128]     (stride 8)
  backbone_fpn[2]:         [1, 256, 64, 64]       (stride 16) ← MAIN FEATURE

SAM image embedding:       [1, 256, 64, 64]       (= backbone_fpn[2])
High-res s1:               [1, 64, 128, 128]      (= conv_s1(backbone_fpn[1]), 256→64)
High-res s0:               [1, 32, 256, 256]      (= conv_s0(backbone_fpn[0]), 256→32)

Prompt encoding:
  Sparse (1 point):        [1, 2, 256]            (point + padding)
  Dense (no mask):         [1, 256, 64, 64]

SAM decoder:
  TwoWay input:            [1, 4096, 256]         (64×64=4096 tokens)
  Output tokens:           [1, 6+N_prompt, 256]   (obj_score + iou + 4 mask + prompts)
  Upscaling step 1:        [1, 64, 128, 128]      (dc1 + conv_s1 skip)
  Upscaling step 2:        [1, 32, 256, 256]      (dc2 + conv_s0 skip)
  Low-res masks:           [1, 4, 256, 256]       (hyper_in @ upscaled)
  IoU predictions:         [1, 4]
  Object score:            [1, 1]

After upscaling to original:
  → [1, 1 or 3, 1024, 1024]

Memory encoder:
  pix_feat input:          [1, 256, 64, 64]       (= backbone_fpn[2])
  Mask input:              [1, 1, 1024, 1024]     (pred_masks_high_res)
  Mask after sigmoid+scale:[1, 1, 1024, 1024]     (sigmoid * 20.0 - 10.0)
  Mask downsampled:        [1, 256, 64, 64]       (4× stride-2: 1024→512→256→128→64)
  Fused:                   [1, 256, 64, 64]       (pix_proj + mask + 2×CXBlock)
  Memory output:           [1, 64, 64, 64]        (out_proj 256→64)

Memory attention (feat_sizes=[64,64]):
  Current tokens:          [4096, 1, 256]          (64×64)
  Memory tokens:           [N_mem×4096, 1, 64]     (N_mem ≤ 7)
  Obj ptr tokens:          [N_ptr×4, 1, 64]        (N_ptr ≤ 16, 256→4×64)
  Output:                  [4096, 1, 256]
  Reshaped:                [1, 256, 64, 64]
```

---

## 17. Appendix: SAM2 Hyperparameters by Variant

| Parameter | Tiny | Small | Base+ | Large |
|---|---|---|---|---|
| `embed_dim` | 96 | 96 | 112 | 144 |
| `num_heads` | 1 | 1 | 2 | 2 |
| `stages` | [1,2,7,2] | [1,2,11,2] | [2,3,16,3] | [2,6,36,4] |
| Total blocks | 12 | 16 | 24 | 48 |
| `global_att_blocks` | [5,7,9] | [7,10,13] | [12,16,20] | [23,33,43] |
| `q_pool` | 3 | 3 | 3 | 3 |
| `window_spec` | [8,4,14,7] | [8,4,14,7] | [8,4,14,7] | [8,4,16,8] |
| Backbone channels | [768,384,192,96] | [768,384,192,96] | [896,448,224,112] | [1152,576,288,144] |
| FPN d_model | 256 | 256 | 256 | 256 |
| FPN top_down_levels | [2,3] | [2,3] | [2,3] | [2,3] |
| Memory attn layers | 4 | 4 | 4 | 4 |
| Memory attn d_model | 256 | 256 | 256 | 256 |
| Memory out_dim | 64 | 64 | 64 | 64 |
| num_maskmem | 7 | 7 | 7 | 7 |
| max_obj_ptrs | 16 | 16 | 16 | 16 |
| SAM decoder depth | 2 | 2 | 2 | 2 |
| SAM num_multimask | 3 | 3 | 3 | 3 |
| sigmoid_scale | 20.0 | 20.0 | 20.0 | 20.0 |
| sigmoid_bias | -10.0 | -10.0 | -10.0 | -10.0 |
| image_size | 1024 | 1024 | 1024 | 1024 |
| Approx params | ~39M | ~46M | ~81M | ~224M |

All variants share identical SAM heads, memory encoder, memory attention,
and object pointer architecture — only the backbone and FPN lateral conv
dimensions change.

**Note on `window_pos_embed_bkg_spatial_size`:** Tiny, Small, and Large all
configure this to `[7, 7]`.  Base+ does NOT override the code default of
`[14, 14]`.  The binary header stores this as `hiera_pos_embed_bkg_h/w`
so each variant is handled correctly.

---

## 18. Appendix: Critical Differences Audit (SAM2 vs SAM3)

This section documents every operation-level difference found between SAM2 and
SAM3 that affects correctness.  Items marked **MUST FIX** require code changes
beyond what Section 3 "Fully Reused" claimed.

### 18.1 Image Preprocessing — **MUST FIX**

| Parameter | SAM3 | SAM2 |
|---|---|---|
| Target size | 1008×1008 | 1024×1024 |
| Mean | [0.5, 0.5, 0.5] | [0.485, 0.456, 0.406] (ImageNet) |
| Std | [0.5, 0.5, 0.5] | [0.229, 0.224, 0.225] (ImageNet) |
| Output range | [-1, +1] | ~[-2.1, +2.6] |

**Action:** Create `sam2_preprocess_image()` with ImageNet normalization.
SAM3's `sam3_preprocess_image()` remains unchanged.  The dispatch happens
inside `sam3_encode_image()` / `sam2_encode_image_hiera()`.

```cpp
// SAM2 preprocessing
static std::vector<float> sam2_preprocess_image(const sam3_image & img, int target_size) {
    // 1. Resize to target_size × target_size (bilinear)
    // 2. Convert to float [0, 1]: pixel / 255.0
    // 3. Normalize per-channel:
    //    R: (v - 0.485) / 0.229
    //    G: (v - 0.456) / 0.224
    //    B: (v - 0.406) / 0.225
    // 4. CHW layout
}
```

### 18.2 Mask Decoder Token Ordering — **MUST FIX**

SAM3 always prepends `obj_score_token` (6 tokens total: obj + iou + 4 masks).
SAM2's behavior depends on config flags:

| Config | SAM2 behavior | Token order |
|---|---|---|
| `pred_obj_scores=True` | obj_score_token present | [obj(0), iou(1), masks(2-5)] |
| `pred_obj_scores=False` | obj_score_token absent | [iou(0), masks(1-4)] |

For SAM2.1 configs, `pred_obj_scores=True` and `pred_obj_scores_mlp=True`,
so the token ordering **matches SAM3's fixed layout**.  However, the code
must handle both cases for older SAM2 checkpoints.

**Action:** SAM2's mask decoder function (`sam2_build_sam_dec_graph`) must
respect the `pred_obj_scores` flag in the header.  When `pred_obj_scores=0`,
use 5-token layout (no obj_score_token) and set `obj_score = sigmoid(10.0)`
as default (object assumed present).

### 18.3 IoU Head Sigmoid — No Change Needed

SAM2.1 HieraL config sets `iou_prediction_use_sigmoid: True`, matching
SAM3's always-on sigmoid behavior.

**Action:** No change needed.  Both models apply sigmoid to IoU predictions.

### 18.4 Object Score Head: Linear vs MLP — Verify

SAM2 with `pred_obj_scores_mlp=True` uses a 3-layer MLP (matching SAM3).
SAM2 with `pred_obj_scores_mlp=False` uses a single `nn.Linear(256, 1)`.

**Action:** Add `pred_obj_scores_mlp` to hparams.  The conversion script
must check which variant the checkpoint uses (1 weight tensor vs 3).
The C++ code uses different weight arrays depending on the flag.

### 18.5 Memory Encoder Mask Interpolation — **MUST FIX**

SAM3 interpolates the mask to 1152×1152 before the 4-stage downsampler
(1152 → 576 → 288 → 144 → 72, matching 72×72 backbone spatial).

SAM2's effective backbone spatial size is **64×64** (after scalp=1 discards
the 32×32 level).  The mask must reach 64×64 after 4 stages of stride-2
downsampling.  Input to downsampler must be `64 × 2⁴ = 1024×1024`.

Since SAM2's `pred_masks_high_res` is interpolated to `image_size=1024×1024`,
**no extra interpolation step is needed** — the high-res mask goes directly
to the downsampler.

**Action:** `sam3_encode_memory()` must parameterize the interpolation
target size:
- SAM3: mask logits (288×288) → sigmoid+scale → interpolate to 1152×1152 → downsample → 72×72
- SAM2: mask logits (256×256) → sigmoid+scale → interpolate to 1024×1024 → downsample → 64×64

The formula: `INTERPOL = feat_size × 16` where `feat_size` = 72 (SAM3) or
64 (SAM2).

### 18.6 Memory Attention: pos_enc_at_attn Flag — Verify

SAM2 MemoryAttentionLayer has configurable PE injection flags:

| Flag | SAM2.1 default | SAM3 behavior |
|---|---|---|
| `pos_enc_at_attn` | `False` | No query_pos in self-attn (uses RoPE only) |
| `pos_enc_at_cross_attn_keys` | `True` | PE added to memory K before projection |
| `pos_enc_at_cross_attn_queries` | `False` | No query_pos in cross-attn Q |

SAM3 matches the SAM2.1 defaults: no explicit query_pos in self-attention
(RoPE handles it), PE added to memory keys only.

**Action:** No code change needed for SAM2.1 configs.  The PE injection
behavior is already correct in SAM3's implementation.

### 18.7 Object Pointer Reshaping: 256→4×64 — Verify

SAM2 conditionally splits pointers: `obj_ptrs.reshape(-1, B, C//mem_dim, mem_dim)`
only when `mem_dim < C` (i.e., 64 < 256 → True for all SAM2 configs).

SAM3 always splits 256→4×64.

Since all SAM2 configs have `mem_dim=64`, the split always happens,
matching SAM3's behavior.

**Action:** No change needed.

### 18.8 dynamic_multimask_via_stability — SAM2 Feature

SAM2 implements a runtime stability check: if single-mask output has low
stability (IoU between mask>0.05 and mask>-0.05 < 0.98), it falls back to
the best multimask output.

This is controlled by `dynamic_multimask_via_stability` (default: `False`
in configs, but can be enabled).

**Action:** Implement `sam3_select_best_mask()` stability logic if not
already present.  The current SAM3 code already has stability score
computation — verify it matches SAM2's formula:
```
stability = IoU(mask > delta, mask > -delta)  where delta = 0.05
threshold = 0.98
```

### 18.9 use_multimask_token_for_obj_ptr — **MUST HANDLE**

Controls which SAM output token becomes the object pointer:
- `False`: use `mask_tokens_out[:, 0]` (first mask token)
- `True` (SAM2.1 HieraL default): use `mask_tokens_out[:, 1:]` (all multimask tokens, best selected by IoU)

SAM3 always uses `mask_tokens_out[:, 0]`.

SAM2.1 HieraL sets `use_multimask_token_for_obj_ptr: true`.  When
`multimask_output=True`, the mask decoder's `forward()` selects the best
among multimask tokens (indices 1-3) based on IoU.  The selected token
becomes the object pointer.

**Action:** Add hparam `use_multimask_token_for_obj_ptr`.  The object
pointer extraction code in `sam3_propagate_single()` must select the
token corresponding to the best mask's IoU score when this flag is set.

### 18.10 Reverse Video Tracking — SAM2 Feature

SAM2's video predictor supports bidirectional tracking with a `reverse`
flag that flips the sign of temporal positional encoding:
```python
tpos_sign_mul = -1 if track_in_reverse else 1
```

**Action:** SAM3's visual tracker should already handle forward-only
tracking.  For SAM2 compatibility, the temporal PE sign should be
parameterizable.  This is a minor change in `sam3_propagate_single()`.

### 18.11 Coordinate Normalization in Video API

SAM2 `add_new_points_or_box()` normalizes user coordinates:
```python
points = points / [video_W, video_H]  # pixel → [0, 1]
points = points * self.image_size      # [0, 1] → [0, 1024]
```

SAM3's `sam3_tracker_add_instance()` performs equivalent normalization
through `sam3_pvs_params` which expects coordinates in the original
image space.

**Action:** Verify that coordinate normalization in `sam3_segment_pvs()`
and `sam3_tracker_add_instance()` produces identical results for SAM2's
1024×1024 internal resolution vs SAM3's 1008×1008.

### 18.12 no_mem_embed on First Frame — **MUST HANDLE**

SAM2.1 HieraL config sets `directly_add_no_mem_embed: true`.

When this is `True` and no memories exist (initial conditioning frame):
```python
pix_feat_with_mem = current_vision_feats[-1] + self.no_mem_embed
```
This **bypasses memory attention entirely** and just adds the no_mem_embed
vector directly to the current features.  No transformer layers are run.

When `False` (older configs): a dummy `no_mem_embed` token is used as a
single-token memory input to the memory attention transformer.

**Action:** When `directly_add_no_mem_embed=True` and memory bank is
empty, `sam3_propagate_single()` must skip the memory attention subgraph
and instead add `no_mem_embed` (broadcast to all spatial positions) to
the current features before passing to the SAM decoder.  This is a
**significant control flow difference** from SAM3.

### 18.12b Additional SAM2.1 Config Flags — **MUST HANDLE**

The SAM2.1 HieraL config reveals several flags that differ from SAM3
defaults and affect the forward pass:

| Flag | SAM2.1 Value | SAM3 Behavior | Impact |
|---|---|---|---|
| `use_mlp_for_obj_ptr_proj` | `true` | 3-layer MLP (matches) | None |
| `fixed_no_obj_ptr` | `true` | Modulate obj_ptr by presence score | Pointer zeroed when obj absent |
| `use_signed_tpos_enc_to_obj_ptrs` | `true` | N/A | Temporal PE can be negative |
| `only_obj_ptrs_in_the_past_for_eval` | `true` | N/A | Skip future pointers |
| `multimask_output_for_tracking` | `true` | Single mask during tracking | 3 masks, best selected |
| `multimask_min_pt_num` | `0` | 1 | Multimask even with 0 points |
| `multimask_max_pt_num` | `1` | 1 | Same |
| `use_mask_input_as_output_without_sam` | `true` | N/A | Skip SAM if mask input given |
| `multimask_output_in_sam` | `true` | N/A (default: false) | Master switch for multimask output |
| `proj_tpos_enc_in_obj_ptrs` | `true` | Already implemented | Temporal PE projected |

**Action:** Add all these as hparams read from the binary header.
The conversion script must extract them from the Hydra config or
checkpoint metadata.  The most impactful are:
- `fixed_no_obj_ptr`: changes `obj_ptr = lambda * obj_ptr` to
  `obj_ptr = lambda * obj_ptr + (1-lambda) * no_obj_ptr`
- `multimask_output_for_tracking`: only affects correction clicks on already-tracked
  frames (not pure propagation — see 18.12c)
- `directly_add_no_mem_embed`: skip memory attention on first frame
- `use_mask_input_as_output_without_sam`: skip both memory attention AND SAM decoder
  when a conditioning mask is provided (see 18.12c)

### 18.12c Multimask and Mask-Input Bypass — Critical Control Flow

**Multimask output decision (`_use_multimask`):**

Multimask (3 masks, best selected by IoU) is enabled ONLY when ALL three conditions hold:
1. `multimask_output_in_sam = True` (master switch — default: False, SAM2.1: True.
   **This flag MUST be stored in the binary header.** Without it, multimask never fires.)
2. `is_init_cond_frame = True` OR `multimask_output_for_tracking = True`
3. `multimask_min_pt_num <= num_pts <= multimask_max_pt_num`

**Critical consequence for pure propagation:** During `propagate_in_video`, there are
no point inputs (`point_inputs=None` → `num_pts=0`). Since `num_pts=0 < multimask_min_pt_num`
(default 1, SAM2.1 config: 0), condition 3 ALWAYS fails for propagation frames when
`multimask_min_pt_num >= 1`. Even with SAM2.1's `multimask_min_pt_num=0`, during pure
propagation `num_pts=0` satisfies `0 <= 0 <= 1`, so multimask CAN fire — but only when
`multimask_output_in_sam=True` AND `multimask_output_for_tracking=True`.

**In practice:** SAM2.1 HieraL has `multimask_min_pt_num=0` and
`multimask_output_for_tracking=True`, so propagation frames DO use multimask output.
The best mask is always selected by `argmax(ious)` before storage.

**`use_mask_input_as_output_without_sam` — conditioning frame bypass:**

When `mask_inputs is not None` (user provides a ground-truth mask) AND this flag is True:
1. Memory attention is SKIPPED (no `_prepare_memory_conditioned_features`)
2. SAM decoder is SKIPPED (no `_forward_sam_heads`)
3. The mask is directly converted to logits: `mask * 20.0 + (-10.0)` (same sigmoid scale/bias)
4. IoU predictions are set to all-ones (dummy)
5. If `use_obj_ptrs_in_encoder=True`: SAM decoder IS called once, but ONLY to extract
   the `obj_ptr` token — mask outputs from this call are discarded
6. Memory encoder still runs to store the mask in the memory bank

This is the "trust the input mask" shortcut for conditioning frames.  All shipped SAM2/2.1
configs set this to True.

**Implementation action:** `sam3_propagate_single()` must check if the current frame
is a conditioning frame with a provided mask.  If so, bypass memory attention and SAM
decoder, directly produce logits from the mask, and only call the decoder for obj_ptr
extraction when needed.

### 18.13 Memory Attention Final Norm Tensor

SAM2's `MemoryAttention` has a final `self.norm` (LayerNorm) applied
after all 4 layers, stored as a separate module:
```python
memory_attention.norm.weight  # [256]
memory_attention.norm.bias    # [256]
```

This is separate from the per-layer norms.  The plan's tensor mapping
(Section 14.3) already includes `mem_attn.norm.*`.

**Action:** Verify that `sam3_build_mem_attn_graph()` applies a final
LayerNorm after the last layer.  Check if SAM3 already has this
(from the agent audit: yes, lines 5091-5093 of sam3.cpp).  Ensure the
weight tensor name matches (`mem_attn.norm.weight/bias`).

### 18.14 Summary of Required Code Changes (beyond backbone/neck)

| Change | Scope | New function? |
|---|---|---|
| SAM2 image preprocessing (ImageNet norm) | `sam2_preprocess_image()` | Yes |
| Mask interpolation size parameterization | `sam3_encode_memory()` | Modify |
| Pred_obj_scores flag in mask decoder | `sam2_build_sam_dec_graph()` | Yes (variant) |
| IoU sigmoid flag | hparams + mask decoder | Modify |
| Object score head Linear vs MLP | conversion script + loader | Conditional |
| Coordinate normalization for 1024 vs 1008 | `sam3_segment_pvs()` | Verify |
| First-frame no_mem_embed handling | `sam3_propagate_single()` | Modify |
| Non-overlapping mask constraint | `sam3_propagate_single()` | Modify |
| Mask binarization from points | `sam3_propagate_single()` | Modify |
| Multimask output for tracking | `sam3_propagate_single()` | Modify |
| Multimask token for obj_ptr selection | `sam3_propagate_single()` | Modify |
| fixed_no_obj_ptr modulation | `sam3_propagate_single()` | Modify |
| 0.1 × curr_pos scaling in mem_attn | `sam3_build_mem_attn_graph()` | Verify |
| num_k_exclude_rope for obj_ptr tokens | `sam3_build_mem_attn_graph()` | Verify |
| rope_k_repeat for cross-attn K length | `sam3_build_mem_attn_graph()` | Verify |
| multimask_output_in_sam master switch | `sam3_propagate_single()` | Modify |

### 18.15 Updated LOC Estimate

| Category | Original Estimate | Revised Estimate |
|---|---|---|
| Backbone + neck (Hiera, FPN) | ~680 | ~680 (unchanged) |
| Model loading + tensor registration | ~400 | ~480 (+80 for new flags) |
| Image preprocessing (SAM2-specific) | 0 | ~60 (new function) |
| Mask decoder (SAM2 variant) | 0 | ~80 (conditional token ordering) |
| Memory encoder parameterization | ~30 | ~70 (interp + non-overlap + binarize) |
| Tracker parameterization (obj_ptr, multimask) | 0 | ~80 (flag-driven logic + multimask_output_in_sam) |
| convert_sam2_to_ggml.py | ~300 | ~380 (+80 for config flags) |
| **Revised total** | **~1500** | **~1930** |

---

## Summary of Changes

| Category | Files Touched | LOC Estimate |
|---|---|---|
| New structs (Hiera, FPN) | sam3.cpp | ~120 |
| hparams additions (SAM2 fields + flags) | sam3.cpp | ~110 |
| sam2_build_hiera_graph + helpers | sam3.cpp | ~400 |
| sam2_build_fpn_neck_graph | sam3.cpp | ~80 |
| sam2_encode_image_hiera | sam3.cpp | ~100 |
| sam2_preprocess_image (ImageNet norm) | sam3.cpp | ~60 |
| sam2_build_sam_dec_graph (token layout variant) | sam3.cpp | ~80 |
| Model loading dispatch | sam3.cpp | ~150 |
| Tensor registration (SAM2) | sam3.cpp | ~200 |
| Shared tensor refactor | sam3.cpp | ~50 |
| Memory encoder parameterization | sam3.cpp | ~70 |
| Tracker flag parameterization | sam3.cpp | ~80 |
| Spatial dimension parameterization | sam3.cpp | ~40 |
| Public API additions | sam3.h | ~10 |
| convert_sam2_to_ggml.py | new file | ~380 |
| **Total new/modified** | | **~1930 LOC** |

The existing SAM3 code (~10,000 lines) remains **functionally untouched** with these
surgical modifications:
1. A model_type dispatch at the top of `sam3_encode_image()`
2. Parameterization of sigmoid_scale/bias and mask interpolation target in `sam3_encode_memory()`
3. Non-overlapping constraint + mask binarization flags in memory encoding path
4. Factoring shared tensor registration out of the SAM3 loader
5. Reading spatial sizes from tensor shapes instead of hardcoding 72
6. IoU sigmoid flag in mask decoder (conditional skip)
7. Multimask output + best-mask selection during tracking (gated by `multimask_output_in_sam`)
8. Object pointer token selection (first vs best-IoU) based on `use_multimask_token_for_obj_ptr`
9. First-frame bypass of memory attention when `directly_add_no_mem_embed=True`
10. Verify `num_k_exclude_rope` and `rope_k_repeat` in memory attention cross-attention
