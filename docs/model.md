# Model

ColonoWorld is the NVIDIA Cosmos3-Nano video model (`cosmos_framework`, image-to-video,
Wan2.2 VAE) fine-tuned with joint conditioning. The Cosmos code and weights are an external
dependency; ColonoWorld adds the modules below and trains only them.

| Module | Parameters | Role |
|---|---|---|
| generation LoRA | rank 16, alpha 32 on `q/k/v/o_proj_moe_gen` | adapts the generation expert |
| `camera_adapter` | `Linear(24, 4096)` | Pluecker rays (2x2 patches) added to every target video token |
| `camera_block_encoders` | 36 x `Linear(24, 4096)` | rays added before self-attention in every block (latent frames 1..) |
| `tube_stream` | input projection, time/height/width/type embeddings, segment embedding (7 x 4096) | tube latent as appended token rows |
| `mask_stream` | same layout, no segment embedding | palette mask latent as appended token rows |
| `global_adapter` | `Linear(13, 4096)`, no bias | per-latent-frame flags (special light, tool, irrigation, class known) |

Camera rays: KB4 fisheye unprojection on the 16 px latent grid, `[o x d, d]` relative to the
first camera; moments are divided by 5.1 before the adapters. Every camera addition is bounded to
10 % of the RMS of the tokens it is added to.

Condition rows: the tube and mask videos are VAE-encoded like the generated video and appended
as extra rows (one row per target token) with the same mRoPE position as the target token at
that latent time and cell. Attention is dense and two-way; condition rows carry no noise.

Guidance (inside each denoiser step, text guidance unchanged):

- tube: `v = v0 + s_t (v_T - v0)`; `s_t = 1` (default) is a single forward;
- tube + mask: `v = v0 + s_t (v_T - v0) + s_m (v_TM - v_T)` with `s_m = 3` by default;
  `v_TM` has all conditions, `v_T` drops the mask rows and global flags, `v0` keeps only camera and text.

## Weights

```
${COSMOS_WEIGHTS}/model/            Cosmos3-Nano network (DCP directory or safetensors)
${COSMOS_WEIGHTS}/Wan2.2_VAE.pth    video VAE
${COSMOS_WEIGHTS}/text_tokenizer/   text tokenizer
${COLONOWORLD_CKPT}                  ColonoWorld weights (colonoworld.pt) {"ema": ..., "raw": ...}
```

`weights.checkpoint` is one file with all ColonoWorld tensors; `weights.use: ema` (default)
selects the EMA weights. Training produces the camera stage and the joint stage separately;
`python -m colonoworld.models.checkpoint --out colonoworld.pt camera_stage.pt joint_stage.pt`
merges them (later files override earlier ones).

## Python API

```python
from colonoworld.utils.config import load_config
from colonoworld.models import load_model

model = load_model(load_config("configs/model.yaml"))
video = model.generate(
    first_frame,            # uint8 (832, 1104, 3)
    cameras,                # (189, 4, 4) camera-to-world
    tube_video,             # uint8 (189, 832, 1104, 3), rendered along the cameras
    cond_mask=None,         # optional uint8 palette video (189, 832, 1104, 3)
    nbi=False,              # True: special light on all frames
    intrinsics=calibration, # KB4 calibration dict
    segment_ids=None,       # optional per-latent-frame segment ids (0..5, 6 = unknown)
    seed=0,
)                           # -> uint8 (189, 832, 1104, 3)
```

Keyword-only extras of `generate`:

| Keyword | Meaning |
|---|---|
| `intrinsics` | KB4 calibration dict for `cameras` (`K_native`, `D`, `native_to_clip`, `clip_size`, optional `field_radius_px`) |
| `field_mask` | optional (52, 69) boolean aperture; rays outside are zero |
| `rays` | precomputed Pluecker rays (48, 6, 52, 69) instead of `cameras` |
| `segment_ids` | per-latent-frame colon segment ids (0 cecum .. 5 rectum, 6 unknown); `segment_ids_from_labels` converts per-frame names |
| `mask_choice` | per mask channel `"mask"` / `"empty"` / `None`; default: channels present in `cond_mask` are `"mask"`, the rest `"empty"` |
| `mask_classes` | boolean (T, H, W) masks per channel used for the global flags (default: read from the palette) |
| `flags` | explicit (48, 13) global flags |
| `return_latents` | also return the final latent |
| `rng_seed` | seed of the global Python / NumPy / torch RNGs during the call (default `seed`) |

`nbi=False` leaves the light unspecified; `nbi=True` marks special light on every frame.
`seed` sets the initial noise; the global RNGs are seeded from `rng_seed` (default `seed`) for the
duration of the call, so a seed reproduces a sample exactly on the same hardware and software.

Defaults: 35 UniPC steps, shift 10, text guidance 1, tube CFG 1, mask CFG 3, generic caption.
One chunk runs on a single GPU with about 40 GB of memory (bf16 weights).
