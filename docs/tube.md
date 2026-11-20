# Tube

The tube is a free-form 3D model of the lumen wall, rendered from the camera path into a
*tube video* that conditions the generator (VAE-encoded and appended as condition rows). It is
not a fixed pipe: the centreline may bend sharply ahead of the camera, and each cross-section may
narrow, widen, flatten or fold. Training tubes, benchmark tubes and synthetic full-colon tubes use
the same representation (`colonoworld.tube.TubeModel`) and the same renderer.

## Representation

`TubeModel` (`colonoworld/tube/model.py`) stores a sweep of cross-sections: `centerline (N, 3)`,
per-section `frames (N, 3, 3)`, wall distances `radii (N, K)` at `angles (K,)`, `arclength (N,)`,
plus the triangle mesh (`vertices`, `faces`) and its texture chart (`chart`: arc length,
circumferential position, half perimeter). `save(path)` / `TubeModel.load(path)` use `.npz`.

A tube built for a video window is a *profile sweep* (`colonoworld/tube/wall_fit.py`): 24 knots
along an axis, each with a centre offset, a log radius and an elliptic anisotropy, interpolated
with PCHIP; haustral folds (crescents between three taeniae) indent the wall; the mesh is the open
side wall.

## Tube from a real clip

`build_tube_from_video(frame0, cameras, calibration, segment, key, cfg)`
(`colonoworld/tube/build_from_video.py`) reads only the clip's first frame:

- frame-0 lumen direction: the darkest blob of the radially normalised, specular-free first frame;
- axis: the camera path's principal direction (paths longer than two radii), otherwise the
  frame-0 lumen ray;
- centreline: smoothed camera path, shifted toward the side where the lumen is seen, bending
  toward the lumen ray ahead of the front-most camera;
- radius: per-segment median of the training tubes; folds: one deterministic draw (seeded by
  `key`) from the training fold patterns; the sweep is inflated so a 0.36 R ball around every
  camera fits inside.

Inputs: camera poses (`relative_c2w`, camera-to-world relative to frame 0, OpenCV axes,
translation in units of the first frame's median depth) from a monocular pose/depth estimator,
and the KB4 fisheye calibration (`colonoworld/tube/camera.py`).

```bash
export COLONOWORLD_DATA=/path/to/data   # see "Data files" below
python scripts/build_tube.py --config configs/tube.yaml --video clip.mp4 \
    --camera-path poses.npz --calibration camera.json --segment transverse --out outputs/tube
```

Outputs `tube.npz`, `tube_params.json`, `cameras.npy` and `tube.mp4`.

## Training tubes

Training tubes are fitted to each training window's wall and use the same representation and
renderer (`scripts/fit_training_tubes.py`):

0. **Seed** (`colonoworld/tube/seed_*.py`, `build_training_seed`), from the real clip and the
   camera/depth estimator outputs (MegaSaM `relative_c2w`, timestamps and first-frame depth):
   - keyframe depth (`seed_depth.keyframe_depth`): every 6th frame, Depth Anything (v1, ViT-L)
     disparity aligned to the poses by robust flow reprojection (frames +-3, +-6);
   - reference radius (`seed_radius.reference_radius`): causal centreline fit of the window
     (trajectory principal curve or frame-0 lumen axis) and the median first-frame depth
     distance to it, checked by a frame-0 vanishing-point gate;
   - fused wall evidence (`seed_depth.collect_evidence`): pose-gated keyframes every 12 frames,
     two-view triangulation fused with the aligned depth, dark-lumen free-space rays;
   - signed-distance field (`seed_sdf.reconstruct_sdf`): screened block fields, joint field,
     heat-kernel regularisation, support-aware envelope;
   - lumen carve (`seed_sdf.fit_lumen_frustum`, `carve_lumen`): one elliptical frustum from the
     largest dark component of every frame, united with the field, far cavities closed;
   - 24-knot profile fit (`seed_fit.seed_from_sdf`): station ellipses from radial exits of the
     SDF, folds from radial deficits (or drawn from `fold_prior.json`), 0.18 R camera
     containment; a low-order refit is selected when it gives every camera a single 8 R opening;
   - native-rate re-render (`seed_fit.render_seed`): consecutive decoded frames from the
     window's first frame, giving `seed_video` and `seed_depth`.

   The second route fits the keyframe wall points directly (`route: "direct"`,
   `seed_fit.seed_from_keyframes`): profiles fitted to the keyframe wall points (`R = 0.3`),
   0.36 R containment, enlarged (x1.15 ... x2, then low-order sweeps) until every camera
   projection shows one opening and at most 20 % of the dark lumen is blocked nearer than 3 R.
1. **Lumen fit** (`colonoworld/tube/fit_training.py`, `fit_training_tube`): starting from the
   window's seed sweep (a profile sweep fitted to the window's fused depth), a seeded search
   (random samples, differential evolution, local steps) deforms the sweep ahead of the cameras
   (throat position, radius, anisotropy, bend, shoulder, dilation) so the rendered tube agrees
   with the real window: lumen position and size per frame, and camera-following optical flow.
   Finalists are rendered at full resolution and must keep one lumen, no wall crossing, a
   0.3 R camera clearance and training luma bounds.
2. **Shape regularisation** (`colonoworld/tube/training_shape.py`, `regularize_training_tube`):
   cross-section deviation clipped to 15 %, short tubes extended, radius clipped and
   slope-limited, bends no sharper than the local radius, per-segment anatomical relief (taeniae,
   haustra, rectal valves); candidates are evaluated in a fixed order (full shape, shape outside the near-camera
   region, near-camera original, original centreline) and the first that meets the shape
   criteria is used.
3. **Render** with the device field picked from the most decisive of every 8th real frame.

Window inputs (one JSONL line each): real clip and window frame indices, calibration, segment,
seed sweep parameters, seed surface (poses, R, field mask), seed tube video and depth; or, to
build the seed first, the raw inputs `camera_npz`, `start` (30 Hz timeline), `frames`, `route`
and optionally a precomputed keyframe-depth `keyframes` npz. Keyframe depth needs the MegaSaM
code checkout (Depth Anything v1 and its local DINOv2 hub) and `depth_anything_vitl14.pth`:

```bash
export COLONOWORLD_DEPTH_CODE=/path/to/megasam COLONOWORLD_DEPTH_WEIGHTS=/path/to/depth_anything_vitl14.pth
python scripts/fit_training_tubes.py --config configs/tube.yaml --windows raw_windows.jsonl \
    --out outputs/training_tubes
```

## Rendering

`render_tube_video(tube, cameras, intrinsics, size=(832, 1104))` (`colonoworld/tube/render.py`)
casts one fisheye ray per pixel against the wall mesh (Mitsuba 3, `llvm_ad_rgb`, CPU), shades it
with a head light and the mucosa texture, and multiplies the frame by the endoscope's device
field (octagon x calibrated lens disk; the variant A/B/C is picked from the clip's first frame by
`pick_device_field`). Exposure is set per call: light radius = median ray depth over every 4th
frame / 1.38. `intrinsics` holds `calibration`, `texture`, and `device_fields` + `device_field`
(or a precomputed `field_alpha`). `write_tube_video` encodes H.264 (CRF 18, 24 fps), the format the
model was trained on. Frames are returned as uint8 `(T, H, W, 3)`.

## Device field thresholds

The device-field variant (A/B/C) must win by a score margin: `render.min_field_margin_eval` (0.11)
when it is picked from the first frame of an evaluation clip, and `render.min_field_margin_training`
(0.15) when it is picked from the most decisive of every 8th frame of a training window.

## Data files

Download the tube data archive (separate from the code; next to the model weights:
`<link to be added>`) and point `COLONOWORLD_DATA` at it:

- `tube/radius_prior.json`, `tube/fold_prior.json`, `full_colon/motion_library.npz`: derived from
  REAL-Colon (Biffi et al., 2024), CC BY 4.0.

Two files are built locally:

- `tube/mucosa_texture.png` is procedural (seeded noise, no images):
  `python scripts/build_tube_assets.py texture --out $COLONOWORLD_DATA/tube/mucosa_texture.png`
  (use OpenCV 5.0 to reproduce the reference texture bit for bit).
- `full_colon/donor_template.npz` is built from REAL-Colon training windows:
  `python scripts/build_tube_assets.py donor-template ...` (see `docs/full_colon.md`).

## Requirements

`numpy`, `scipy`, `opencv-python-headless`, `av`, `mitsuba` (3.x, LLVM variant) and `drjit`.
