# Full-colon shapes

Full-procedure generation follows a cecum-to-rectum withdrawal through a synthetic whole colon.
`build_full_colon(shape_name, cfg)` (`colonoworld/tube/synthetic_colon.py`) returns
`(TubeModel, cameras (T, 4, 4), segments)`; `segments` lists runs of frames with their anatomical
segment (`name`, `start`, `end`, `segment_id`: cecum 0, ascending 1, transverse 2, descending 3,
sigmoid 4, rectum 5, flexures 6).

## Shapes

`configs/full_colon_shapes.yaml` defines seven shapes:

| shape | course |
|---|---|
| `standard` | high, near-horizontal transverse colon |
| `u_shaped` | transverse colon hanging into the pelvis as a U |
| `v_shaped` | transverse colon dipping into the pelvis as a V |
| `short_compact` | short colon with a mobile, medial cecum |
| `twisted` | long redundant transverse colon with several shallow loops |
| `long_sigmoid` | redundant sigmoid loop rising to the transverse level |
| `textbook` | textbook anatomy with labelled segments |

Each shape gives anatomical control points per segment, target length, segment diameters,
per-segment withdrawal speed, deformations and a seeded free-form wall
(`colonoworld/tube/colon_anatomy.py`). The camera path starts with a 189-frame dwell in the cecum
and then withdraws to the rectum. Viewing directions and wall offsets are replayed from training
windows of the same segment (motion library), cross-faded over 16 frames, and pulled back toward
the centreline wherever the wall would come closer than the clearance target. `twisted` and
`long_sigmoid` use a stronger off-axis view gain (1.1 and 1.3) and a closer clearance target.

## Data

`${COLONOWORLD_DATA}/full_colon/donor_template.npz` (reference colon: cecal-dwell rotations,
withdrawal step pattern, fold relief, camera field) and
`${COLONOWORLD_DATA}/full_colon/motion_library.npz` (per training window: segment, length,
`look (T, 3)` tube tangent in camera coordinates, `offset (T, 3)` lateral camera offset in camera
coordinates / local radius). The library is produced from fitted training tubes with
`build_motion_library(windows)` + `save_motion_library` and is part of the data download (derived
from REAL-Colon (Biffi et al., 2024), CC BY 4.0).

The donor template is not distributed; build it from REAL-Colon training windows:

```bash
python scripts/build_tube_assets.py donor-template --config configs/full_colon_shapes.yaml \
    --reference-frame /path/to/real_colon/001-001_frame_31667.jpg
```

Inputs (section `donor_template_build` of the config lists the REAL-Colon window ids), under
`${COLONOWORLD_DATA}/full_colon/donor_inputs/`, part of the data download (`<link to be added>`;
derived from REAL-Colon (Biffi et al., 2024), CC BY 4.0): `window_fits/<window>.json` (fitted
sweep parameters of the 52 pool pieces, 49 distinct windows), `window_poses/<window>.npz` (poses
and R of the 128 camera windows) and `view_controls.npy` (per-frame base-view tilts). The
reference frame is a real image and is not distributed: pass REAL-Colon video `001-001`, frame
`31667` with `--reference-frame` (the native 1352x1080 frame is mapped to the clip canvas with
`reference_frame_native_to_clip`; a frame already on the 1104x832 canvas is used as is). Only
its lens content mask is used.

## Usage

```bash
export COLONOWORLD_DATA=/path/to/data
python scripts/build_full_colon.py --config configs/full_colon_shapes.yaml \
    --shape standard --out outputs/full_colon/standard --render
```

Writes `tube.npz`, `cameras.npy`, `segments.json` and, with `--render`, one tube video per
generation chunk (`chunk_XX.mp4`, 189 frames, stride 188) with `chunk_XX_cameras.npy` (poses
relative to the chunk's first frame). Every chunk is rendered with
`colonoworld.tube.render.render_tube_video` (device field A, the fixed synthetic calibration) and
gets its own exposure, exactly like the training tubes.
