# Condition inputs

Besides the camera trajectory and the rendered tube, the model takes a **palette mask video** and
per-frame flags (`colonoworld.conditions`).

## Palette

| class   | RGB           |
|---------|---------------|
| polyp   | (255, 0, 0)   |
| tool    | (0, 255, 0)   |
| water   | (0, 0, 255)   |
| blood   | (255, 255, 0) |
| snare   | (255, 0, 255) |

Each class mask is painted with a soft 3 px edge, in the order polyp, blood, water, tool, snare
(later classes cover earlier ones); pixels outside the endoscope field are black.
`water` is the irrigation class. Frame flags are `special_light, dye, tool, irrigation`; NBI is
the global `nbi` switch of `ColonoWorldModel.generate`.

```python
from colonoworld.conditions import render_palette_video
palette = render_palette_video({"polyp": polyp_masks, "tool": tool_masks}, field_mask=field)
```

A condition file (`cond.npz`, used by the builders in this section) holds boolean `(T, H, W)` masks `polyp, tool, water, snare`,
`flags (T, 4)`, `palette (T, H, W, 3)` and `field_mask (H, W)`.

## Camera intrinsics

A JSON file with `K` (3x3, output pixels) and `D` (4 Kannala-Brandt fisheye coefficients).
If the calibration was made at the sensor resolution it may also carry `K_native`,
`native_to_clip` (`K = native_to_clip @ K_native`) and `field_radius_px`.

## Snare

A short straight sheath leaves the working channel at the lower right of the scope tip and a
single continuous wire loop opens forward from its tip. The loop is open unless `--close` narrows
it over `--close-frames`. The tool mask holds the sheath only; the wire goes to the snare class.
Deployment ramps in over 15 frames from `--onset`; the snare is present until `--offset`.

```bash
python scripts/build_snare_insert.py --config configs/conditions.yaml \
    --intrinsics camera.json --base-cond cond.npz --out out/snare \
    --onset 30 --offset 170 --exit-mm 5 5 --sheath-len-mm 11 --loop-len-mm 20 \
    --loop-width-mm 14 --tilt-deg 25 --direction-len-mm 16.5
```

Other examples: `--sheath-len-mm 16 --loop-len-mm 28 --loop-width-mm 22 --tilt-deg -30
--aim-xy 0.62 0.80 --yaw-deg 25`, or a partly closing loop with `--sheath-len-mm 8
--loop-len-mm 22 --loop-width-mm 16 --tilt-deg -20 --aim-xy 0.86 0.64 --yaw-deg 35 --close 0.45
--close-frames 110 150`. Without `--base-cond`, pass `--field-mask field.npy --num-frames T`.
`--clear polyp water` empties those classes of the base condition.
Outputs: `cond.npz`, `palette.mp4`, `snare.json` (geometry per frame).

## Polyp

A polyp mesh (`sessile`, `pedunculated` or `flat`) is placed on the fitted tube wall and
ray-cast on every frame with the fisheye camera rays, so the wall hides it where appropriate.
The tube surface file (`surface.npz`, written next to each training tube by
`scripts/fit_training_tubes.py`) is an `.npz` with `V` (vertices), `F` (faces), `poses` (`T x 4 x 4`
camera-to-world), `R` (tube radius, equal to 25 mm) and `field_mask`. The diameter in pose
units is `size_mm / 25 * R`.

```bash
python scripts/build_polyp_insert.py --config configs/conditions.yaml \
    --surface surface.npz --intrinsics camera.json --pixel 480 529 --place-frame 94 \
    --size-mm 16 --onset 13 --out out/polyp
```

`--centre/--normal` place the polyp at an explicit wall point instead of a pixel ray.
`--size-mm auto` picks from `polyp.sizes_mm` the size whose median in-view area is closest to
`polyp.target_area`. Frames before `--onset` (and from `--offset` on) carry no polyp.
`--base-cond` adds the polyp to an existing condition (e.g. a snare).
Outputs: `polyp.npz` (`mask`, `visible`, placement), `cond.npz` and `palette.mp4`.

Polyp rendering needs Mitsuba 3 (`pip install mitsuba`, CPU `llvm_ad_rgb` variant).

## Using conditions for generation

- Single chunk (`scripts/generate.py`): pass `palette.mp4` or `cond.npz` as `inputs.mask`.
- Full procedure (`scripts/generate_full_procedure.py`): each chunk of the conditions manifest
  takes an `.npz` with `choice`, `mask_<class>` and optional `nbi` / `nbi_known`
  (see `docs/inference.md`); its `mask_<class>` arrays are the `polyp`, `tool`, `water` and
  `snare` masks of `cond.npz`.

