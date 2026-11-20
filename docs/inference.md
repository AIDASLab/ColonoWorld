# Inference

Two entry points, both on top of `ColonoWorldModel.generate` (see `docs/model.md`):

- `scripts/generate.py` (`colonoworld.inference.generate`): one 189-frame chunk;
- `scripts/generate_full_procedure.py` (`colonoworld.inference.chain`): a full cecum-to-rectum
  withdrawal as a chain of chunks.

Defaults for both: EMA weights, generic caption, 35 UniPC steps, text guidance 1, tube CFG 1,
mask CFG 3.

## Single chunk

```bash
python scripts/generate.py --config configs/inference.yaml \
    inputs.first_frame=frame0.png inputs.cameras=cameras.npy inputs.intrinsics=calib.json \
    inputs.tube=tube.mp4 inputs.mask=masks.npz inputs.nbi=false output.dir=outputs/chunk
```

`inputs.tube` is a rendered tube video or a `TubeModel` `.npz` (rendered along the cameras);
`inputs.mask` is a palette video or boolean class masks. Writes `generated.mp4` (plus
`tube.mp4` / `mask.mp4` previews).

## Full procedure

```bash
export COLONOWORLD_DATA=/path/to/data COLONOWORLD_CKPT=/path/to/ckpt COSMOS_WEIGHTS=/path/to/cosmos
python scripts/generate_full_procedure.py --config configs/full_procedure.yaml \
    --shape standard --frame0 ${COLONOWORLD_DATA}/full_procedure/cecum_frame0.png --out outputs/full_procedure/standard
```

### Start frame

The examples start from a cecum frame of the REAL-Colon dataset (video `003-003`, frame `25114`).
Frames are not distributed with the code. REAL-Colon provides every video as numbered frame
images; map frame 25114 of video `003-003` onto the clip canvas:

```bash
python scripts/extract_realcolon_frame.py --config configs/full_procedure.yaml \
    --frame /path/to/REAL-Colon/003-003_frames/<frame 25114>.jpg \
    --out ${COLONOWORLD_DATA}/full_procedure/cecum_frame0.png
```

Any cecum frame on the 1104x832 canvas can be used as `--frame0`.

The synthetic colon of `--shape` (see `docs/full_colon.md`) is split into 189-frame chunks that
share one frame (stride 188). For chunk `k`:

1. the start frame is `--frame0` for the first generated chunk, otherwise the last generated frame
   of chunk `k - 1`;
2. the camera poses are the chunk's poses relative to its first frame; rays outside the endoscope
   field of the synthetic videos are zeroed;
3. the tube is rendered along the chunk's poses and stored as H.264 (`chunk_XX.mp4` in
   `chain.tube_dir`, the format of the training tubes), then decoded for generation;
4. segment ids are the commanded anatomy at output frames 0, 4, 8, ... (flexures: null id);
5. the seed is `seed + seed_stride * k` (default stride 1000003; the textbook-shape examples use
   `chain.seed_stride=1`);
6. optional conditions come from the conditions manifest (below).

Frame 0 of each generated chunk is replaced by its exact start frame. Outputs in `output.dir`:

| file | content |
|---|---|
| `chunks/chunk_XX.mp4` | generated chunk (189 frames) |
| `chunks/chunk_XX_last.npy` | last frame of the chunk, `uint8 (832, 1104, 3)` |
| `full_procedure.mp4` | stitched video (first chunk complete, later chunks without their shared first frame) |
| `summary.json` | chunks, seeds, settings |
| `tube/` | synthetic colon and tube chunks (when `chain.tube_dir` is not set) |

A prepared tube directory from `scripts/build_full_colon.py --render` can be passed as
`chain.tube_dir`; missing chunk videos are rendered on demand.

Options (`configs/full_procedure.yaml`, CLI flags in brackets): `chain.shape` (`--shape`),
`chain.frame0` (`--frame0`), `chain.start_segment` (`--start-segment`), `chain.segments`
(`--segments`), `chain.conditions` (`--conditions`), `chain.segment_ids` (`commanded` or `null`),
`chain.seed`, `chain.seed_stride`, `sampling.*`, `output.dir` (`--out`). Any key can also be set
with `key.sub=value` arguments.

### Regenerating one chunk

Every chunk depends only on its start frame, so a single chunk can be regenerated from the saved
last frame of the previous one:

```bash
python scripts/generate_full_procedure.py --config configs/full_procedure.yaml --shape textbook \
    --start-segment 8 --segments 1 --frame0 outputs/full_procedure/textbook/chunks/chunk_07_last.npy \
    --conditions conditions.json --out outputs/full_procedure/textbook_chunk08
```

### Conditions manifest

```json
{"chunks": {"8": "chunk08.npz", "14": "chunk14.npz"}}
```

Paths are relative to the manifest. Chunks that are not listed are generated from camera and
tube only. Each `.npz` covers the 189 output frames of its chunk (frame 0 = the start frame):

| key | content |
|---|---|
| `choice` | JSON string, e.g. `{"polyp": "mask", "tool": "empty", "water": null, "snare": null}`; values `"mask"`, `"empty"` or `null` |
| `mask_<class>` | for each `"mask"` class: boolean `(189, 832, 1104)`, or `np.packbits(mask, axis=-1)` |
| `nbi`, `nbi_known` | optional per-frame special light value and known bit (0/1) |

`"mask"` supplies the class mask, `"empty"` states that the class is absent, `null` leaves it
unspecified; a snare needs the tool channel. The masks are painted into the palette video
(`colonoworld.conditions.render_palette_video`) and the global flags are derived with
`colonoworld.models.global_flags`. A chunk with any non-null class uses mask CFG
(`sampling.mask_cfg`); special light alone rides on the global flags.

### Python

```python
from colonoworld.utils.config import load_config
from colonoworld.inference import run_full_procedure

cfg = load_config("configs/full_procedure.yaml", ["chain.shape=u_shaped", "chain.segments=3"])
out_dir = run_full_procedure(cfg)          # optional: run_full_procedure(cfg, model=loaded_model)
```

One chunk takes one GPU with about 40 GB of memory; a standard procedure has 28 chunks
(5265 frames, about 3.7 minutes of video at 24 fps).
