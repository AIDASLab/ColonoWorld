# Evaluation

Code: `colonoworld/eval/` (`camera_metrics.py`, `fvd_fid_is.py`, `lumen_following.py`),
CLI: `scripts/evaluate.py`, settings: `configs/eval.yaml`.

## Camera control (RotErr, TransErr, CamMC)

Camera control is measured by reading the camera path back from the generated video and comparing
it with the commanded path.

**Read-back.** The read-back uses an external monocular SfM / depth estimator,
[MegaSaM](https://github.com/mega-sam/mega-sam), which is installed separately and is not part of
this package. Each video (first 189 frames, 1104x832) is rectified from the Kannala-Brandt fisheye
calibration to a 640x480 pinhole image with a fixed intrinsic matrix, then run through MegaSaM at
512x384 with full bundle adjustment. The output of one video is stored as an `.npz` with

- `relative_c2w`: `(N, 4, 4)` camera-to-world poses, OpenCV axes (x right, y down, z forward),
  frame 0 = identity, translation in units of the frame-0 median scene depth;
- `valid` (optional): `(N,)` boolean, frames set to NaN when false.

`camera_metrics.load_readback(path)` loads this file. Any estimator that writes the same format can
be used. A video without an output (or with fewer frames) is scored as untracked.

The reference path is the camera path of the real clip, in the same format.

**Main protocol** (`score_clip_windowed`, `aggregate`): windows of 16 frames sampled at stride 8
(`w, w+8, ..., w+120`, `w = 0, 8, ..., 64`, 9 windows per 189-frame clip). Each window is rebased to
its first camera for both paths, translations of each path are divided by their maximum distance
from the first camera, and

- RotErr = sum over the 16 frames of the geodesic angle [rad],
- TransErr = sum of L2 translation distances,
- CamMC = sum of Frobenius norms of the 3x4 `[R|t]` differences,
- DirErr = angle [deg] between the displacements of the last frame (windows whose reference moves
  less than 0.05 are skipped; a read-back moving less than 0.005 counts as 90 deg).

Per clip: mean over windows; table: mean over clips. Untracked clips are scored with the
static-camera path (`Q = I`) and counted in `n_untracked`.

**Summed protocol** (`score_clip_summed`): sums over frames 1..188 of the rotation angle [deg],
`|s t_Q - t_P|` and the 3x4 Frobenius difference, with a global scale `s` fitted by least squares on
development clips (`fit_scale`). Values are clipped at fixed caps, which are also the score of untracked clips.

**Full procedure** (`score_procedure`): a chained video of `K` chunks (189 frames each, the
hand-off frame shared) is read back chunk by chunk; each chunk is scored with the main protocol
against its command and the procedure value is the mean over chunks.

**Long videos** (`stitch_chunks`, `score_long`): overlapping read-back chunks are stitched (chained
rotations, per-chunk translation units), and per-window Sim(3) ATE, normalized ATE, 1 s RPE and
rotation drift [deg/min] are reported.

```
python scripts/evaluate.py camera --config configs/eval.yaml --manifest camera.csv --out results/camera
# camera.csv: clip,reference_pose,readback
python scripts/evaluate.py procedure --commands commands.npy --readbacks readback_dir --out results/procedure
# readback_dir/chunk_000.npz, chunk_001.npz, ...
```

## FVD, FID, IS

Feature networks (download once, place in `${COLONOWORLD_DATA}/eval_weights/`):

| File | Source | sha256 |
| --- | --- | --- |
| `i3d_torchscript.pt` | StyleGAN-V I3D: `https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1` | `bec6519f66ea534e953026b4ae2c65553c17bf105611c746d904657e5860a5e2` |
| `inception-2015-12-05.pt` | StyleGAN2-ADA: `https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt` | `f58cb9b6ec323ed63459aa4fb441fe750cfe39fafad6da5cb504a16f19e958f4` |

Per video, the first 189 frames are decoded in stream order (PyAV) and multiplied by the clip's
field-of-view mask (a binary PNG, nearest-resized to the video size).

- FVD: I3D features of 16-frame windows with stride 1 (174 per clip), bilinear resize to 224x224,
  values in [-1, 1]; Frechet distance between the generated and the real window features.
- FID: Inception pool features of every frame at native size; Frechet distance.
- IS: Inception 1008-way probabilities of every generated frame, 10 consecutive splits.

`matched_metrics` gives the same-size variant (fixed half-splits of the reference set).

```
python scripts/evaluate.py quality --manifest quality.csv --out results/quality
# quality.csv: clip,generated_video,reference_video,field_mask
```

## Lumen following

Checks that the dark lumen of the generated video lies where the conditioning tube places it. Per
frame on a 160x120 grid, the generated lumen is the darkest connected region below 0.45 x the
median field luma; the render lumen is the largest region of tube depth beyond 2 x the median depth
(field pixels without depth, i.e. looking out of the open far end of the tube, count as deepest).
Reported per clip and per set: centroid distance / image diagonal, area ratio (geometric mean), IoU,
absent fractions and the lumen recall versus the render.

```
python scripts/evaluate.py lumen --manifest lumen.csv --out results/lumen
# lumen.csv: clip,generated_video,tube_aux   (tube_aux .npz: depth (T,h,w), field_mask (h,w))
```

## Dependencies

`numpy`, `scipy`, `pyyaml`; `torch` and `av` (PyAV) for FVD / FID / IS; `opencv-python-headless`
for lumen following; `pillow` for field masks.
