# Training

ColonoWorld is a Cosmos3-Nano video model fine-tuned in three stages. Each stage starts from the
weights of the previous one; only LoRA adapters of the generation expert and the ColonoWorld
condition modules are trained (the Cosmos backbone and the VAE stay frozen).

1. **Camera** (1,200 updates): Pluecker-ray camera adapter and LoRA.
2. **Tube** (15,500 updates): rendered-tube latent stream (VAE-encoded tube video appended as
   token rows next to the noisy video tokens) on clean windows.
3. **Joint conditioning** (2,500 updates; `configs/train_joint.yaml`): tube + condition-mask
   stream (polyp, tool, snare, irrigation palette video) + per-latent-frame global flags (special
   light, tool, irrigation, class-known bits) + colon-segment embedding, on a stratified mix of
   windows. The last 750 updates use the recipe of `configs/train_joint.yaml`.

All stages use 8 GPUs with an effective batch of 8. Update counts of stages 2 and 3 continue the
same counter (the released model is update 18,000 of stage 2+3, i.e. 19,200 updates in total);
the released weights are the EMA at that point.

## Data preparation

```bash
export COLONOWORLD_DATA=/path/to/data COLONOWORLD_CKPT=/path/to/colonoworld.pt
python scripts/prepare_data.py --config configs/train_joint.yaml \
    --index $COLONOWORLD_DATA/train/index.jsonl --out $COLONOWORLD_DATA/train
```

The index lists source windows (65 or 189 frames at 24 fps, 832x1104) with their camera
estimates, KB4 calibration, per-frame colon-segment labels and condition annotations; the format
is documented at the top of `scripts/prepare_data.py`. For each window the script builds and
renders the tube (`colonoworld.tube`), rasterises the condition masks (snare wire polylines,
irrigation jet ribbons from the nozzle to the jet target), writes `manifest.jsonl` and records
the admission result of every other window in `rejected.json`. `--cache-latents` additionally stores
the VAE latent of each untransformed tube.

### Window admission

| Gate | Rule |
| --- | --- |
| Camera estimate | finite, rigid poses for every frame |
| Tube wall fit | blocked fraction <= 0.20, no missing lumen depth, one lumen opening, camera inside the tube and no wall crossing in every frame |
| Tube agreement | flow support >= 0.4 in every 4-frame interval, and either block direction cosine >= 0.35, radial-motion correlation >= 0.4 and magnitude ratio in [0.4, 2.5], or both videos static (>= 80% static intervals, p90 flow <= 0.35 px at 160x120) |

Agreement metrics are computed by `colonoworld.training.data.flow_agreement` (Farneback flow on
frames 0, 4, 8, ... at 160x120 inside the endoscope field, highlights and the darkest 2% removed).

### Strata

A window may be drawn in the strata listed in its `strata` field. The **clean** stratum only takes
windows with no polyp, no tool, no irrigation, white light, no dye and no bubbles; windows
carrying any of these labels are kept for their own stratum. The released model used 734 clean
candidate windows. Clean windows in the top 30% of the lateral/rotational tube-motion score are
drawn twice, forward-moving windows (majority of 4-frame intervals classified as forward motion
from the camera estimate and tube depth) four times.

| Stratum | Weight | Window pick |
| --- | --- | --- |
| clean | 0.30 | weighted clean list |
| polyp | 0.25 | entry draw 50% (polyp absent in the first played frame, forced direction), otherwise any polyp window |
| irrigation | 0.15 | per irrigation event; entry draw 50%; jet onsets between frames 6 and 48 weighted x3 |
| tool | 0.15 | entry draw 30% |
| NBI | 0.15 | windows switching light between first and last frame x8, other switching windows x2 |

Entry draws sample up to 8 augmentation plans and keep the first that preserves the entry; the
eighth plan keeps the window as an entry draw. The per-frame occupancy used for this entry check and
for the tool/irrigation presence flags is computed from the spatially augmented masks in
tube-frame order (static-camera draws repeat frame 0); the video, tube and mask stream follow
the drawn playback direction.

### Condition dropout

Per draw and class, the mask stream carries the mask (polyp .8, tool .9, water .9; always for
entry draws), an explicit empty mask for labelled-absent classes (.3), or nothing. Each class is
independently unknown with probability .1, all conditions with .05, all frame flags with .1.
Per update the tube is dropped (.10) or tube and camera are dropped (.05). The colon-segment
embedding is null in 10% of draws and the caption omits the segment in 10%.

### Augmentation

Mirror (h/v), roll in [-15, 15] deg with optional drift up to 3 deg, static-camera draws (10%),
crop to the common inscribed optical disk; real video, tube, masks and camera poses are
transformed together. The conditioning frame is corrupted in 50% of draws (pixel degradations and
VAE round trips 40%, latent noise only 10%) so that chained chunks start robustly from generated
frames. Positive irrigation draws weight the flow-matching loss by 3 inside the jet region
dilated by 25 px.

## Training

```bash
torchrun --nproc_per_node=8 scripts/train.py --config configs/train_joint.yaml
```

`run.init_checkpoint` is the starting checkpoint: by default the released ColonoWorld checkpoint
(fine-tuning from the released model); point it to your own stage-2 or stage-3 checkpoint to
reproduce the stage order above. Seeds are the named keys `sampler.seed`, `sampler.dropout_seed`
and `run.noise_seed`. Learning rates: 1e-4 condition tokens,
1e-3 mask-stream input projection and global-flag adapter, 2e-5 LoRA; AdamW, 100-update warmup,
gradient-norm clip 1, EMA 0.99, checkpoints every 250 updates (`checkpoint_<update>.pt` with raw and
EMA weights and the optimizer state). Draws are a pure function of `(sampler.seed, update, draw)`,
so a resumed or re-run job sees the same data.

Required packages beyond the base install: `torch`, `av`, `Pillow`, the Cosmos framework
(`cosmos_framework`), and the tube renderer dependencies listed in `docs/tube.md`.
