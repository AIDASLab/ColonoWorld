# SPDX-License-Identifier: CC-BY-NC-4.0
"""Training augmentation: paired pixel/pose transforms and conditioning-frame corruption.

Spatial augmentation (applied identically to the real video, the rendered tube and the condition
mask video, with the camera poses transformed consistently):

* horizontal / vertical mirror (p=.5 each), a roll in [-15, 15] deg and, in half of the draws, a
  linear roll drift up to +-3 deg over the window; the intrinsics are mirrored and the camera
  rotation conjugated so that ``det(R)=+1`` is preserved;
* a static-camera draw (10%): every frame repeats frame 0 (real frame plus N(0, 0.5) DN noise);
* the field of view is cropped to the common inscribed optical disk of all transformed frames;
  latent cells that are partially masked at any time get zero camera rays.

Conditioning-frame corruption (hand-off robustness for chunk chaining): 50% of draws replace the
VAE latent of frame 0 by the latent of a degraded frame (colour/exposure/Lab shifts, specular
highlight removal, sensor grain, colour blotches, blur, VAE round trips) plus latent noise, so the
model learns to continue from imperfect generated frames.
"""

from __future__ import annotations

import copy
import hashlib
import math
import random
from collections.abc import Callable
from typing import Any

import numpy as np

SIGMA_DN = 0.5


# --------------------------------------------------------------------------------------------
# Draw parameters
# --------------------------------------------------------------------------------------------


def static_fraction(schedule: str, update_after_start: int) -> float:
    """Static-camera probability from a schedule like ``"0.25@500,0.10"``."""
    if update_after_start < 1:
        raise ValueError("static schedule starts at the first update")
    parts = schedule.split(",")
    for i, part in enumerate(parts):
        fields = part.split("@")
        fraction = float(fields[0])
        if i == len(parts) - 1 or update_after_start <= int(fields[1]):
            return fraction
        update_after_start -= int(fields[1])
    raise ValueError(schedule)


def plan(
    seed: int,
    update: int,
    draw: int,
    enabled: bool = True,
    static_schedule: str = "0.10",
    start_update: int = 0,
) -> dict[str, Any]:
    """Augmentation parameters of one draw, seeded by ``sha256(seed:update:draw)``."""
    fraction = static_fraction(static_schedule, update - start_update)
    value = int.from_bytes(
        hashlib.sha256(f"{seed}:{update}:{draw}".encode()).digest()[:4], "little"
    )
    r = np.random.RandomState(value)
    p = dict(enabled=enabled, seed=value, static=bool(enabled and r.rand() < fraction))
    p["hflip"] = bool(enabled and r.rand() < 0.5)
    p["vflip"] = bool(enabled and r.rand() < 0.5)
    p["roll_deg"] = float(r.uniform(-15, 15)) if enabled else 0.0
    p["drift_deg"] = float(r.uniform(-3, 3)) if enabled and r.rand() < 0.5 else 0.0
    p["reverse"] = False
    return p


def force_direction(p: dict[str, Any], forced: str | None) -> dict[str, Any]:
    """Entry draws fix the playback direction and disable the static-camera draw."""
    if forced is not None:
        p["static"] = False
        p["reverse"] = forced == "reverse"
    return p


def selected_indices(frames: int, p: dict[str, Any]) -> np.ndarray:
    """Tube-frame index of every output frame (before static repetition)."""
    idx = np.arange(frames)
    return idx[::-1].copy() if p["reverse"] else idx


# --------------------------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------------------------


def plucker_rays(
    c2w: np.ndarray, cam: dict[str, Any], size: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pluecker rays [t,6,H/16,W/16], valid cells [H/16,W/16] and latent frame indices."""
    from colonoworld.models.camera import latent_time_indices
    from colonoworld.models.camera import plucker_rays as rays

    r, valid = rays(c2w, cam, size)
    return r, valid, latent_time_indices(len(c2w))


def aperture(cam: dict[str, Any], h: int, w: int) -> np.ndarray:
    from colonoworld.models.camera import unproject_kb4

    yy, xx = np.mgrid[:h, :w]
    _, valid = unproject_kb4(np.stack([xx, yy], -1).reshape(-1, 2), cam)
    return valid.reshape(h, w)


def camera_transforms(
    cam: dict[str, Any], n: int, p: dict[str, Any]
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Mirrored intrinsics, per-frame rotations ``Q_t = Rz(roll_t) diag(h, v, 1)`` and the
    image affines ``A_t = K' Q_t K^-1``."""
    cam = copy.deepcopy(cam)
    w, h = cam["clip_size"]
    k = np.asarray(cam["K_clip"], float)
    new = k.copy()
    if p["hflip"]:
        new[0, 2] = w - 1 - k[0, 2]
    if p["vflip"]:
        new[1, 2] = h - 1 - k[1, 2]
    cam["K_clip"] = new.tolist()
    cam["native_to_clip"] = (new @ np.linalg.inv(np.asarray(cam["K_native"]))).tolist()
    f = np.diag([-1.0 if p["hflip"] else 1.0, -1.0 if p["vflip"] else 1.0, 1.0])
    qs, affines = [], []
    for a in np.deg2rad(p["roll_deg"] + np.linspace(0, p["drift_deg"], n)):
        r = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])
        q = r @ f
        qs.append(q)
        affines.append(new @ q @ np.linalg.inv(k))
    return cam, np.stack(qs), np.stack(affines)


def transform_poses(poses: np.ndarray, qs: np.ndarray) -> np.ndarray:
    """``C2W'_t = Q_0 inv(P_0) P_t Q_t^-1`` (paired world/camera reflection keeps det=+1)."""
    rel = np.linalg.inv(poses[0])[None] @ poses
    out = rel.copy()
    out[:, :3, :3] = np.einsum("ij,tjk,tlk->til", qs[0], rel[:, :3, :3], qs)
    out[:, :3, 3] = rel[:, :3, 3] @ qs[0].T
    return out


def warp(frame: np.ndarray, affine: np.ndarray, nearest: bool = False) -> np.ndarray:
    import cv2

    return cv2.warpAffine(
        frame,
        affine[:2],
        (frame.shape[1], frame.shape[0]),
        flags=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def apply(
    real: np.ndarray,
    tube: np.ndarray,
    poses: np.ndarray,
    cam: dict[str, Any],
    p: dict[str, Any],
    field: np.ndarray | None = None,
    extra: np.ndarray | None = None,
) -> dict[str, Any]:
    """Apply one draw to real/tube videos (THWC uint8), poses [T,4,4] and an optional
    condition-mask video ``extra`` (same temporal and spatial ops as the tube).

    Returns transformed videos, poses, calibration, Pluecker rays/valid cells, per-frame
    apertures and the affines used.
    """
    n, h, w, _ = real.shape
    if p["reverse"]:
        poses = poses[::-1].copy()
        poses = np.linalg.inv(poses[0])[None] @ poses
        real, tube = real[::-1].copy(), tube[::-1].copy()
        if extra is not None:
            extra = extra[::-1].copy()
    if p["static"]:
        real = np.repeat(real[:1], n, axis=0)
        tube = np.repeat(tube[:1], n, axis=0)
        poses = np.repeat(poses[:1], n, axis=0)
        if extra is not None:
            extra = np.repeat(extra[:1], n, axis=0)
        noise = np.random.RandomState(p["seed"]).normal(0, SIGMA_DN, real.shape)
        real = np.clip(np.rint(real.astype(np.float32) + noise), 0, 255).astype(np.uint8)
        p["drift_deg"] = 0.0
    newcam, qs, affines = camera_transforms(cam, n, p)
    newposes = transform_poses(poses, qs)
    if not p["enabled"]:
        rays, valid, latent = plucker_rays(poses, cam, (h, w))
        return dict(
            real=real,
            tube=tube,
            poses=poses,
            calibration=cam,
            rays=rays,
            valid=valid,
            aperture=np.repeat(aperture(cam, h, w)[None], n, 0),
            affines=affines,
            latent=latent,
            extra=extra,
        )
    if field is None:
        field = aperture(cam, h, w)
    # Common inscribed optical disk: inside every transformed aperture and source rectangle.
    kinv = np.linalg.inv(np.asarray(newcam["K_clip"]))
    yy, xx = np.mgrid[:h, :w]
    xy = np.stack([xx, yy, np.ones_like(xx)], -1) @ kinv.T
    border = np.concatenate([xy[0], xy[-1], xy[:, 0], xy[:, -1]])
    radius = float(np.linalg.norm(border[:, :2], axis=1).min()) * 0.985
    disk = np.linalg.norm(xy[:, :, :2], axis=-1) <= radius
    masks, rr, tt, ee = [], [], [], []
    for i, (r, t, a) in enumerate(zip(real, tube, affines, strict=True)):
        m = (warp(field.astype(np.uint8), a, nearest=True) > 0) & disk
        r, t = warp(r, a), warp(t, a)
        r[~m] = 0
        t[~m] = 0
        masks.append(m)
        rr.append(r)
        tt.append(t)
        if extra is not None:
            e = warp(extra[i], a)
            e[~m] = 0
            ee.append(e)
    masks = np.stack(masks)
    rays, valid, latent = plucker_rays(newposes, newcam, (h, w))
    # Exclude every latent cell that is partially masked at any time.
    valid &= masks.all(0).reshape(h // 16, 16, w // 16, 16).all((1, 3))
    rays *= valid[None, None]
    return dict(
        real=np.stack(rr),
        tube=np.stack(tt),
        poses=newposes,
        calibration=newcam,
        rays=rays,
        valid=valid,
        aperture=masks,
        affines=affines,
        latent=latent,
        extra=np.stack(ee) if extra is not None else None,
    )


def spatial_bool(region: np.ndarray, result: dict[str, Any], p: dict[str, Any]) -> np.ndarray:
    """Map an output-ordered bool [T,H,W] region with the draw's spatial transform."""
    out = np.empty_like(region)
    for i, (frame, affine) in enumerate(zip(region, result["affines"], strict=True)):
        mapped = warp(frame.astype(np.uint8), affine, True) > 0 if p["enabled"] else frame.copy()
        out[i] = mapped & result["aperture"][i]
    return np.ascontiguousarray(out)


def dilated_loss_weights(
    region: np.ndarray, weight: float = 3.0, dilate_px: int = 25
) -> np.ndarray:
    """Pixel loss weights: ``weight`` inside the region dilated by an elliptic kernel."""
    import cv2

    if dilate_px:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1,) * 2)
        region = np.stack([cv2.dilate(f.astype(np.uint8), kernel).astype(bool) for f in region])
    return np.float32(1.0) + np.float32(weight - 1.0) * region.astype(np.float32)


# --------------------------------------------------------------------------------------------
# Conditioning-frame corruption
# --------------------------------------------------------------------------------------------

HANDOFF = dict(
    mode_budget=dict(clean=0.50, pixel=0.40, latent=0.10),
    exposure=(0.85, 1.18),
    colour=(0.93, 1.07),
    gamma=(0.9, 1.1),
    lab_l_shift=(-9.0, 3.0),
    lab_a_shift=(-6.0, 6.0),
    stress_lab_a_shift=(-12.0, 12.0),
    lab_b_scale=(0.7, 1.05),
    p_highlight_shrink=0.7,
    highlight_threshold=(0.8, 0.92),
    highlight_chroma=0.12,
    highlight_remove_share=(0.2, 0.8),
    highlight_erosion_radius=(0, 1),
    highlight_inpaint_radius=3,
    kept_min_erode_area=64,
    p_grain=0.8,
    p_stress=0.10,
    grain_target=(1.2, 3.0),
    stress_grain_target=(3.0, 5.0),
    grain_channel_correlation=0.7,
    grain_tolerance=0.05,
    grain_bisection_iterations=12,
    p_blotches=0.15,
    stress_p_blotches=0.5,
    patch_count=(1, 3),
    patch_center_radius=(0.35, 0.75),
    patch_sigma=(0.07, 0.18),
    patch_a_shift=(-25.0, -10.0),
    patch_l_shift=(-8.0, 0.0),
    annulus=(0.18, 0.95),
    annulus_feather=0.08,
    p_blur=0.3,
    blur_sigma=(0.2, 0.8),
    p_roundtrip=0.65,
    roundtrips=(1, 3),
    p_stacked_noise=0.3,
    noise_std_fraction=(0.05, 0.3),
)


def handoff_params(seed: int, draw: int, update: int, enabled: bool = True) -> dict[str, Any]:
    """Corruption parameters of one draw: mode clean/pixel/latent = 50/40/10."""
    R = HANDOFF
    draw_seed = int.from_bytes(
        hashlib.sha256(f"{seed}:{draw}:{update}:0".encode()).digest()[:8], "big"
    )
    rng = random.Random(draw_seed % 2**63)
    mode_draw = rng.random()
    mode = "clean"
    if enabled:
        cumulative = 0.0
        for candidate, probability in R["mode_budget"].items():
            cumulative += probability
            if mode_draw < cumulative:
                mode = candidate
                break
    p: dict[str, Any] = dict(mode=mode, roundtrips=0, noise_fraction=0.0, patches=[])
    if mode == "clean":
        return p
    add_noise = rng.random() < R["p_stacked_noise"] if mode == "pixel" else True
    if add_noise:
        p["noise_fraction"] = rng.uniform(*R["noise_std_fraction"])
        p["noise_seed"] = rng.randrange(2**63)
    if mode == "latent":
        return p

    def uniform(name: str) -> float:
        return rng.uniform(*R[name])

    stress = rng.random() < R["p_stress"]
    exposure = uniform("exposure")
    colour = [uniform("colour") for _ in range(3)]
    p.update(
        gains=[v * exposure / (sum(colour) / 3.0) for v in colour],
        gamma=uniform("gamma"),
        lab_l_shift=uniform("lab_l_shift"),
        lab_a_shift=uniform("stress_lab_a_shift" if stress else "lab_a_shift"),
        lab_b_scale=uniform("lab_b_scale"),
    )
    p["highlight_shrink"] = rng.random() < R["p_highlight_shrink"]
    if p["highlight_shrink"]:
        p.update(
            highlight_threshold=uniform("highlight_threshold"),
            highlight_remove_share=uniform("highlight_remove_share"),
            highlight_erosion_radius=rng.randint(*R["highlight_erosion_radius"]),
            highlight_blob_seed=rng.randrange(2**63),
        )
    p["grain"] = rng.random() < R["p_grain"]
    if p["grain"]:
        lo, hi = R["stress_grain_target" if stress else "grain_target"]
        p["grain_target"] = math.exp(rng.uniform(math.log(lo), math.log(hi)))
        p["grain_seed"] = rng.randrange(2**63)
    blotches = rng.random() < R["stress_p_blotches" if stress else "p_blotches"]
    for _ in range(rng.randint(*R["patch_count"]) if blotches else 0):
        angle = rng.uniform(0.0, 2.0 * math.pi)
        radius = uniform("patch_center_radius")
        p["patches"].append(
            dict(
                x=radius * math.cos(angle),
                y=radius * math.sin(angle),
                sigma=uniform("patch_sigma"),
                a_shift=uniform("patch_a_shift"),
                l_shift=uniform("patch_l_shift"),
            )
        )
    p["blur_sigma"] = uniform("blur_sigma") if rng.random() < R["p_blur"] else 0.0
    p["roundtrips"] = rng.randint(*R["roundtrips"]) if rng.random() < R["p_roundtrip"] else 0
    return p


def _quantize(x):
    return (x.clamp(0.0, 1.0) * 255.0).round().to(dtype=__import__("torch").uint8)


def _blur(x, sigma: float):
    import torch
    import torch.nn.functional as F

    if sigma == 0.0:
        return x
    radius = max(1, math.ceil(3.0 * sigma))
    q = torch.arange(-radius, radius + 1, device=x.device, dtype=torch.float32)
    k = torch.exp(-q.square() / (2.0 * sigma * sigma))
    k = k / k.sum()
    y = x.permute(1, 0, 2, 3)
    y = F.conv2d(
        F.pad(y, (radius, radius, 0, 0), mode="replicate"),
        k.view(1, 1, 1, -1).expand(3, 1, 1, -1),
        groups=3,
    )
    y = F.conv2d(
        F.pad(y, (0, 0, radius, radius), mode="replicate"),
        k.view(1, 1, -1, 1).expand(3, 1, -1, 1),
        groups=3,
    )
    return y.permute(1, 0, 2, 3)


_RGB2XYZ = [
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
]
_XYZ2RGB = [
    [3.2404542, -1.5371385, -0.4985314],
    [-0.9692660, 1.8760108, 0.0415560],
    [0.0556434, -0.2040259, 1.0572252],
]
_WHITE = [0.95047, 1.0, 1.08883]


def _rgb_to_lab(rgb):
    import torch

    rgb = rgb.clamp(0.0, 1.0)
    linear = torch.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055).pow(2.4))
    xyz = torch.einsum("ij,jthw->ithw", rgb.new_tensor(_RGB2XYZ), linear)
    xyz = xyz / rgb.new_tensor(_WHITE).view(3, 1, 1, 1)
    delta = 6.0 / 29.0
    f = torch.where(
        xyz > delta**3, xyz.clamp_min(0.0).pow(1.0 / 3.0), xyz / (3.0 * delta**2) + 4.0 / 29.0
    )
    return torch.stack((116.0 * f[1] - 16.0, 500.0 * (f[0] - f[1]), 200.0 * (f[1] - f[2])))


def _lab_to_rgb(lab):
    import torch

    fy = (lab[0] + 16.0) / 116.0
    f = torch.stack((fy + lab[1] / 500.0, fy, fy - lab[2] / 200.0))
    delta = 6.0 / 29.0
    xyz = torch.where(f > delta, f.pow(3), 3.0 * delta**2 * (f - 4.0 / 29.0))
    xyz = xyz * lab.new_tensor(_WHITE).view(3, 1, 1, 1)
    linear = torch.einsum("ij,jthw->ithw", lab.new_tensor(_XYZ2RGB), xyz)
    rgb = torch.where(
        linear <= 0.0031308, 12.92 * linear, 1.055 * linear.clamp_min(0.0).pow(1.0 / 2.4) - 0.055
    )
    return rgb.clamp(0.0, 1.0)


def _geometry(x):
    import torch

    h, w = x.shape[-2:]
    scale = min(h, w) / 2.0
    yy, xx = torch.meshgrid(
        (torch.arange(h, device=x.device, dtype=torch.float32) - (h - 1) / 2.0) / scale,
        (torch.arange(w, device=x.device, dtype=torch.float32) - (w - 1) / 2.0) / scale,
        indexing="ij",
    )
    return xx, yy, (xx.square() + yy.square()).sqrt()


def _specular(x, threshold: float, chroma: float):
    luma = (x * x.new_tensor([0.2126, 0.7152, 0.0722]).view(3, 1, 1, 1)).sum(dim=0)[0]
    spread = (x.amax(dim=0) - x.amin(dim=0))[0]
    return (luma > threshold) & (spread < chroma)


def _shrink_highlights(x, p: dict[str, Any]):
    """Remove a share of specular blobs and erode the kept ones (Telea inpainting)."""
    import cv2
    import torch

    R = HANDOFF
    original = _quantize(x)
    quantized = original.float() / 255.0
    mask = _specular(quantized, p["highlight_threshold"], R["highlight_chroma"])
    if int(mask.sum()) == 0:
        return quantized
    components, labels = cv2.connectedComponents(
        mask.cpu().numpy().astype(np.uint8), connectivity=8
    )
    blobs = components - 1
    removed_count = min(blobs, max(0, int(math.floor(p["highlight_remove_share"] * blobs + 0.5))))
    removed_ids = sorted(
        random.Random(p["highlight_blob_seed"]).sample(range(1, components), removed_count)
    )
    lookup = np.zeros(components, dtype=bool)
    lookup[removed_ids] = True
    removed = lookup[labels]
    kept = (labels > 0) & ~removed
    r = p["highlight_erosion_radius"]
    if r == 0:
        core = kept.copy()
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        core = cv2.erode(
            kept.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0
        ).astype(bool)
        areas = np.bincount(labels.ravel(), minlength=components)
        core |= (areas < R["kept_min_erode_area"])[labels] & kept
    removed_dilated = cv2.dilate(
        removed.astype(np.uint8),
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)
    inpaint_mask = removed_dilated | (kept & ~core)
    image = original[:, 0].permute(1, 2, 0).cpu().numpy().copy()
    if 0 < int(inpaint_mask.sum()) < inpaint_mask.size:
        image = cv2.inpaint(
            image,
            inpaint_mask.astype(np.uint8) * 255,
            R["highlight_inpaint_radius"],
            cv2.INPAINT_TELEA,
        )
    out = torch.from_numpy(np.ascontiguousarray(image)).to(x.device)
    return out.permute(2, 0, 1).unsqueeze(1).float() / 255.0


def _energy(x, inside) -> float:
    return float((x - _blur(x, 1.0)).square()[:, 0, inside].mean())


def _grain(x, p: dict[str, Any]):
    """Spatially high-pass, channel-correlated grain scaled to a target high-frequency energy."""
    import torch
    import torch.nn.functional as F

    R = HANDOFF
    base = _quantize(x).float() / 255.0
    _, _, rr = _geometry(base)
    inside = rr <= R["annulus"][1]
    baseline = _energy(base, inside)
    if baseline == 0.0:
        return base
    gen = torch.Generator(device="cpu").manual_seed(p["grain_seed"])
    h, w = base.shape[-2:]
    shared = torch.randn((1, 1, h, w), generator=gen, dtype=torch.float32)
    independent = torch.randn((3, 1, h, w), generator=gen, dtype=torch.float32)
    c = R["grain_channel_correlation"]
    g = (math.sqrt(c) * shared + math.sqrt(1.0 - c) * independent).to(base.device)
    smooth = F.avg_pool2d(
        F.pad(g.permute(1, 0, 2, 3), (1, 1, 1, 1), mode="replicate"), kernel_size=3, stride=1
    ).permute(1, 0, 2, 3)
    noise = g - smooth
    noise = noise - noise[:, 0, inside].mean(dim=1).view(3, 1, 1, 1)
    target, tolerance = p["grain_target"], R["grain_tolerance"]
    best, best_error = base, abs(1.0 / target - 1.0)

    def candidate(scale: float) -> float:
        nonlocal best, best_error
        out = _quantize(base + scale * noise).float() / 255.0
        ratio = _energy(out, inside) / baseline
        error = abs(ratio / target - 1.0)
        if error < best_error:
            best, best_error = out, error
        return ratio

    low = 0.0
    high = max(
        1e-6, 2.0 * math.sqrt((target - 1.0) * baseline / max(_energy(noise, inside), 1e-30))
    )
    high_ratio = candidate(high)
    for _ in range(16):
        if high_ratio >= target or best_error <= tolerance:
            break
        high *= 2.0
        high_ratio = candidate(high)
    if high_ratio >= target:
        for _ in range(R["grain_bisection_iterations"]):
            if best_error <= tolerance:
                break
            middle = (low + high) / 2.0
            if candidate(middle) < target:
                low = middle
            else:
                high = middle
    return best


def corrupt_pixels(frame, p: dict[str, Any]):
    """Degrade a uint8 [3,1,H,W] frame according to pixel-mode parameters."""
    import torch

    if p["mode"] != "pixel":
        return frame
    with torch.no_grad(), torch.autocast(device_type=frame.device.type, enabled=False):
        x = frame.float() / 255.0
        x = (x * x.new_tensor(p["gains"]).view(3, 1, 1, 1)).clamp(0.0, 1.0).pow(p["gamma"])
        lab = _rgb_to_lab(x)
        lab[0] += p["lab_l_shift"]
        lab[1] += p["lab_a_shift"]
        lab[2] *= p["lab_b_scale"]
        x = _lab_to_rgb(lab)
        if p["highlight_shrink"]:
            x = _shrink_highlights(x, p)
        if p["grain"]:
            x = _grain(x, p)
        if p["patches"]:
            xx, yy, rr = _geometry(x)
            lo, hi = HANDOFF["annulus"]
            feather = HANDOFF["annulus_feather"]
            ring = ((rr - lo) / feather).clamp(0.0, 1.0) * ((hi - rr) / feather).clamp(0.0, 1.0)
            lab = _rgb_to_lab(x)
            for patch in p["patches"]:
                weight = (
                    torch.exp(
                        -((xx - patch["x"]).square() + (yy - patch["y"]).square())
                        / (2.0 * patch["sigma"] ** 2)
                    )
                    * ring
                )
                lab[0, 0] += patch["l_shift"] * weight
                lab[1, 0] += patch["a_shift"] * weight
            x = _lab_to_rgb(lab)
        return _quantize(_blur(x, p["blur_sigma"]))


def conditioning_latent(frame, encode: Callable, decode: Callable, p: dict[str, Any]):
    """Latent of a corrupted conditioning frame (uint8 [3,1,H,W]).

    ``encode`` maps [1,3,1,H,W] in [-1, 1] to a latent [1,C,1,h,w]; ``decode`` is its inverse.
    Pixel mode degrades the frame and runs 0-3 VAE round trips; both modes add Gaussian latent
    noise with a standard deviation of ``noise_fraction`` times the latent standard deviation.
    """
    import torch

    frame = corrupt_pixels(frame, p)
    with torch.no_grad(), torch.autocast(device_type=frame.device.type, enabled=False):
        z = encode(frame[None].float() / 127.5 - 1.0)
        for _ in range(p["roundtrips"] if p["mode"] == "pixel" else 0):
            px = decode(z)
            frame = (((px[0].float().clamp(-1.0, 1.0) + 1.0) / 2.0) * 255.0).round().to(torch.uint8)
            z = encode(frame[None].float() / 127.5 - 1.0)
        z = z.float()
        sigma = p["noise_fraction"] * float(z.std(unbiased=False))
        if p["noise_fraction"] > 0.0:
            gen = torch.Generator(device="cpu").manual_seed(p["noise_seed"])
            z = z + torch.randn(z.shape, generator=gen, dtype=torch.float32).to(z.device) * sigma
    return z
