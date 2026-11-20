# SPDX-License-Identifier: CC-BY-NC-4.0
"""Camera-control metrics from a camera read-back of the generated video.

The generated video is passed through an external monocular SfM / depth estimator (see
``docs/evaluation.md``), which returns per-frame camera-to-world poses. These are compared with the
commanded (reference) camera path.

Conventions: ``P`` is the reference path and ``Q`` the read-back, both ``(N, 4, 4)`` camera-to-world
matrices in OpenCV axes (x right, y down, z forward) with frame 0 equal to the identity and
translations in units of the frame-0 median scene depth.

Main protocol (windowed, used for the main table):
    windows of 16 frames sampled at stride 8 (frames ``w, w+8, ..., w+120``) for ``w = 0, 8, ...``;
    each window is rebased to its first camera, translations of each trajectory are divided by
    their maximum distance from the first camera, and
    ``RotErr`` = sum of geodesic angles [rad], ``TransErr`` = sum of L2 distances,
    ``CamMC`` = sum of Frobenius norms of the 3x4 pose differences. Per clip: mean over windows.
    A read-back without a valid output takes the value of a static camera (``Q = I``).

Summed protocol (188-frame sums with a global scale ``s`` and caps for untracked clips) and
the long-video protocol (chunk stitching, Sim(3) ATE, RPE, rotation drift) are also provided.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

EPS = 1e-9
WINDOW_KEYS = ("RotErr", "TransErr", "CamMC")


# --------------------------------------------------------------------------------------- loading
def load_readback(
    path: str | Path, key: str = "relative_c2w", orthonormalize: bool = False
) -> np.ndarray:
    """Load read-back poses ``(N, 4, 4)`` (float64) from an estimator output ``.npz``.

    Frames flagged invalid by an optional boolean ``valid`` array are set to NaN. With
    ``orthonormalize`` the rotation blocks are projected onto SO(3).
    """
    with np.load(path, allow_pickle=False) as z:
        poses = z[key].astype(np.float64)
        valid = z["valid"].astype(bool) if "valid" in z.files else np.ones(len(poses), bool)
    poses[~valid] = np.nan
    return orthonormalize_poses(poses) if orthonormalize else poses


def is_valid_readback(Q: np.ndarray | None, n_frames: int) -> bool:
    """True when the read-back has ``n_frames`` finite poses."""
    return Q is not None and np.asarray(Q).shape == (n_frames, 4, 4) and bool(np.isfinite(Q).all())


def static_path(n_frames: int) -> np.ndarray:
    """Identity poses: the read-back of a camera that does not move."""
    return np.tile(np.eye(4), (n_frames, 1, 1))


# --------------------------------------------------------------------------------- windowed (main)
def rebase(T: np.ndarray) -> np.ndarray:
    """Express poses relative to the first one: ``inv(T[0]) @ T``."""
    return np.linalg.inv(T[0])[None] @ T


def normalize_translation(T: np.ndarray) -> np.ndarray:
    """Translations divided by their maximum norm (+1e-9)."""
    t = T[:, :3, 3]
    return t / (np.linalg.norm(t, axis=1).max() + EPS)


def window_metrics(P: np.ndarray, Q: np.ndarray) -> dict[str, float]:
    """RotErr [rad], TransErr and CamMC of one sampled frame set (rebased and normalized here)."""
    Pb, Qb = rebase(P), rebase(Q)
    RP, RQ = Pb[:, :3, :3], Qb[:, :3, :3]
    cos = (np.trace(np.swapaxes(RP, 1, 2) @ RQ, axis1=1, axis2=2) - 1) / 2
    angle = np.arccos(np.clip(cos, -1, 1))
    tP, tQ = normalize_translation(Pb), normalize_translation(Qb)
    A = np.concatenate([RP, tP[..., None]], -1)
    B = np.concatenate([RQ, tQ[..., None]], -1)
    return {
        "RotErr": float(angle.sum()),
        "TransErr": float(np.linalg.norm(tQ - tP, axis=1).sum()),
        "CamMC": float(np.linalg.norm((B - A).reshape(len(A), -1), axis=1).sum()),
    }


def window_indices(
    n_frames: int, frames: int = 16, frame_stride: int = 8, start_step: int = 8
) -> list[np.ndarray]:
    """Frame indices of every evaluation window inside ``n_frames``."""
    span = (frames - 1) * frame_stride
    return [np.arange(w, w + span + 1, frame_stride) for w in range(0, n_frames - span, start_step)]


def direction_error(
    P: np.ndarray, Q: np.ndarray, gt_min_disp: float, readback_zero_disp: float
) -> tuple[float | None, str]:
    """Angle [deg] between reference and read-back displacement of the last frame of a window.

    Returns ``(None, "excluded")`` when the reference barely moves, ``(90.0, "zero")`` when the
    read-back barely moves, otherwise ``(angle, "ok")``.
    """
    tp, tq = rebase(P)[-1, :3, 3], rebase(Q)[-1, :3, 3]
    a, b = np.linalg.norm(tp), np.linalg.norm(tq)
    if a < gt_min_disp:
        return None, "excluded"
    if b < readback_zero_disp:
        return 90.0, "zero"
    return float(np.degrees(np.arccos(np.clip(tp @ tq / (a * b), -1, 1)))), "ok"


def clip_windowed_metrics(
    P: np.ndarray,
    Q: np.ndarray,
    frames: int = 16,
    frame_stride: int = 8,
    start_step: int = 8,
    gt_min_disp: float = 0.05,
    readback_zero_disp: float = 0.005,
) -> dict[str, Any]:
    """Main per-clip metrics: mean over windows of RotErr, TransErr, CamMC and DirErr [deg]."""
    idx = window_indices(len(P), frames, frame_stride, start_step)
    per = [window_metrics(P[ix], Q[ix]) for ix in idx]
    out: dict[str, Any] = {k: float(np.mean([p[k] for p in per])) for k in WINDOW_KEYS}
    dirs = [direction_error(P[ix], Q[ix], gt_min_disp, readback_zero_disp) for ix in idx]
    used = [a for a, flag in dirs if flag != "excluded"]
    out.update(
        DirErr=float(np.mean(used)) if used else None,
        n_windows=len(per),
        n_dir_windows=len(used),
        n_dir_excluded=sum(flag == "excluded" for _, flag in dirs),
        n_dir_zero_readback=sum(flag == "zero" for _, flag in dirs),
    )
    return out


def clip_subsampled_metrics(
    P: np.ndarray, Q: np.ndarray, frames: int = 16, step: int = 12
) -> dict[str, float]:
    """Single-set variant: ``frames`` frames ``0, step, ...`` rebased to frame 0."""
    ix = np.arange(frames) * step
    return window_metrics(P[ix], Q[ix])


def score_clip_windowed(
    P: np.ndarray, Q: np.ndarray | None, settings: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Windowed metrics of one clip; an invalid read-back takes the static-camera value."""
    settings = settings or {}
    untracked = not is_valid_readback(Q, len(P))
    Qe = static_path(len(P)) if untracked else Q
    return {"untracked": untracked, **clip_windowed_metrics(P, Qe, **settings)}


def aggregate(rows: list[dict[str, Any]], keys: tuple[str, ...] = WINDOW_KEYS + ("DirErr",)):
    """Table value: mean over clips per key (``None`` skipped) and the untracked count."""
    out: dict[str, Any] = {
        "n_clips": len(rows),
        "n_untracked": sum(bool(r["untracked"]) for r in rows),
    }
    for k in keys:
        vals = [r[k] for r in rows if r.get(k) is not None]
        out[k] = float(np.mean(vals)) if vals else None
    return out


# ---------------------------------------------------------------------------- summed (appendix)
SUMMED_KEYS = ("rot_err_sum_deg", "rot_err_mean_deg", "trans_err_sum", "cammc")


def rotation_angles_deg(RP: np.ndarray, RQ: np.ndarray) -> np.ndarray:
    """Geodesic angle [deg] ``arccos((tr(R_Q R_P^T) - 1) / 2)`` of rotation stacks."""
    cos = (np.trace(RQ @ np.swapaxes(RP, -1, -2), axis1=-2, axis2=-1) - 1) / 2
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def summed_metrics(P: np.ndarray, Q: np.ndarray, scale: float | None) -> dict[str, float | None]:
    """Sums over frames ``1..N-1``; ``scale`` multiplies the read-back translation.

    With ``scale=None`` only the rotation terms are computed.
    """
    RP, RQ, tP, tQ = P[1:, :3, :3], Q[1:, :3, :3], P[1:, :3, 3], Q[1:, :3, 3]
    rot = rotation_angles_deg(RP, RQ)
    out: dict[str, float | None] = {
        "rot_err_sum_deg": float(rot.sum()),
        "rot_err_mean_deg": float(rot.mean()),
        "trans_err_sum": None,
        "cammc": None,
    }
    if scale is not None:
        out["trans_err_sum"] = float(np.linalg.norm(scale * tQ - tP, axis=1).sum())
        A = np.concatenate([RQ, (scale * tQ)[..., None]], -1)
        B = np.concatenate([RP, tP[..., None]], -1)
        out["cammc"] = float(np.linalg.norm(A - B, axis=(1, 2)).sum())
    return out


def score_clip_summed(
    P: np.ndarray, Q: np.ndarray | None, scale: float | None, cap: dict[str, float]
) -> dict[str, Any]:
    """Summed metrics clipped at ``cap``; a read-back of the wrong length or none takes the cap."""
    if Q is None or len(Q) != len(P):
        rec: dict[str, Any] = {"untracked": True}
        for k in SUMMED_KEYS:
            rec[k] = None
            usable = scale is not None or k.startswith("rot")
            rec[k + "_capped"] = cap[k] if usable else None
        return rec
    m = summed_metrics(P, Q, scale)
    rec = {"untracked": False, **m}
    for k in SUMMED_KEYS:
        rec[k + "_capped"] = None if m[k] is None else min(m[k], cap[k])
    return rec


def fit_scale(pairs: list[tuple[np.ndarray, np.ndarray | None]]) -> tuple[float | None, int]:
    """Global least-squares scale ``sum <t_Q, t_P> / sum <t_Q, t_Q>`` over usable clips."""
    num = den = 0.0
    n = 0
    for P, Q in pairs:
        if Q is None or len(Q) != len(P) or np.abs(P[1:, :3, 3]).sum() < 1e-9:
            continue
        num += float((Q[1:, :3, 3] * P[1:, :3, 3]).sum())
        den += float((Q[1:, :3, 3] ** 2).sum())
        n += 1
    return (num / den if den > 0 else None), n


# ------------------------------------------------------------------------- full-procedure chunks
def chunk_grid(n_frames: int, stride: int = 188) -> list[tuple[int, int]]:
    """Generation chunks ``(start, end)`` (inclusive) of a chained video with ``n_frames`` frames.

    Chunk ``k`` covers global frames ``stride*k .. stride*(k+1)``; consecutive chunks share the
    hand-off frame.
    """
    if n_frames <= 1 or (n_frames - 1) % stride:
        raise ValueError(f"{n_frames} frames is not 1 + a multiple of {stride}")
    return [(stride * k, stride * (k + 1)) for k in range((n_frames - 1) // stride)]


def stitched_frame_owner(t: int, stride: int = 188) -> int:
    """Chunk whose frame is used for global frame ``t`` (chunk 0 owns 0..stride)."""
    return 0 if t <= stride else (t - 1) // stride


def score_procedure(
    commands: np.ndarray,
    readbacks: list[np.ndarray | None],
    settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Full-procedure camera metrics: windowed metrics of each chunk read-back vs its command.

    Args:
        commands: ``(K, L, 4, 4)`` per-chunk commanded paths (each rebased to its first frame).
        readbacks: per-chunk read-back poses (``None`` / non-finite: static value).
        settings: window settings passed to :func:`clip_windowed_metrics`.
    """
    rows = []
    for k, Q in enumerate(readbacks):
        rows.append({"chunk": k, **score_clip_windowed(commands[k], Q, settings)})
    tracked = [r for r in rows if not r["untracked"]]
    keys = WINDOW_KEYS + ("DirErr",)
    return {
        "chunks": rows,
        "mean": {k: v for k, v in aggregate(rows, keys).items() if k in keys},
        "mean_tracked_only": {k: v for k, v in aggregate(tracked, keys).items() if k in keys},
        "n_untracked": len(rows) - len(tracked),
    }


# -------------------------------------------------------------------------- long-video stitching
LONG_CHUNK, LONG_OVERLAP, MIN_WINDOW, RPE_DELTA, FPS = 240, 48, 96, 24, 24.0


def grid_reference(n: int, length: int = LONG_CHUNK, overlap: int = LONG_OVERLAP):
    """Reference read-back grid: starts ``0, length-overlap, ...``, last chunk ending at n-1."""
    if n <= length:
        return [(0, n - 1)]
    starts = list(range(0, n - length + 1, length - overlap))
    if starts[-1] + length < n:
        starts.append(n - length)
    return [(s, s + length - 1) for s in starts]


def grid_scoring(
    n: int,
    length: int = LONG_CHUNK,
    overlap: int = LONG_OVERLAP,
    first: int = 144,
    offset: int = 96,
    min_last: int = 96,
):
    """Scoring grid whose chunk boundaries all differ from :func:`grid_reference`."""
    if n <= first:
        return [(0, n - 1)]
    out = [(0, first - 1)]
    s = offset
    while s + length - 1 <= n - 1:
        out.append((s, s + length - 1))
        s += length - overlap
    if out[-1][1] < n - 1:
        if len(out) > 1:
            st = min(out[-1][0] + (length - overlap), n - min_last)
        else:
            st = max(first - overlap, n - length)
        st = max(st, out[-1][0] + 1)
        if (st, n - 1) in grid_reference(n, length, overlap):
            st = max(out[-1][0] + 1, st - 24)
        out.append((st, n - 1))
    return out


def rotation_angle_deg(R: np.ndarray) -> np.ndarray:
    """Rotation angle via ``atan2(|skew|, (tr-1)/2)`` (stable near zero)."""
    R = np.asarray(R)
    c = (np.trace(R, axis1=-2, axis2=-1) - 1) / 2
    w = (
        np.stack(
            [R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]],
            -1,
        )
        / 2
    )
    return np.degrees(np.arctan2(np.linalg.norm(w, axis=-1), c))


def orthonormalize_poses(P: np.ndarray) -> np.ndarray:
    """Project the rotation blocks of ``(N, 4, 4)`` poses onto SO(3); NaN rows stay NaN."""
    P = np.array(P, float)
    ok = np.isfinite(P.reshape(len(P), -1)).all(1)
    if ok.any():
        U, _, Vt = np.linalg.svd(P[ok, :3, :3])
        D = np.ones((ok.sum(), 3))
        D[:, 2] = np.sign(np.linalg.det(U @ Vt))
        P[ok, :3, :3] = (U * D[:, None, :]) @ Vt
    return P


def chordal_mean(Rs: np.ndarray) -> np.ndarray:
    """Chordal L2 mean of a rotation stack."""
    U, _, Vt = np.linalg.svd(np.sum(Rs, 0))
    return U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt


def _so3_log(R: np.ndarray) -> np.ndarray:
    th = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if th < 1e-9:
        return np.zeros(3)
    if np.pi - th < 1e-6:
        A = (R + np.eye(3)) / 2
        i = int(np.argmax(np.diag(A)))
        return A[:, i] / np.sqrt(max(A[i, i], 1e-12)) * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(th))
    return w * th


def _so3_exp(v: np.ndarray) -> np.ndarray:
    th = np.linalg.norm(v)
    if th < 1e-12:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def slerp(Ra: np.ndarray, Rb: np.ndarray, w: float) -> np.ndarray:
    """Geodesic interpolation from ``Ra`` (w=0) to ``Rb`` (w=1)."""
    return Ra @ _so3_exp(w * _so3_log(Ra.T @ Rb))


def stitch_chunks(chunks: list[dict[str, Any]], n: int) -> dict[str, Any]:
    """Stitch overlapping read-back chunks into one path.

    Each chunk is ``dict(start, end, P)`` with ``P`` frame-0 relative poses in the chunk's own
    translation unit (``P=None`` for an untracked chunk). Rotations are chained (chordal mean
    over the overlap, SLERP blend inside it); translations keep each chunk's unit and switch at the
    overlap middle with a continuous centre. Frames of an untracked chunk not covered by a
    neighbour hold the last pose, are flagged ``filled`` and start a new rotation ``piece``.
    """
    G = np.full((n, 4, 4), np.nan)
    src = np.full(n, -1)
    filled = np.zeros(n, bool)
    piece = np.full(n, -1)
    aligned: dict[int, np.ndarray] = {}
    prev, pid = None, -1
    for k, c in enumerate(chunks):
        s, e = int(c["start"]), int(c["end"])
        if c.get("P") is None:
            continue
        P = np.asarray(c["P"], float)
        if len(P) != e - s + 1:
            raise ValueError(f"chunk {k}: {len(P)} poses for frames {s}..{e}")
        if prev is not None and s <= chunks[prev]["end"]:
            ps, o0, o1 = int(chunks[prev]["start"]), s, int(chunks[prev]["end"])
            Ap = aligned[prev]
            ov = np.arange(o0, o1 + 1)
            Ra, Rb = Ap[ov - ps, :3, :3], P[ov - s, :3, :3]
            Rk = chordal_mean(np.einsum("nij,nkj->nik", Ra, Rb))
            m = (o0 + o1 + 1) // 2
            Ak = np.repeat(np.eye(4)[None], len(P), 0)
            Ak[:, :3, :3] = Rk[None] @ P[:, :3, :3]
            ck = (Rk @ P[:, :3, 3].T).T
            Ak[:, :3, 3] = ck - ck[m - s] + Ap[m - ps, :3, 3]
            aligned[k] = Ak
            for t in range(m, e + 1):
                G[t], src[t], piece[t] = Ak[t - s], k, pid
            for t in range(o0, o1 + 1):
                w = (t - o0 + 1) / (o1 - o0 + 2)
                G[t, :3, :3] = slerp(Ap[t - ps, :3, :3], Ak[t - s, :3, :3], w)
        else:
            pid += 1
            Ak = P.copy()
            if prev is not None:
                pe = int(chunks[prev]["end"])
                last = aligned[prev][pe - int(chunks[prev]["start"])]
                Ak = last[None] @ P
                for t in range(pe + 1, s):
                    G[t], src[t], filled[t], piece[t] = last, -1, True, -1
            aligned[k] = Ak
            for t in range(s, e + 1):
                G[t], src[t], piece[t] = Ak[t - s], k, pid
        prev = k
    if prev is not None and int(chunks[prev]["end"]) < n - 1:
        pe = int(chunks[prev]["end"])
        last = aligned[prev][pe - int(chunks[prev]["start"])]
        for t in range(pe + 1, n):
            G[t], filled[t] = last, True
    return {"G": G, "src": src, "filled": filled, "piece": piece}


def commands_from_path(G: np.ndarray, length: int = 189) -> np.ndarray:
    """Split a stitched path into per-chunk commands ``(K, length, 4, 4)`` (shared hand-off)."""
    G = np.array(G, float)
    ok = np.isfinite(G.reshape(len(G), -1)).all(1)
    if not ok.all():
        first = int(np.argmax(ok))
        G[:first] = G[first]
    G = np.linalg.inv(G[0])[None] @ G
    n = len(G)
    out = []
    for k in range(int(np.ceil((n - 1) / (length - 1)))):
        seg = G[(length - 1) * k : (length - 1) * k + length]
        seg = np.linalg.inv(seg[0])[None] @ seg
        if len(seg) < length:
            seg = np.concatenate([seg, np.repeat(seg[-1:], length - len(seg), 0)])
        out.append(seg)
    return np.stack(out)


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    """Sim(3) ``(s, R, t, degenerate)`` minimising ``|dst - (s R src + t)|^2`` (Umeyama 1991)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    var_s = (xs**2).sum() / len(src)
    U, S, Vt = np.linalg.svd(xd.T @ xs / len(src))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt)) or 1.0
    R = U @ D @ Vt
    s = float((S * np.diag(D)).sum() / var_s) if (with_scale and var_s > 1e-12) else 1.0
    return s, R, mu_d - s * R @ mu_s, bool(var_s <= 1e-12)


def _stats(x: list[float | None]) -> dict[str, Any]:
    x = [v for v in x if v is not None and np.isfinite(v)]
    return {
        "n": len(x),
        "mean": float(np.mean(x)) if x else None,
        "median": float(np.median(x)) if x else None,
    }


def score_long(
    gt: dict[str, Any],
    est_chunks: list[dict[str, Any]],
    n: int,
    fps: float = FPS,
    delta: int = RPE_DELTA,
    min_window: int = MIN_WINDOW,
) -> dict[str, Any]:
    """Long-video trajectory metrics.

    ``gt`` is :func:`stitch_chunks` of the reference chunks; ``est_chunks`` are the read-back
    chunks of the scored video on the scoring grid. Windows are (reference unit segment x estimate
    chunk) intersections of at least ``min_window`` frames. Per window: Sim(3) ATE, normalized ATE
    (ATE / RMS spread of the reference centres) and RPE over ``delta``-frame pairs. Rotation drift
    is the least-squares slope [deg/min] of the chained rotation error over unbroken pieces.
    """
    G, src, filled = gt["G"], gt["src"], gt["filled"]
    gt_ok = np.isfinite(G.reshape(n, -1)).all(1) & ~filled & (src >= 0)
    cover = np.zeros(n, bool)
    for c in est_chunks:
        if c.get("P") is not None:
            cover[c["start"] : c["end"] + 1] = True
    out: dict[str, Any] = {
        "frames": n,
        "gt_frames": int(gt_ok.sum()),
        "untracked_frame_fraction": float(1 - cover[gt_ok].mean()) if gt_ok.any() else None,
    }
    windows = []
    for a in sorted({int(x) for x in src if x >= 0}):
        seg = np.nonzero((src == a) & gt_ok)[0]
        for b, c in enumerate(est_chunks):
            f = seg[(seg >= c["start"]) & (seg <= c["end"])]
            if len(f) >= min_window:
                windows.append((a, b, f))
    rows, rpe_t, rpe_r = [], [], []
    for a, b, f in windows:
        c = est_chunks[b]
        row: dict[str, Any] = {
            "gt_chunk": a,
            "est_chunk": b,
            "first": int(f[0]),
            "last": int(f[-1]),
        }
        if c.get("P") is None:
            rows.append({**row, "untracked": True})
            continue
        Q = np.asarray(c["P"], float)[f - c["start"]]
        Pw = G[f]
        okq = np.isfinite(Q.reshape(len(Q), -1)).all(1)
        if okq.sum() < 3:
            rows.append({**row, "untracked": True})
            continue
        Q, Pw, fw = Q[okq], Pw[okq], f[okq]
        s, R, t, _ = umeyama(Q[:, :3, 3], Pw[:, :3, 3])
        res = np.linalg.norm((s * (R @ Q[:, :3, 3].T)).T + t - Pw[:, :3, 3], axis=1)
        spread = float(np.sqrt(((Pw[:, :3, 3] - Pw[:, :3, 3].mean(0)) ** 2).sum(1).mean()))
        ate = float(np.sqrt((res**2).mean()))
        idx = {int(x): i for i, x in enumerate(fw)}
        tr, ro = [], []
        for i, x in enumerate(fw):
            j = idx.get(int(x) + delta)
            if j is None:
                continue
            E = np.linalg.inv(Pw[i]) @ Pw[j]
            Qi, Qj = Q[i].copy(), Q[j].copy()
            Qi[:3, 3] *= s
            Qj[:3, 3] *= s
            D = np.linalg.inv(E) @ (np.linalg.inv(Qi) @ Qj)
            tr.append(float(np.linalg.norm(D[:3, 3])))
            ro.append(float(rotation_angle_deg(D[:3, :3])))
        rpe_t += tr
        rpe_r += ro
        rows.append(
            {
                **row,
                "untracked": False,
                "scale": s,
                "ate": ate,
                "nate": (ate / spread) if spread > 1e-9 else None,
                "rpe_trans": float(np.mean(tr)) if tr else None,
                "rpe_rot_deg": float(np.mean(ro)) if ro else None,
            }
        )
    ok = [r for r in rows if not r["untracked"]]
    out["windows"] = {"count": len(rows), "untracked": len(rows) - len(ok)}
    out["ate"] = _stats([r["ate"] for r in ok])
    out["nate"] = _stats([r["nate"] for r in ok])
    out["rpe"] = {"pairs": len(rpe_t), "trans": _stats(rpe_t), "rot_deg": _stats(rpe_r)}

    E = stitch_chunks(est_chunks, n)
    RG, RE = G[:, :3, :3], E["G"][:, :3, :3]
    both = gt_ok & np.isfinite(RE.reshape(n, -1)).all(1) & ~E["filled"]
    err = np.full(n, np.nan)
    tau = np.full(n, np.nan)
    t = 0
    while t < n:
        if not both[t]:
            t += 1
            continue
        t0, pg, pe = t, gt["piece"][t], E["piece"][t]
        while t < n and both[t] and gt["piece"][t] == pg and E["piece"][t] == pe:
            t += 1
        seg = np.arange(t0, t)
        rg = np.einsum("ji,njk->nik", RG[t0], RG[seg])
        re = np.einsum("ji,njk->nik", RE[t0], RE[seg])
        err[seg] = rotation_angle_deg(np.einsum("nji,njk->nik", rg, re))
        tau[seg] = (seg - t0) / fps / 60.0
    m = np.isfinite(err)
    denom = (tau[m] ** 2).sum() if m.any() else 0.0
    out["rotation_drift_deg_per_min"] = (
        float((err[m] * tau[m]).sum() / denom) if denom > 0 else None
    )
    out["rotation_drift_mean_deg"] = float(np.nanmean(err)) if m.any() else None
    out["window_rows"] = rows
    return out
