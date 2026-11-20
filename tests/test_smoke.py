# SPDX-License-Identifier: CC-BY-NC-4.0
"""Smoke test: every module imports and every config loads on CPU."""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

import pytest

import colonoworld
from colonoworld.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]
MODULES = sorted(m.name for m in pkgutil.walk_packages(colonoworld.__path__, prefix="colonoworld."))
CONFIGS = sorted((ROOT / "configs").glob("*.yaml"))


@pytest.mark.parametrize("name", MODULES)
def test_import(name: str) -> None:
    importlib.import_module(name)


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_config_loads(path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = ("COLONOWORLD_DATA", "COLONOWORLD_CKPT", "COSMOS_WEIGHTS")
    for var in env + ("COLONOWORLD_DEPTH_CODE", "COLONOWORLD_DEPTH_WEIGHTS"):
        monkeypatch.setenv(var, f"/data/{var.lower()}")
    cfg = load_config(path, overrides=["smoke.flag=true"])
    assert isinstance(cfg, dict) and cfg["smoke"]["flag"] is True
    assert "${COLONOWORLD" not in str(cfg) and "${COSMOS" not in str(cfg)


def test_dry_run_config_and_palette() -> None:
    """Tiny CPU dry run: inference config plus a palette render of random masks."""
    import numpy as np

    from colonoworld.conditions.palette import MASK_CLASSES, render_palette_video

    cfg = load_config(ROOT / "configs" / "inference.yaml")
    assert cfg
    rng = np.random.default_rng(0)
    masks = {c: rng.random((2, 16, 16)) > 0.7 for c in MASK_CLASSES}
    video = render_palette_video(masks)
    assert video.shape == (2, 16, 16, 3) and video.dtype == np.uint8
