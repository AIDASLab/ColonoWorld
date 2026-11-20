# SPDX-License-Identifier: CC-BY-NC-4.0
"""YAML configuration loading with environment-variable and dotted-key overrides."""

from __future__ import annotations

import copy
import os
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand(value: Any, variables: dict[str, str]) -> Any:
    """Recursively replace ``${NAME}`` in strings using ``variables`` then the environment."""
    if isinstance(value, str):
        return _VAR.sub(
            lambda m: variables.get(m.group(1), os.environ.get(m.group(1), m.group(0))), value
        )
    if isinstance(value, dict):
        return {k: _expand(v, variables) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v, variables) for v in value]
    return value


def _set_dotted(cfg: dict[str, Any], key: str, value: Any) -> None:
    node = cfg
    *parents, leaf = key.split(".")
    for part in parents:
        node = node.setdefault(part, {})
    node[leaf] = value


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> dict[str, Any]:
    """Load a YAML config.

    Args:
        path: YAML file. A top-level ``base: other.yaml`` key is merged first and an
            ``include: [a.yaml, ...]`` list is merged after the file (both relative to ``path``).
        overrides: ``key.sub=value`` strings; values are parsed as YAML scalars.

    Returns:
        The merged configuration with ``${VAR}`` references resolved from the ``paths`` section
        and the environment.
    """
    path = Path(path)
    cfg = yaml.safe_load(path.read_text()) or {}
    base = cfg.pop("base", None)
    includes = cfg.pop("include", None) or []
    if base is not None:
        merged = load_config(path.parent / base)
        _deep_update(merged, cfg)
        cfg = merged
    for name in [includes] if isinstance(includes, str) else includes:
        _deep_update(cfg, yaml.safe_load((path.parent / name).read_text()) or {})
    cfg = copy.deepcopy(cfg)
    for item in overrides:
        key, _, raw = item.partition("=")
        _set_dotted(cfg, key.strip(), yaml.safe_load(raw))
    variables = {k: str(v) for k, v in (cfg.get("paths") or {}).items()}
    return _expand(cfg, variables)


def _deep_update(dst: dict[str, Any], src: dict[str, Any]) -> None:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_update(dst[key], value)
        else:
            dst[key] = value
