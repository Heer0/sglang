"""Process-global offload accounting for the perf report.

Each disaggregated role runs in its own ``sglang generate`` process, so a plain
module-global naturally scopes to one role. Two things are captured:

(A) config  — how offload was set up (groups, layers, resident vs streamed,
    prefetch size, policy). Registered once when the managers are ready.
(B) runtime — always-on H2D byte count + prefetch-op count (pure Python adds,
    off the CUDA path); and, ONLY when SOMA_OFFLOAD_PROFILE=1, the actual
    copy-stream time via paired CUDA events summed at snapshot (device idle by
    then). Timing is opt-in because this pipeline's kernel dispatch proved
    sensitive to added device work — the default path stays untouched.

``snapshot()`` returns a JSON-able dict folded into the perf dump's ``meta``.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Tuple

PROFILE: bool = os.environ.get("SOMA_OFFLOAD_PROFILE", "0") == "1"

_h2d_bytes: int = 0
_prefetch_ops: int = 0
_config: Dict[str, Any] = {}
_event_pairs: List[Tuple[Any, Any]] = []
_itemsize_cache: Dict[Any, int] = {}


def register_config(
    *,
    component: str,
    groups: int,
    total_layers: int,
    resident_layers: int,
    prefetch_per_group: str,
    policy: str,
) -> None:
    """Record the offload configuration (B's context / A)."""
    _config[component] = {
        "groups": groups,
        "total_layers": total_layers,
        "resident_layers": resident_layers,
        "streamed_layers": max(total_layers - resident_layers, 0),
        "prefetch_per_group": prefetch_per_group,
        "policy": policy,
    }


def _itemsize(dtype) -> int:
    hit = _itemsize_cache.get(dtype)
    if hit is not None:
        return hit
    try:
        import torch

        size = torch.empty(0, dtype=dtype).element_size()
    except Exception:
        size = 0
    _itemsize_cache[dtype] = size
    return size


def add_layer(weight_metadata_values: Iterable[Dict[str, Any]]) -> None:
    """Account one prefetched layer's H2D volume from its weight metadata.

    Called after the copies; pure Python, touches no CUDA. Covers every hosting
    path (consolidated / strided / mapped) since it reads shape+dtype.
    """
    global _h2d_bytes, _prefetch_ops
    total = 0
    for meta in weight_metadata_values:
        shape = meta.get("shape")
        dtype = meta.get("dtype")
        if shape is None or dtype is None:
            continue
        numel = 1
        for dim in shape:
            numel *= int(dim)
        total += numel * _itemsize(dtype)
    _h2d_bytes += total
    _prefetch_ops += 1


def record_pair(start_event, end_event) -> None:
    """Stash a copy-stream event pair (opt-in profiling only)."""
    if start_event is not None and end_event is not None:
        _event_pairs.append((start_event, end_event))


def snapshot() -> Dict[str, Any]:
    """JSON-able offload summary for the perf dump's meta."""
    out: Dict[str, Any] = {
        "config": _config,
        "h2d_bytes": _h2d_bytes,
        "h2d_gib": round(_h2d_bytes / (1024**3), 3),
        "prefetch_ops": _prefetch_ops,
        "profiled": PROFILE,
    }
    if PROFILE and _event_pairs:
        total_ms = 0.0
        counted = 0
        for start, end in _event_pairs:
            try:
                total_ms += float(start.elapsed_time(end))
                counted += 1
            except Exception:
                pass
        out["copy_stream_ms"] = round(total_ms, 2)
        out["copy_events"] = counted
    return out


def reset() -> None:
    global _h2d_bytes, _prefetch_ops
    _h2d_bytes = 0
    _prefetch_ops = 0
    _config.clear()
    _event_pairs.clear()
