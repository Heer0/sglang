#!/usr/bin/env python3
"""Consolidate the per-role sglang perf dumps into one report.

Each disaggregated role (encoder/denoiser/decoder) is a separate `sglang
generate` process; with --perf-dump-path each writes a benchmark JSON (stages,
denoise_steps_ms, memory_checkpoints, total_duration_ms). This aggregates the
three into a single human-readable table + a merged report.json, so the .sh
ends with detailed per-layer performance — the report the pipeline owes.

Usage:
    soma_perf_report.py OUT.report.json ROLE:JSON:WALL_S [ROLE:JSON:WALL_S ...]

WALL_S is the wall-clock the .sh measured for that role (load + inference);
JSON's total_duration_ms is pure in-server inference, so WALL_S - infer =
the model-load cost of holding only that role's weights. Missing/failed JSONs
degrade gracefully (the role still shows its wall time).
"""
from __future__ import annotations

import json
import os
import sys


def _load(path: str) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _peak(report: dict, key: str) -> float:
    """Max of a memory field across all captured checkpoints."""
    checkpoints = (report or {}).get("memory_checkpoints", {}) or {}
    vals = [snap.get(key, 0.0) or 0.0 for snap in checkpoints.values()]
    return max(vals) if vals else 0.0


def _steps_summary(report: dict) -> tuple[int, float, float, float]:
    steps = [s.get("duration_ms", 0.0) for s in (report or {}).get("denoise_steps_ms", [])]
    if not steps:
        return 0, 0.0, 0.0, 0.0
    return len(steps), sum(steps) / len(steps), min(steps), max(steps)


def _fmt_s(ms: float) -> str:
    return f"{ms / 1000.0:6.2f}"


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2

    out_path = argv[0]
    roles: list[tuple[str, str, float]] = []
    for spec in argv[1:]:
        role, _, rest = spec.partition(":")
        path, _, wall = rest.rpartition(":")
        try:
            wall_s = float(wall)
        except ValueError:
            path, wall_s = rest, 0.0
        roles.append((role, path, wall_s))

    merged: dict = {"roles": {}, "totals": {}}
    rows = []
    tot_wall = tot_infer = 0.0
    peak_vram = peak_host = 0.0

    for role, path, wall_s in roles:
        rep = _load(path)
        infer_ms = float((rep or {}).get("total_duration_ms", 0.0) or 0.0)
        vram = _peak(rep, "peak_reserved_mb")
        host = _peak(rep, "peak_host_anon_mb")
        n, mean_ms, min_ms, max_ms = _steps_summary(rep)
        load_s = max(wall_s - infer_ms / 1000.0, 0.0)

        tot_wall += wall_s
        tot_infer += infer_ms
        peak_vram = max(peak_vram, vram)
        peak_host = max(peak_host, host)

        step_txt = f"{n}x mean {mean_ms:.0f}ms ({min_ms:.0f}-{max_ms:.0f})" if n else "-"
        rows.append(
            (role, f"{wall_s:6.2f}", _fmt_s(infer_ms), f"{load_s:6.2f}",
             f"{vram:8.0f}", f"{host:8.0f}", step_txt)
        )
        meta = (rep or {}).get("meta", {}) or {}
        offload = meta.get("offload", {}) or {}
        file_role = str(meta.get("role", "") or "")
        # A file whose self-reported role doesn't match the slot it was loaded
        # into means the perf dumps got crossed (e.g. stale $$-reused files).
        if file_role and role not in file_role.lower():
            print(f" WARNING: {role} slot holds a dump self-reporting role="
                  f"'{file_role}' — crossed/stale perf files.")
        merged["roles"][role] = {
            "wall_s": round(wall_s, 3),
            "infer_s": round(infer_ms / 1000.0, 3),
            "load_s": round(load_s, 3),
            "peak_vram_mb": round(vram, 1),
            "peak_host_anon_mb": round(host, 1),
            "denoise_steps": {"count": n, "mean_ms": round(mean_ms, 2),
                              "min_ms": round(min_ms, 2), "max_ms": round(max_ms, 2)},
            "stages": (rep or {}).get("steps", []),
            "offload": offload,
            "missing": rep is None,
        }

    merged["totals"] = {
        "wall_s": round(tot_wall, 3),
        "infer_s": round(tot_infer / 1000.0, 3),
        "peak_vram_mb": round(peak_vram, 1),
        "peak_host_anon_mb": round(peak_host, 1),
    }

    hdr = ("ROLE", "wall_s", "infer_s", "load_s", "vram_MB", "host_MB", "denoise")
    line = "-" * 92
    print("\n" + line)
    print(" SOMA disaggregated LTX-2 — per-role performance report")
    print(line)
    print(f"{hdr[0]:<9}{hdr[1]:>7}{hdr[2]:>8}{hdr[3]:>8}{hdr[4]:>9}{hdr[5]:>9}   {hdr[6]}")
    print(line)
    for r in rows:
        print(f"{r[0]:<9}{r[1]:>7}{r[2]:>8}{r[3]:>8}{r[4]:>9}{r[5]:>9}   {r[6]}")
    print(line)
    print(f"{'TOTAL':<9}{tot_wall:>7.2f}{_fmt_s(tot_infer):>8}{'':>8}"
          f"{peak_vram:>9.0f}{peak_host:>9.0f}   (peak = max across roles)")
    print(line)
    print(" note: roles run sequentially — peak VRAM/host is the max of ONE")
    print(" resident component at a time, not their sum. That is the win.")
    print(line)

    # offload detail (A config + B runtime) for whichever roles reported it.
    any_off = False
    for role, _, _ in roles:
        off = merged["roles"].get(role, {}).get("offload") or {}
        cfg = off.get("config") or {}
        if not cfg and not off.get("h2d_bytes"):
            continue
        if not any_off:
            print(" OFFLOAD (streamed weights per role)")
            print(line)
            any_off = True
        # config can hold several component groups; fold into one line each.
        for comp, c in cfg.items():
            print(f"  {role:<8} {comp:<10} resident {c.get('resident_layers', 0)}/"
                  f"{c.get('total_layers', 0)}  streamed {c.get('streamed_layers', 0)}"
                  f"  prefetch {c.get('prefetch_per_group', '?')}"
                  f"  policy {c.get('policy', '?')}")
        vol = f"{off.get('h2d_gib', 0.0):.2f} GiB over {off.get('prefetch_ops', 0)} prefetch ops"
        timing = (f"  |  copy-stream {off['copy_stream_ms']:.0f}ms / {off.get('copy_events', 0)} ev"
                  if off.get("profiled") and "copy_stream_ms" in off
                  else "  |  copy-time: run with SOMA_OFFLOAD_PROFILE=1")
        print(f"  {role:<8} {'H2D':<10} {vol}{timing}")
    if any_off:
        print(line)
    print("")

    try:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2)
        print(f" merged report: {out_path}\n")
    except OSError as e:
        print(f" WARNING: could not write {out_path}: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
