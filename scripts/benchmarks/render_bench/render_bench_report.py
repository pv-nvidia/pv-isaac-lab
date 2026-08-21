# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Charts and pooled statistics for the render-bench sample dumps (NVBug 6431561).

Reads the TSVs written by ``render_bench_runner.py --samples_out`` and, **per
``guest_halt_poll_ns`` value**, emits

* a pooled stats table — including the **total** time spent inside the metric and
  its share of the measured-loop wall clock, which the median/p95/p99 view hides;
* an ASCII sorted-duration curve on a "nines" x-axis (``-log10(1-p)``), so the
  p99/p999 tail gets as much width as the body;
* an ASCII box plot and an ASCII call-ordered view (spikes vs drift);
* the same three views as a standalone SVG, plus a small-multiples SVG that puts
  every halt-poll value on one shared pair of axes.

Every chart in a run shares one global log y-range, so the per-halt-poll SVGs can
be flipped through or laid side by side and compared directly.

Standalone and stdlib-only — run it on any set of sample files, from any
directory, long after the benchmark finished::

    python3 render_bench_report.py --samples-dir logs/run_20260821-120000/samples
    python3 render_bench_report.py --samples a.tsv b.tsv --svg-dir /tmp/charts
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import re
import statistics
import sys
from dataclasses import dataclass, field

# Fixed categorical slot order (validated data-viz palette, slots 1 and 2): the
# renderer owns its hue everywhere, so a chart with only one arm never repaints
# the survivor. Light and dark are separately stepped for their own surface.
ARMS = ("newton", "ovrtx")
_SLOT = {"newton": 1, "ovrtx": 2}

# Nearest-rank sample counts below which a quantile is just max_ms — same
# thresholds the runner warns on (keep in sync with render_bench_runner.py).
_MIN_CALLS = {"p99": 52, "p999": 502}
# Beyond this many remaining samples the curve's shape is one-off outliers, not
# a distribution: charts shade it and the tables footnote it.
_THIN_TAIL = 10


# ----------------------------------------------------------------------------- data
@dataclass
class Series:
    """All recorded calls for one renderer at one (num_envs, halt_poll) point."""

    arm: str
    per_run: list[list[float]] = field(default_factory=list)  # call order, one entry per pass
    wall_ms: float = 0.0

    @property
    def ordered(self) -> list[float]:
        return [v for run in self.per_run for v in run]

    @property
    def sorted(self) -> list[float]:
        return sorted(self.ordered)

    @property
    def runs(self) -> int:
        return len(self.per_run)


def _pct(sorted_vals: list[float], q: float) -> float:
    """Nearest-rank percentile, identical to the runner's (no interpolation)."""
    if not sorted_vals:
        return float("nan")
    idx = min(len(sorted_vals) - 1, int(round(q / 100.0 * (len(sorted_vals) - 1))))
    return sorted_vals[idx]


def load_samples(path: str, metric: str) -> tuple[dict[str, str], list[float]]:
    """Parse one ``--samples_out`` TSV into (header metadata, call durations [ms])."""
    meta: dict[str, str] = {}
    vals: list[float] = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                for tok in line[1:].split():
                    if "=" in tok:
                        key, val = tok.split("=", 1)
                        meta[key] = val
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3 or parts[0] == "metric":
                continue
            if parts[0] == metric:
                vals.append(float(parts[2]))
    return meta, vals


def group_samples(paths: list[str], metric: str) -> dict[tuple[str, str], dict[str, Series]]:
    """Pool sample files into ``{(num_envs, halt_poll): {arm: Series}}``."""
    groups: dict[tuple[str, str], dict[str, Series]] = {}
    for path in sorted(paths):
        meta, vals = load_samples(path, metric)
        if not vals:
            print(f"!!! {path}: no '{metric}' samples — skipped", file=sys.stderr)
            continue
        arm = meta.get("renderer", "?")
        key = (meta.get("num_envs", "?"), meta.get("halt_poll", "asis"))
        series = groups.setdefault(key, {}).setdefault(arm, Series(arm))
        series.per_run.append(vals)
        series.wall_ms += float(meta.get("wall_ms", 0.0) or 0.0)
    return groups


def _halt_sort_key(halt: str) -> tuple[int, float, str]:
    # "asis" (no sweep) sorts first; everything else numerically.
    return (0, 0.0, "") if halt == "asis" else (1, float(halt), halt)


def sorted_groups(groups: dict) -> list[tuple[str, str]]:
    return sorted(groups, key=lambda k: (float(k[0]) if k[0].isdigit() else math.inf, _halt_sort_key(k[1])))


def stats_of(series: Series) -> dict[str, float]:
    v = series.sorted
    total = sum(v)
    wall = series.wall_ms
    top1_n = max(1, round(len(v) * 0.01))
    return {
        "n": len(v),
        "runs": series.runs,
        "total_ms": total,
        "wall_ms": wall,
        "share_pct": (100.0 * total / wall) if wall > 0 else float("nan"),
        "mean": statistics.fmean(v),
        "median": statistics.median(v),
        "p25": _pct(v, 25),
        "p75": _pct(v, 75),
        "p95": _pct(v, 95),
        "p99": _pct(v, 99),
        "p999": _pct(v, 99.9),  # kept for parity with the RESULT line; not quoted in the report
        # Mean of the worst 1% of calls. Unlike p999 (a single order statistic:
        # the 3rd-worst call at the default 2000, i.e. noise) this averages n/100
        # calls, so it moves with the tail's WEIGHT rather than with one outlier.
        "top1_n": top1_n,
        "top1_mean": statistics.fmean(v[-top1_n:]),
        "min": v[0],
        "max": v[-1],
    }


def global_range(groups: dict) -> tuple[float, float]:
    """One log y-range for every chart in the run, so the SVGs are comparable."""
    lo, hi = math.inf, -math.inf
    for arms in groups.values():
        for series in arms.values():
            v = series.sorted
            lo, hi = min(lo, v[0]), max(hi, v[-1])
    if not math.isfinite(lo):
        return 1.0, 10.0
    return max(lo * 0.95, 1e-3), hi * 1.05


# ----------------------------------------------------------------------------- scales
def _nines(p_pct: float) -> float:
    """ "Nines" axis position of a percentile: p99.9 sits three units out."""
    return -math.log10(max(1.0 - p_pct / 100.0, 1e-12))


def _log_frac(v: float, lo: float, hi: float) -> float:
    return (math.log10(max(v, 1e-9)) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))


# Coarse to fine: a wide range gets 1/2/5-per-decade ticks, a narrow one (a
# well-behaved arm's whole distribution can span less than 2x) still gets a
# labelled grid instead of a single lonely "3".
_TICK_STEPS = (
    (1.0, 2.0, 5.0),
    (1.0, 1.5, 2.0, 3.0, 5.0, 7.0),
    (1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0),
    tuple(1.0 + 0.25 * i for i in range(36)),
    tuple(1.0 + 0.1 * i for i in range(90)),
)


def _nice_ticks(lo: float, hi: float, want: int = 6) -> list[float]:
    """Human-readable tick values inside a log range, densest set that stays readable."""
    decades = range(int(math.floor(math.log10(lo))), int(math.ceil(math.log10(hi))) + 1)
    inside: list[float] = []
    for steps in _TICK_STEPS:
        cand = [st * 10.0**d for d in decades for st in steps]
        inside = sorted({round(c, 6) for c in cand if lo <= c <= hi})
        if len(inside) >= min(4, want):
            break
    while len(inside) > want:
        inside = inside[::2]
    return inside


def _shares_the_loop(metric: str) -> bool:
    """Is a "% of the measured loop" meaningful for this metric?

    Yes for a per-substep call like ``write_data_to_sim``; no for ``full_frame``,
    which *is* the loop and would trivially report ~100%.
    """
    return metric != "full_frame"


def _ratio(a: float, b: float) -> str:
    return f"{a / b:.2f}x" if b > 0 else "-"


def text_report(groups: dict, metric: str, lo: float, hi: float) -> list[str]:
    """The terminal report: pooled stats per (num_envs, guest_halt_poll_ns).

    Tables only — the shape of the distribution lives in the SVGs, which show the
    whole tail instead of asking a scalar quantile to stand in for it.
    """
    out = [
        "",
        f"=========== {metric}: per guest_halt_poll_ns ===========",
        "  pooled over all passes; totals compare only at equal --num_steps (calls scale with it)",
    ]
    for envs, halt in sorted_groups(groups):
        arms = groups[(envs, halt)]
        out += ["", f"  ---------- envs={envs}  guest_halt_poll_ns={halt} ----------", ""]
        share_col = f"{'of loop':>8} " if _shares_the_loop(metric) else ""
        out.append(
            f"  {'renderer':<8} {'runs':>4} {'calls':>6} {'total_s':>9} {share_col}"
            f"{'mean':>8} {'median':>8} {'p95':>8} {'p99':>8} {'worst1%':>8} {'max':>8}"
        )
        st = {}
        for arm in ARMS:
            if arm not in arms:
                continue
            s = st[arm] = stats_of(arms[arm])
            share = f"{s['share_pct']:>7.1f}% " if _shares_the_loop(metric) else ""
            out.append(
                f"  {arm:<8} {s['runs']:>4} {s['n']:>6} {s['total_ms'] / 1e3:>9.3f} {share}"
                f"{s['mean']:>8.4f} {s['median']:>8.4f} {s['p95']:>8.4f} {s['p99']:>8.4f}"
                f" {s['top1_mean']:>8.4f} {s['max']:>8.4f}"
            )
        if len(st) == 2:
            n, o = st["newton"], st["ovrtx"]
            out.append(
                f"  -> ovrtx/newton: {_ratio(o['total_ms'], n['total_ms'])} total, "
                f"{_ratio(o['median'], n['median'])} median, {_ratio(o['p95'], n['p95'])} p95, "
                f"{_ratio(o['p99'], n['p99'])} p99, {_ratio(o['top1_mean'], n['top1_mean'])} worst1%"
            )
            out.append(
                f"     total: newton {n['total_ms'] / 1e3:.2f}s vs ovrtx {o['total_ms'] / 1e3:.2f}s"
                f"  (+{(o['total_ms'] - n['total_ms']) / 1e3:.2f}s over {n['n']} calls,"
                f" {(o['total_ms'] - n['total_ms']) / max(n['n'], 1):.3f} ms/call)"
            )
        out += _tail_caveats(st)
    halts = {halt for _, halt in groups}
    if len(halts) > 1:
        out += ["", "  ---------- halt-poll sweep: totals and tail ----------", ""]
        out.append(
            f"  {'halt_poll_ns':>12} {'newton_s':>9} {'ovrtx_s':>9} {'total':>7} {'median':>7}"
            f" {'worst1%':>8} {'ovrtx_worst1%':>14}"
        )
        for envs, halt in sorted_groups(groups):
            arms = groups[(envs, halt)]
            st = {arm: stats_of(series) for arm, series in arms.items()}
            n, o = st.get("newton"), st.get("ovrtx")
            out.append(
                f"  {halt:>12} {n['total_ms'] / 1e3 if n else float('nan'):>9.3f}"
                f" {o['total_ms'] / 1e3 if o else float('nan'):>9.3f}"
                f" {_ratio(o['total_ms'], n['total_ms']) if n and o else '-':>7}"
                f" {_ratio(o['median'], n['median']) if n and o else '-':>7}"
                f" {_ratio(o['top1_mean'], n['top1_mean']) if n and o else '-':>8}"
                f" {o['top1_mean'] if o else float('nan'):>14.4f}"
            )
        out.append("")
        out.append("  Halt values run in a fixed order within a pass (reversed on even passes):")
        out.append("  at --repeats 1 a monotonic trend here may just be machine drift.")
    out.append("")
    out.append("  worst1% = mean of the slowest 1% of calls: the tail number to quote. The full")
    out.append("  tail shape is in the SVG percentile curve, which needs no quantile at all.")
    out.append(f"  All charts share one log y-range ({lo:.2f}..{hi:.2f} ms) — SVGs are comparable.")
    out.append("=" * 62)
    return out


def _tail_caveats(st: dict[str, dict[str, float]]) -> list[str]:
    """Say when the sample count is too small for the tail columns to mean anything."""
    out = []
    thin_p99 = [f"{arm} n={s['n']}" for arm, s in st.items() if s["n"] < _MIN_CALLS["p99"]]
    if thin_p99:
        out.append(
            f"     !!! p99 is nearest-rank over {' / '.join(thin_p99)} — it needs >= {_MIN_CALLS['p99']}"
            " calls to differ from max; raise --num_steps"
        )
    thin_top = [f"{arm} {int(s['top1_n'])} call(s)" for arm, s in st.items() if s["top1_n"] < 10]
    if thin_top:
        out.append(
            f"     !!! worst1% averages only {' / '.join(thin_top)} — it is an outlier, not a tail;"
            " raise --num_steps (1000 steps ~= 2000 calls -> 20 calls averaged)"
        )
    return out


# ----------------------------------------------------------------------------- svg
# Chart chrome, ink and the two categorical slots, light and dark. The dark
# column is the same two hues re-stepped for the dark surface (a selected dark
# mode, not an automatic flip); a viewer's OS setting picks between them.
# Colours ride on SVG **presentation attributes**, not CSS variables: Inkscape's
# CSS parser (libcroco) rejects custom properties and `var()` outright, which
# dropped every fill and rendered the whole chart black. Presentation attributes
# need no CSS engine at all, so any converter (Inkscape, rsvg, browsers) gets the
# light rendering right; the CSS below is a pure enhancement — it overrides those
# attributes only where a viewer reports a dark colour scheme.
_PAINT = {
    "bg": 'fill="#fcfcfb"',
    "ttl": 'font-size="19" font-weight="600" fill="#0b0b0b"',
    "sub": 'font-size="12.5" fill="#52514e"',
    "cap": 'font-size="12.5" fill="#52514e"',
    "lbl": 'font-size="13" font-weight="600" fill="#0b0b0b"',
    "tick": 'font-size="11" fill="#898781"',
    "foot": 'font-size="11" fill="#898781"',
    "val": 'font-size="21" font-weight="600" fill="#0b0b0b"',
    "grid": 'fill="none" stroke="#e1e0d9" stroke-width="1"',
    "axis": 'fill="none" stroke="#c3c2b7" stroke-width="1"',
    "ring": 'fill="none" stroke="#0b0b0b" stroke-opacity="0.10" stroke-width="1"',
    "thin": 'fill="#e1e0d9" opacity="0.55"',
    "l1": 'fill="none" stroke="#2a78d6" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"',
    "l2": 'fill="none" stroke="#eb6834" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"',
    "f1": 'fill="#2a78d6"',
    "f2": 'fill="#eb6834"',
    "band1": 'fill="#2a78d6" opacity="0.16"',
    "band2": 'fill="#eb6834" opacity="0.16"',
    "box1": 'fill="#2a78d6" fill-opacity="0.22" stroke="#2a78d6" stroke-width="2"',
    "box2": 'fill="#eb6834" fill-opacity="0.22" stroke="#eb6834" stroke-width="2"',
    "gap": 'fill="none" stroke="#fcfcfb" stroke-width="4" stroke-linecap="butt"',
}

# The same eight hues re-stepped for the dark surface (a selected dark mode, not
# an automatic flip). Plain class selectors and literal hexes only — no custom
# properties — so a parser that does read this block cannot trip over it.
_DARK_CSS = """
  @media (prefers-color-scheme: dark) {
    .bg { fill: #1a1a19; }
    .ttl, .lbl, .val { fill: #ffffff; }
    .sub, .cap { fill: #c3c2b7; }
    .grid { stroke: #2c2c2a; }
    .axis { stroke: #383835; }
    .ring { stroke: #ffffff; stroke-opacity: 0.12; }
    .thin { fill: #2c2c2a; }
    .l1 { stroke: #3987e5; }
    .l2 { stroke: #d95926; }
    .f1, .band1 { fill: #3987e5; }
    .f2, .band2 { fill: #d95926; }
    .box1 { fill: #3987e5; stroke: #3987e5; }
    .box2 { fill: #d95926; stroke: #d95926; }
    .gap { stroke: #1a1a19; }
  }
"""

_FONT = "system-ui, -apple-system, \'Segoe UI\', sans-serif"


def _paint(svg: str) -> str:
    """Give every ``class="x"`` its light-mode presentation attributes."""
    return re.sub(
        r'class="([a-z0-9]+)"',
        lambda m: f'class="{m.group(1)}" {_PAINT.get(m.group(1), "")}'.rstrip(),
        svg,
    )


def _esc(text: str) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _svg_doc(width: int, height: int, body: str, label: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}"'
        f' height="{height}" font-family="{_FONT}" role="img" aria-label="{_esc(label)}">\n'
        f"<style>{_DARK_CSS}</style>\n"
        + _paint(f'<rect class="bg" x="0" y="0" width="{width}" height="{height}"/>\n{body}')
        + "\n</svg>\n"
    )


class _LogY:
    """Shared log y-scale: one instance drives every panel of every chart."""

    def __init__(self, lo: float, hi: float, top: float, bottom: float):
        self.lo, self.hi, self.top, self.bottom = lo, hi, top, bottom

    def px(self, v: float) -> float:
        return self.bottom - _log_frac(v, self.lo, self.hi) * (self.bottom - self.top)

    def ticks(self) -> list[float]:
        return _nice_ticks(self.lo, self.hi)


def _text(x: float, y: float, text: str, cls: str = "tick", anchor: str = "start") -> str:
    return f'<text class="{cls}" x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}">{_esc(text)}</text>'


def _grid_y(x0: float, x1: float, scale: _LogY, labels: bool = True) -> str:
    out = []
    for v in scale.ticks():
        y = scale.px(v)
        out.append(f'<line class="grid" x1="{x0:.1f}" y1="{y:.1f}" x2="{x1:.1f}" y2="{y:.1f}"/>')
        if labels:
            out.append(_text(x0 - 8, y + 4, f"{v:g}", "tick", "end"))
    return "\n".join(out)


def _legend(x: float, y: float, arms: dict[str, Series], anchor: str = "start") -> str:
    """Identity is never colour-alone: every chart with two arms carries this."""
    present = [a for a in ARMS if a in arms]
    widths = [34 + 7.2 * len(a) for a in present]
    x0 = x - sum(w + 18 for w in widths) + 18 if anchor == "end" else x
    out, dx = [], 0.0
    for arm, width in zip(present, widths):
        out.append(f'<rect class="f{_SLOT[arm]}" x="{x0 + dx:.1f}" y="{y - 9:.1f}" width="14" height="4" rx="2"/>')
        out.append(_text(x0 + dx + 20, y - 1, arm, "cap"))
        dx += width + 18
    return "\n".join(out)


def _svg_percentile(
    x0: float, y0: float, w: float, h: float, arms: dict[str, Series], scale: _LogY, compact: bool = False
) -> str:
    """Sorted duration vs a nines percentile axis — the tail gets half the width."""
    n_max = max(len(s.sorted) for s in arms.values())
    n_min = min(len(s.sorted) for s in arms.values())
    t_max = math.log10(n_max)
    out = []

    def xpx(t: float) -> float:
        return x0 + (t / t_max) * w

    t_thin = _nines(100.0 * (1.0 - _THIN_TAIL / n_min))
    if 0 < t_thin < t_max:
        out.append(
            f'<rect class="thin" x="{xpx(t_thin):.1f}" y="{y0:.1f}" width="{w - (xpx(t_thin) - x0):.1f}"'
            f' height="{h:.1f}"><title>fewer than {_THIN_TAIL} calls remain out here:'
            f" shape is individual outliers, not a distribution</title></rect>"
        )
        if not compact:
            out.append(_text(xpx(t_thin) + 6, y0 + 14, f"<{_THIN_TAIL} calls", "foot"))
    out.append(_grid_y(x0, x0 + w, scale))
    marks = (
        ((50, "p50"), (99, "p99"))
        if compact
        else ((50, "p50"), (90, "p90"), (99, "p99"), (99.9, "p99.9"), (99.99, "p99.99"))
    )
    for p, lab in marks:
        if _nines(p) > t_max:
            continue
        x = xpx(_nines(p))
        out.append(f'<line class="grid" x1="{x:.1f}" y1="{y0:.1f}" x2="{x:.1f}" y2="{y0 + h:.1f}"/>')
        out.append(_text(x, y0 + h + 18, lab, "tick", "middle"))
    out.append(_text(x0 + w, y0 + h + 18, "max", "tick", "middle"))
    out.append(f'<line class="axis" x1="{x0:.1f}" y1="{y0 + h:.1f}" x2="{x0 + w:.1f}" y2="{y0 + h:.1f}"/>')

    ends = []
    for arm in ARMS:
        series = arms.get(arm)
        if series is None:
            continue
        vals = series.sorted
        n = len(vals)
        pts = []
        for i in range(241):
            t = t_max * i / 240
            if t > math.log10(n):
                break
            idx = min(n - 1, int(round((1.0 - 10.0**-t) * (n - 1))))
            pts.append(f"{xpx(t):.1f},{scale.px(vals[idx]):.1f}")
        out.append(f'<polyline class="l{_SLOT[arm]}" points="{" ".join(pts)}"/>')
        ends.append((arm, xpx(min(math.log10(n), t_max)), scale.px(vals[-1])))
    # Direct labels at the curve ends; nudge apart if the two arms finish together.
    ends.sort(key=lambda e: e[1])
    if len(ends) == 2 and abs(ends[0][2] - ends[1][2]) < 15:
        hi_first = ends[0][2] <= ends[1][2]
        ends[0] = (ends[0][0], ends[0][1], ends[0][2] - (0 if hi_first else 8))
        ends[1] = (ends[1][0], ends[1][1], ends[1][2] + (8 if hi_first else 0))
    for arm, x, y in ends:
        out.append(f'<circle class="f{_SLOT[arm]}" cx="{x:.1f}" cy="{y:.1f}" r="4.5"/>')
        out.append(_text(x - 8, y - 8, arm, "cap", "end"))
    return "\n".join(out)


def _svg_box(x0: float, y0: float, w: float, h: float, arms: dict[str, Series], scale: _LogY) -> str:
    """Vertical box plots on the SAME y-scale as the percentile panel beside them."""
    out = [_grid_y(x0, x0 + w, scale, labels=False)]
    present = [a for a in ARMS if a in arms]
    slot_w = w / max(len(present), 1)
    for i, arm in enumerate(present):
        st = stats_of(arms[arm])
        cx = x0 + slot_w * (i + 0.5)
        bw = min(56.0, slot_w * 0.5)
        top, bot = scale.px(st["p75"]), scale.px(st["p25"])
        out.append(
            f'<line class="axis" x1="{cx:.1f}" y1="{scale.px(st["min"]):.1f}" x2="{cx:.1f}"'
            f' y2="{scale.px(st["max"]):.1f}"/>'
        )
        out.append(
            f'<rect class="box{_SLOT[arm]}" x="{cx - bw / 2:.1f}" y="{top:.1f}" width="{bw:.1f}"'
            f' height="{max(bot - top, 2):.1f}" rx="4"><title>{arm}: p25 {st["p25"]:.2f} / median'
            f" {st['median']:.2f} / p75 {st['p75']:.2f} ms</title></rect>"
        )
        med = scale.px(st["median"])
        out.append(f'<line class="gap" x1="{cx - bw / 2:.1f}" y1="{med:.1f}" x2="{cx + bw / 2:.1f}" y2="{med:.1f}"/>')
        out.append(
            f'<line class="l{_SLOT[arm]}" x1="{cx - bw / 2:.1f}" y1="{med:.1f}" x2="{cx + bw / 2:.1f}" y2="{med:.1f}"/>'
        )
        ytop1, ymax = scale.px(st["top1_mean"]), scale.px(st["max"])
        # The two often coincide on a short run; split the callouts rather than
        # overprint them.
        dytop1, dymax = (8.0, -6.0) if abs(ytop1 - ymax) < 14 else (4.0, 4.0)
        for val, label, y, dy in (
            (st["top1_mean"], "worst 1%", ytop1, dytop1),
            (st["max"], "max", ymax, dymax),
        ):
            out.append(f'<circle class="gap" cx="{cx:.1f}" cy="{y:.1f}" r="5"/>')
            out.append(
                f'<circle class="f{_SLOT[arm]}" cx="{cx:.1f}" cy="{y:.1f}" r="4"><title>{arm} {label}:'
                f" {val:.2f} ms</title></circle>"
            )
            out.append(_text(cx + 10, y + dy, label, "foot"))
        out.append(_text(cx, y0 + h + 18, arm, "cap", "middle"))
    return "\n".join(out)


def _svg_timeline(x0: float, y0: float, w: float, h: float, arms: dict[str, Series], scale: _LogY) -> str:
    """Call order, per-bucket min..max band + median line: spikes vs drift."""
    out = [_grid_y(x0, x0 + w, scale)]
    buckets = 240
    n_calls = max(len(s.ordered) for s in arms.values())
    for arm in ARMS:
        series = arms.get(arm)
        if series is None:
            continue
        vals = series.ordered
        n = len(vals)
        lo_pts, hi_pts, med_pts = [], [], []
        for b in range(buckets):
            chunk = vals[n * b // buckets : max(n * (b + 1) // buckets, n * b // buckets + 1)]
            if not chunk:
                continue
            x = x0 + w * (b + 0.5) / buckets
            lo_pts.append((x, scale.px(min(chunk))))
            hi_pts.append((x, scale.px(max(chunk))))
            med_pts.append(f"{x:.1f},{scale.px(statistics.median(chunk)):.1f}")
        band = " ".join(f"{x:.1f},{y:.1f}" for x, y in hi_pts + lo_pts[::-1])
        out.append(f'<polygon class="band{_SLOT[arm]}" points="{band}"/>')
        out.append(f'<polyline class="l{_SLOT[arm]}" points="{" ".join(med_pts)}"/>')
    out.append(f'<line class="axis" x1="{x0:.1f}" y1="{y0 + h:.1f}" x2="{x0 + w:.1f}" y2="{y0 + h:.1f}"/>')
    out.append(_text(x0, y0 + h + 18, "call 0", "tick"))
    out.append(_text(x0 + w, y0 + h + 18, f"call {n_calls}", "tick", "end"))
    return "\n".join(out)


def _svg_tiles(x0: float, y0: float, w: float, arms: dict[str, Series], metric: str) -> str:
    """Headline numbers the percentile curve cannot show: totals and the ratios."""
    st = {arm: stats_of(s) for arm, s in arms.items()}
    tiles: list[tuple[str, str, str]] = []
    for arm in ARMS:
        if arm not in st:
            continue
        s = st[arm]
        # Never quote a pooled multi-pass wall as if it were one loop.
        loop = (
            f"{s['wall_ms'] / 1e3:.1f}s loop"
            if s["runs"] == 1
            else f"{s['wall_ms'] / s['runs'] / 1e3:.1f}s loop x{s['runs']} passes"
        )
        caption = f"{s['share_pct']:.1f}% of {loop}" if _shares_the_loop(metric) else loop
        tiles.append((f"{arm}: total in {metric}", f"{s['total_ms'] / 1e3:.2f} s", f"{caption} · {s['n']} calls"))
    if len(st) == 2:
        n, o = st["newton"], st["ovrtx"]
        tiles.append(
            (
                "ovrtx / newton total",
                _ratio(o["total_ms"], n["total_ms"]),
                f"+{(o['total_ms'] - n['total_ms']) / 1e3:.2f} s over the same call count",
            )
        )
        tiles.append(
            (
                "ovrtx / newton tail",
                _ratio(o["top1_mean"], n["top1_mean"]) + " worst 1%",
                f"median {_ratio(o['median'], n['median'])} · p99 {_ratio(o['p99'], n['p99'])}",
            )
        )
    out, tw = [], (w - 16 * (len(tiles) - 1)) / max(len(tiles), 1)
    for i, (label, value, caption) in enumerate(tiles):
        x = x0 + i * (tw + 16)
        out.append(f'<rect class="ring" x="{x:.1f}" y="{y0:.1f}" width="{tw:.1f}" height="78" rx="8"/>')
        out.append(_text(x + 14, y0 + 22, label, "foot"))
        out.append(_text(x + 14, y0 + 48, value, "val"))
        out.append(_text(x + 14, y0 + 67, caption, "foot"))
    return "\n".join(out)


def group_svg(envs: str, halt: str, arms: dict[str, Series], scale_range: tuple[float, float], metric: str) -> str:
    """One standalone SVG for one guest_halt_poll_ns value: totals, tail, order."""
    width, height = 1120, 830
    lo, hi = scale_range
    top = _LogY(lo, hi, 230, 508)
    bottom = _LogY(lo, hi, 592, 762)
    calls = " / ".join(f"{arm} {len(arms[arm].sorted)}" for arm in ARMS if arm in arms)
    passes = max((s.runs for s in arms.values()), default=0)
    halt_txt = "unchanged" if halt == "asis" else f"{int(halt):,} ns".replace(",", " ")
    body = [
        _text(32, 44, f"{metric} — ovrtx vs newton", "ttl"),
        _text(32, 68, f"{envs} envs · guest_halt_poll_ns {halt_txt} · calls: {calls} · {passes} pass(es)", "sub"),
        _svg_tiles(32, 92, width - 64, arms, metric),
        _text(32, 212, "sorted call duration vs percentile", "lbl"),
        _legend(768, 212, arms, anchor="end"),
        _text(32, 224, "ms (log)", "tick"),
        _svg_percentile(92, 230, 676, 278, arms, top),
        _text(828, 212, "distribution", "lbl"),
        _svg_box(828, 230, 228, 278, arms, top),
        _text(32, 574, "call order — per-bucket min/max band with the bucket median", "lbl"),
        _svg_timeline(92, 592, 964, 170, arms, bottom),
        _text(
            32,
            800,
            f"Nines x-axis (-log10(1-p)); both panels share one log y-range "
            f"({lo:.2f}–{hi:.2f} ms) held fixed across every chart in this run.",
            "foot",
        ),
        _text(
            32,
            814,
            f"The curve is the tail: no scalar quantile stands in for it. p99 needs >= "
            f"{_MIN_CALLS['p99']} calls to differ from max; the shaded band is where <{_THIN_TAIL} remain.",
            "foot",
        ),
    ]
    return _svg_doc(width, height, "\n".join(body), f"{metric} percentile, distribution and call-order charts")


def summary_svg(groups: dict, scale_range: tuple[float, float], metric: str) -> str:
    """Small multiples: every halt-poll value's percentile curve on identical axes."""
    lo, hi = scale_range
    keys = sorted_groups(groups)
    cols = min(3, max(len(keys), 1))
    rows = (len(keys) + cols - 1) // cols
    pw, ph, gx, gy = 300, 190, 64, 96
    width = 64 + cols * pw + (cols - 1) * gx + 32
    height = 148 + rows * ph + (rows - 1) * gy + 72
    body = [
        _text(32, 44, f"{metric} — tail by guest_halt_poll_ns", "ttl"),
        _text(32, 68, "same nines x-axis and log y-range in every panel, so panels compare directly", "sub"),
        _legend(width - 32, 68, {a: s for arms in groups.values() for a, s in arms.items()}, anchor="end"),
    ]
    for i, key in enumerate(keys):
        x0 = 64 + (i % cols) * (pw + gx)
        y0 = 148 + (i // cols) * (ph + gy)
        envs, halt = key
        st = {arm: stats_of(series) for arm, series in groups[key].items()}
        body.append(_text(x0 - 32, y0 - 30, f"halt_poll {halt} · {envs} envs", "lbl"))
        if len(st) == 2:
            n, o = st["newton"], st["ovrtx"]
            body.append(
                _text(
                    x0 - 32,
                    y0 - 12,
                    f"ovrtx/newton {_ratio(o['total_ms'], n['total_ms'])} total ·"
                    f" {_ratio(o['median'], n['median'])} median ·"
                    f" {_ratio(o['top1_mean'], n['top1_mean'])} worst1%",
                    "foot",
                )
            )
        body.append(_svg_percentile(x0, y0, pw, ph, groups[key], _LogY(lo, hi, y0, y0 + ph), compact=True))
    return _svg_doc(width, height, "\n".join(body), f"{metric} percentile curves per guest_halt_poll_ns")


# ----------------------------------------------------------------------------- cli
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--samples", nargs="+", default=[], help="Sample TSVs from --samples_out.")
    parser.add_argument("--samples_dir", default="", help="Directory to glob '*.tsv' from, added to --samples.")
    parser.add_argument("--metric", default="write_data_to_sim", help="Metric column to chart.")
    parser.add_argument("--svg_dir", default="", help="Where to write the SVGs (default: <samples>/../charts).")
    parser.add_argument("--no_svg", action="store_true", help="Terminal report only.")
    parser.add_argument("--text_out", default="", help="Also write the terminal report to this file.")
    args = parser.parse_args(argv)

    paths = list(args.samples)
    if args.samples_dir:
        paths += sorted(glob.glob(os.path.join(args.samples_dir, "*.tsv")))
    if not paths:
        parser.error("no sample files: pass --samples FILE... or --samples_dir DIR")
    groups = group_samples(paths, args.metric)
    if not groups:
        print(f"!!! no '{args.metric}' samples in {len(paths)} file(s)", file=sys.stderr)
        return 2

    scale_range = global_range(groups)
    lines = text_report(groups, args.metric, scale_range[0], scale_range[1])
    report = "\n".join(lines)
    print(report)
    if args.text_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.text_out)) or ".", exist_ok=True)
        with open(args.text_out, "w") as fh:
            fh.write(report + "\n")

    if args.no_svg:
        return 0
    svg_dir = args.svg_dir or os.path.join(os.path.dirname(os.path.abspath(paths[0])), "charts")
    os.makedirs(svg_dir, exist_ok=True)
    written = []
    for envs, halt in sorted_groups(groups):
        path = os.path.join(svg_dir, f"{args.metric}_n{envs}_hp{halt}.svg")
        with open(path, "w") as fh:
            fh.write(group_svg(envs, halt, groups[(envs, halt)], scale_range, args.metric))
        written.append(path)
    if len(groups) > 1:
        path = os.path.join(svg_dir, f"{args.metric}_summary.svg")
        with open(path, "w") as fh:
            fh.write(summary_svg(groups, scale_range, args.metric))
        written.append(path)
    print("  charts: " + "\n          ".join(written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
