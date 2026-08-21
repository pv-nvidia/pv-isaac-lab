# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""NVBug 6431561: is ``InteractiveScene.write_data_to_sim`` slower under the OVRTX
renderer than under Newton-Warp, on an otherwise identical scene?

This is the **kitless** path: Newton physics (`newton_mjwarp`) + a kitless
renderer (`ovrtx` or `newton_renderer`). Isaac Sim / Kit is never launched —
`launch_simulation` does nothing for kitless backends. So there is no isaacsim
dependency here at all.

Default metric is ``write_data_to_sim``; ``--full`` also times the whole
``DirectRLEnv.step`` frame. `--ovrtx wheel|local` (handled by run_render_bench.sh)
selects the ovrtx runtime for the `ovrtx` renderer.

Ticket CLI (the `--enable_cameras/--headless` flags are accepted no-ops):

    python render_bench_runner.py --renderer newton --num_envs 1024 --enable_cameras --headless
    python render_bench_runner.py --renderer ovrtx  --num_envs 1024 --enable_cameras --headless
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

# launch_simulation is backend-aware: it launches Kit only for Kit-based backends
# and does nothing for kitless ones (Newton physics + ovrtx/newton renderer).
# pv2-isaac-lab may not re-export these from isaaclab.app, so fall back to the submodule.
try:
    from isaaclab.app import add_launcher_args, launch_simulation
except ImportError:
    from isaaclab.app.sim_launcher import add_launcher_args, launch_simulation

from isaaclab_tasks.utils import setup_preset_cli

_RENDERER_PRESET = {"ovrtx": "ovrtx", "newton": "newton_renderer"}
_PHYSICS_PRESET = "newton_mjwarp"  # always Newton -> keeps the run kitless (no Isaac Sim/Kit)

parser = argparse.ArgumentParser(description="NVBug 6431561: write_data_to_sim timing per renderer (kitless).")
parser.add_argument("--renderer", choices=sorted(_RENDERER_PRESET), required=True)
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--num_steps", type=int, default=30, help="Measured steps.")
parser.add_argument("--warmup", type=int, default=12, help="Warmup steps (>=10; OVRTX RT pipeline builds slowly).")
parser.add_argument("--full", action="store_true", help="Also time full-frame DirectRLEnv.step (default: write only).")
parser.add_argument(
    "--data_preset", default="rgb", help="Data-type preset token (rgb, simple_shading_diffuse_mdl, ...)."
)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--no_cupti",
    action="store_true",
    help="Disable OVRTX's CUPTI-based GPU memory tracking (carb memoryScope trackAddress).",
)
parser.add_argument(
    "--profile",
    action="store_true",
    help=(
        "Emit NVTX ranges around each write_data_to_sim and cudaProfiler start/stop around the"
        " measured region, for nsys --capture-range=cudaProfilerApi (see run_render_bench.sh --nsys)."
    ),
)
parser.add_argument(
    "--step_sleep",
    type=float,
    default=0.0,
    help=(
        "Sleep this many seconds between measured steps (outside the timed write). Lets renderer"
        " GPU/background activity drain before the next write: recovery implicates work concurrent"
        " with the write window (incl. GPU-power-driven CPU throttling with its time constants)."
    ),
)
parser.add_argument(
    "--deprioritize_workers",
    action="store_true",
    help=(
        "After warmup, renice every thread except the sim thread to nice 19. Starves renderer"
        " threads of CPU while the sim thread runs: recovery implicates CPU competition; no"
        " recovery exonerates it (e.g. GPU-side or platform-level effects)."
    ),
)
parser.add_argument(
    "--gil_probe",
    action="store_true",
    help=(
        "Raise sys.setswitchinterval to 1.0s during the measured loop. If the ovrtx write"
        " recovers toward newton speed, the slowdown is GIL contention (another thread —"
        " Python-level or a C thread re-entering via callbacks — repeatedly taking the GIL"
        " from the sim thread). Restores the default interval afterwards."
    ),
)
parser.add_argument(
    "--pin_sim",
    action="store_true",
    help=(
        "Affinity experiment: confine every thread spawned during setup (incl. the renderer's"
        " workers) to CPUs 2..N-1, then pin the sim thread alone to CPU 0 before measuring,"
        " leaving CPU 1 (its likely SMT sibling on paired-vCPU hosts) idle. If the ovrtx write"
        " recovers to newton speed, the ~1.7x CPU slowdown is sibling/cache contention from"
        " renderer threads; if it stays slow, it is a global effect (host all-core frequency)."
    ),
)
parser.add_argument(
    "--samples_out",
    default="",
    help=(
        "Write every recorded call duration to this TSV (metric/call/ms + a self-describing"
        " header) so render_bench_report.py can chart the distribution and the tail."
    ),
)
parser.add_argument(
    "--run_meta",
    default="",
    help=(
        "Comma-separated key=value pairs copied verbatim into the --samples_out header"
        " (run_render_bench.sh passes halt_poll/pass/arm-tag context this way)."
    ),
)
add_launcher_args(parser)
# Accept the ticket's --enable_cameras/--headless even if this Isaac Lab no longer
# registers them (kitless renderers are headless; camera scenes auto-enable cameras).
for _flag in ("--enable_cameras", "--headless"):
    if _flag not in parser._option_string_actions:
        parser.add_argument(_flag, action="store_true", help="Accepted for ticket-CLI compatibility (no-op).")

args_cli, hydra_args = setup_preset_cli(parser)

if args_cli.no_cupti:
    # Kitless runs never see kit_args or --/ flags (RendererWrapper passes argv=nullptr to
    # carb::startupFramework); the only channels into the ovrtx runtime are OVRTX_-prefixed
    # env vars ('_' -> '/' forms the settings path, carb EnvironmentVariableParser) and the
    # wheel's ovrtx.config.json. This targets the setting that arms carb.cudainterop's CUPTI
    # hooks (GPU memory tracking v2 trackAddress, default ON), which subscribe CUPTI activity
    # tracing to every CUDA runtime/driver call process-wide once the renderer is constructed.
    # Both leading-slash spellings are set because carb's env-override examples use no leading
    # slash while Kit-style paths carry one; the runner VERIFIES the outcome after the run via
    # the CUPTI subscriber-slot probe ("cupti tracking:" PT_MARK) — trust that line, not this.
    _KEY = "plugins_carb.memorytracking.plugin_memoryScope_trackAddress_enabled"
    os.environ[f"OVRTX_{_KEY}"] = "false"
    os.environ[f"OVRTX__{_KEY}"] = "false"
    print("PT_MARK cupti gpu-memory tracking disable requested via OVRTX_ settings env vars", flush=True)

if args_cli.pin_sim:
    # Linux sched_setaffinity(0, ...) targets the CALLING THREAD; threads created
    # afterwards inherit it. Setting {2..N-1} here (before the renderer exists) makes
    # every worker thread land on CPUs >= 2; main() re-pins this thread to {0} before
    # the measured loop.
    _ncpu = os.cpu_count() or 4
    os.sched_setaffinity(0, set(range(2, _ncpu)))
    print(f"PT_MARK pin_sim: setup threads confined to CPUs 2-{_ncpu - 1}", flush=True)

# newton physics + chosen renderer + data type. Pinning newton_mjwarp guarantees
# kitless (the render-bench default renderer is the Kit RTX one, which would need Kit).
_preset = f"{_PHYSICS_PRESET},{_RENDERER_PRESET[args_cli.renderer]},{args_cli.data_preset}"
sys.argv = [sys.argv[0], f"presets={_preset}", *hydra_args]

# Kitless: do NOT launch AppLauncher. Import the env stack (import-safe without Kit).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gymnasium as gym  # noqa: E402
import render_bench_task  # noqa: F401  registers the gym id  # noqa: E402
import torch  # noqa: E402

from isaaclab.envs import DirectRLEnv  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402

from isaaclab_tasks.utils import resolve_task_config  # noqa: E402

TASK = "Repro-RenderBench-Franka-Cabinet-v0"

# --- timers: write_data_to_sim always; full-frame DirectRLEnv.step only with --full ----
_WRITE_MS: list[float] = []
_FRAME_MS: list[float] = []
_RECORD = {"on": False}
# Wall clock of the measured loop, so a total can be quoted as a SHARE of the
# frame budget instead of as a bare millisecond sum (which only scales with
# --num_steps and compares across runs at equal step counts).
_LOOP_WALL_MS = [0.0]

_orig_write = InteractiveScene.write_data_to_sim


def _timed_write(self):
    if not _RECORD["on"]:
        return _orig_write(self)
    if args_cli.profile:
        torch.cuda.nvtx.range_push("write_data_to_sim")
    t0 = time.perf_counter()
    out = _orig_write(self)
    _WRITE_MS.append((time.perf_counter() - t0) * 1e3)
    if args_cli.profile:
        torch.cuda.nvtx.range_pop()
    return out


InteractiveScene.write_data_to_sim = _timed_write

if args_cli.full:
    _orig_step = DirectRLEnv.step

    def _timed_step(self, action):
        if not _RECORD["on"]:
            return _orig_step(self, action)
        t0 = time.perf_counter()
        out = _orig_step(self, action)
        _FRAME_MS.append((time.perf_counter() - t0) * 1e3)
        return out

    DirectRLEnv.step = _timed_step


def _mark(msg: str) -> None:
    print(f"PT_MARK {msg}", flush=True)


def _sched_snapshot() -> dict:
    """Scheduling-pressure counters for the measured window.

    threads: process thread count (ovrtx residency adds ~200; newton ~none).
    nonvol: involuntary context switches of THIS thread (the one timing the
    writes) — each one is the scheduler preempting the sim mid-work, the
    mechanism by which renderer threads / host steal inflate a CPU-burst span.
    steal_ticks: CPU time the hypervisor took from the whole guest (0 on bare
    metal). Deltas across the measured loop attribute a slow run to scheduling
    pressure instead of leaving it a mystery.
    """
    snap = {
        "threads": 0,
        "nonvol": 0,
        "vol": 0,
        "minflt": 0,
        "majflt": 0,
        "steal_ticks": 0,
        "tlb_ipi": 0,
        "call_ipi": 0,
    }
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("Threads:"):
                snap["threads"] = int(line.split()[1])
                break
    with open("/proc/thread-self/status") as fh:
        for line in fh:
            if line.startswith("voluntary_ctxt_switches:"):
                snap["vol"] = int(line.split()[1])
            elif line.startswith("nonvoluntary_ctxt_switches:"):
                snap["nonvol"] = int(line.split()[1])
    # minflt/majflt of THIS thread: /proc/thread-self/stat fields 10/12
    # (1-indexed, counted after the parenthesized comm which may contain spaces).
    with open("/proc/thread-self/stat") as fh:
        stat = fh.read()
    fields = stat[stat.rfind(")") + 2 :].split()  # fields[0] is field 3 (state)
    snap["minflt"] = int(fields[7])
    snap["majflt"] = int(fields[9])
    with open("/proc/stat") as fh:
        parts = fh.readline().split()
        if len(parts) > 8:
            snap["steal_ticks"] = int(parts[8])
    # TLB-shootdown and function-call IPIs, summed over all CPUs (system-wide).
    # Renderer threads share the sim's address space, so their unmap/madvise
    # churn interrupts EVERY core running this process — including the sim
    # thread's — and each IPI costs ~10-20x more inside a KVM guest. This is
    # the in-process interference that separate-process CPU hogs cannot
    # reproduce and CPU pinning cannot avoid.
    with open("/proc/interrupts") as fh:
        for line in fh:
            parts = line.split()
            if parts and parts[0].rstrip(":") in ("TLB", "CAL"):
                total = sum(int(x) for x in parts[1:] if x.isdigit())
                snap["tlb_ipi" if parts[0].startswith("TLB") else "call_ipi"] = total
    return snap


def _cupti_subscriber_state() -> str:
    """Report whether a CUPTI subscriber is active in this process.

    CUPTI allows exactly one subscriber per library instance, so probing
    ``cuptiSubscribe`` on the libcupti this process has MAPPED (per
    /proc/self/maps — a different copy would have its own independent slot)
    gives a mechanism-proof answer: slot held means carb.cudainterop's GPU
    memory tracking (or another profiler) is live; slot free or library never
    loaded means it is not. Call only after measurement — a probe subscriber,
    even with a no-op callback, taxes every CUDA call while held.
    """
    import ctypes

    path = None
    with open("/proc/self/maps") as fh:
        for line in fh:
            if "libcupti" in line:
                path = line.split()[-1]
                break
    if path is None:
        return "inactive (libcupti not loaded in this process)"
    try:
        lib = ctypes.CDLL(path)
    except OSError as exc:
        return f"unknown (mapped {path} but CDLL failed: {exc})"
    cb = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p)(lambda *a: None)
    handle = ctypes.c_void_p()
    rc = lib.cuptiSubscribe(ctypes.byref(handle), cb, None)
    if rc == 0:
        lib.cuptiUnsubscribe(handle)
        return f"inactive (subscriber slot free; probed {path})"
    return f"ACTIVE (subscriber slot held, cuptiSubscribe rc={rc}; {path})"


# Nearest-rank sample counts below which a quantile is just max_ms. With
# idx = round(q/100 * (n-1)), p99 separates from max at n=52 and p99.9 at n=502.
_MIN_CALLS = {"p99": 52, "p999": 502}


def _pct(sorted_vals: list[float], q: float) -> float:
    # Nearest-rank, no interpolation (see _MIN_CALLS). Keep as-is: switching to
    # an interpolating percentile would move already-quoted p95 numbers.
    if not sorted_vals:
        return float("nan")
    idx = min(len(sorted_vals) - 1, int(round(q / 100.0 * (len(sorted_vals) - 1))))
    return sorted_vals[idx]


def _report(metric: str, vals: list[float]) -> None:
    v = sorted(vals)
    if not v:
        print(f"[render-bench] {metric}: no samples", flush=True)
        return
    # total_ms/share_pct come last so run_render_bench.sh can bolt on new fields
    # without disturbing the median..p999 sed capture that the report table parses.
    total = sum(v)
    wall = _LOOP_WALL_MS[0]
    # Share OF THE MEASURED LOOP: meaningful for a per-substep call like
    # write_data_to_sim, trivially ~100% for full_frame (which is the loop).
    share = (100.0 * total / wall) if wall > 0 else float("nan")
    print(
        f"RESULT metric={metric} renderer={args_cli.renderer} "
        f"ovrtx_source={os.environ.get('OVRTX_SOURCE', 'wheel')} num_envs={args_cli.num_envs} "
        f"calls={len(v)} median_ms={statistics.median(v):.4f} mean_ms={statistics.fmean(v):.4f} "
        f"p95_ms={_pct(v, 95):.4f} p99_ms={_pct(v, 99):.4f} p999_ms={_pct(v, 99.9):.4f} "
        f"min_ms={v[0]:.4f} max_ms={v[-1]:.4f} "
        f"total_ms={total:.2f} loop_wall_ms={wall:.2f} loop_share_pct={share:.2f}",
        flush=True,
    )
    # The sample count is calls, not steps: a step records one write per physics
    # substep, so only the runner knows whether the tail quantiles are real.
    thin = [name for name, need in _MIN_CALLS.items() if len(v) < need]
    if thin:
        print(
            f"PT_MARK WARNING: {metric}: {'/'.join(thin)} == max_ms with calls={len(v)}"
            f" (nearest-rank needs >= {_MIN_CALLS['p99']} calls for p99,"
            f" >= {_MIN_CALLS['p999']} for p999) — raise --num_steps to read the tail",
            flush=True,
        )


def _write_samples(path: str) -> None:
    """Dump every recorded call duration, with enough header to chart it later.

    One TSV row per call (``metric``/``call``/``ms``) plus a ``#`` header carrying
    the run's identity, so ``render_bench_report.py`` never has to re-derive
    renderer/env-count/halt-poll from a file name.
    """
    meta = {
        "renderer": args_cli.renderer,
        "ovrtx_source": os.environ.get("OVRTX_SOURCE", "wheel"),
        "num_envs": args_cli.num_envs,
        "num_steps": args_cli.num_steps,
        "warmup": args_cli.warmup,
        "no_cupti": int(args_cli.no_cupti),
        "pin_sim": int(args_cli.pin_sim),
        "wall_ms": f"{_LOOP_WALL_MS[0]:.2f}",
    }
    for pair in args_cli.run_meta.split(","):
        key, _, val = pair.partition("=")
        if key.strip():
            meta[key.strip()] = val.strip()
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as fh:
        fh.write("# render-bench samples v1\n")
        fh.write("# " + " ".join(f"{k}={v}" for k, v in meta.items()) + "\n")
        fh.write("metric\tcall\tms\n")
        for metric, vals in (("write_data_to_sim", _WRITE_MS), ("full_frame", _FRAME_MS)):
            for i, ms in enumerate(vals):
                fh.write(f"{metric}\t{i}\t{ms:.6f}\n")
    print(f"PT_MARK samples: {len(_WRITE_MS)} write + {len(_FRAME_MS)} frame calls -> {path}", flush=True)


def main(env_cfg) -> None:
    _mark("main: entered")
    env_cfg.scene.num_envs = args_cli.num_envs
    if getattr(args_cli, "device", None) is not None:
        env_cfg.sim.device = args_cli.device
    env_cfg.seed = args_cli.seed

    _mark("main: gym.make")
    env = gym.make(TASK, cfg=env_cfg, render_mode=None)
    _mark("main: env.reset")
    env.reset()
    _mark("main: reset done")
    num_envs = env.unwrapped.num_envs
    action_dim = env.unwrapped.single_action_space.shape[0]
    device = env.unwrapped.device

    def _one():
        env.step(2.0 * torch.rand(num_envs, action_dim, device=device) - 1.0)

    if args_cli.pin_sim:
        os.sched_setaffinity(0, {0})  # sim thread alone on CPU 0; CPU 1 left idle
        _mark(f"pin_sim: sim thread pinned to CPU 0 (affinity now {sorted(os.sched_getaffinity(0))})")

    _mark(f"main: warmup x{args_cli.warmup}")
    for _ in range(args_cli.warmup):
        _one()
    import threading as _threading

    _mark(f"python threads: {[t.name for t in _threading.enumerate()]}")

    if args_cli.deprioritize_workers:
        _me = _threading.get_native_id()
        _reniced = 0
        for _tid in os.listdir("/proc/self/task"):
            if int(_tid) != _me:
                try:
                    os.setpriority(os.PRIO_PROCESS, int(_tid), 19)
                    _reniced += 1
                except OSError:
                    pass
        _mark(f"deprioritize_workers: {_reniced} threads reniced to 19 (sim thread untouched)")

    _gil_interval0 = sys.getswitchinterval()
    if args_cli.gil_probe:
        sys.setswitchinterval(1.0)
        _mark("gil_probe: switch interval raised to 1.0s for the measured loop")

    _mark(f"main: measuring x{args_cli.num_steps}")
    sched0 = _sched_snapshot()
    _RECORD["on"] = True
    if args_cli.profile:
        # Opens the nsys --capture-range=cudaProfilerApi window: warmup (RT pipeline
        # build, module loads) stays out of the report; no-op without a profiler.
        torch.cuda.profiler.start()
    _loop_t0 = time.perf_counter()
    for i in range(args_cli.num_steps):
        if args_cli.step_sleep > 0.0:
            time.sleep(args_cli.step_sleep)  # outside the timed writes; lets GPU/background work drain
        if args_cli.profile:
            torch.cuda.nvtx.range_push(f"bench_step_{i}")
        _one()
        if args_cli.profile:
            torch.cuda.nvtx.range_pop()
    _LOOP_WALL_MS[0] = (time.perf_counter() - _loop_t0) * 1e3
    if args_cli.profile:
        torch.cuda.profiler.stop()
    _RECORD["on"] = False
    if args_cli.gil_probe:
        sys.setswitchinterval(_gil_interval0)
    _mark(f"main: measured done (write samples={len(_WRITE_MS)})")
    sched1 = _sched_snapshot()
    steal_ms = (sched1["steal_ticks"] - sched0["steal_ticks"]) * 1000 // max(1, os.sysconf("SC_CLK_TCK"))
    _mark(
        f"sched: threads={sched1['threads']}"
        f" write_thread_preemptions={sched1['nonvol'] - sched0['nonvol']}"
        f" write_thread_vol_switches={sched1['vol'] - sched0['vol']}"
        f" write_thread_minflt={sched1['minflt'] - sched0['minflt']}"
        f" write_thread_majflt={sched1['majflt'] - sched0['majflt']}"
        f" guest_steal_ms={steal_ms}"
        f" tlb_shootdown_ipis={sched1['tlb_ipi'] - sched0['tlb_ipi']}"
        f" call_ipis={sched1['call_ipi'] - sched0['call_ipi']}"
        f" (measured window, IPIs system-wide; high vol_switches = GIL handoffs / blocking)"
    )
    cupti_state = _cupti_subscriber_state()
    _mark(f"cupti tracking: {cupti_state}")
    if args_cli.no_cupti and cupti_state.startswith("ACTIVE"):
        _mark("WARNING: --no_cupti requested but a CUPTI subscriber is ACTIVE — the disable did not take!")
    env.close()

    print("=" * 78, flush=True)
    _report("write_data_to_sim", _WRITE_MS)  # primary (ticket) metric — always
    if args_cli.full:
        _report("full_frame", _FRAME_MS)
    print("=" * 78, flush=True)
    if args_cli.samples_out:
        _write_samples(args_cli.samples_out)


if __name__ == "__main__":
    _mark("resolving task config")
    env_cfg, _ = resolve_task_config(TASK, None)
    _mark("task config resolved; entering launch_simulation (kitless for newton)")
    with launch_simulation(env_cfg, args_cli):
        main(env_cfg)
    _mark("exited launch_simulation")
