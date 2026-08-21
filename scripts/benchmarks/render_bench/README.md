# render-bench — NVBug 6431561: `write_data_to_sim` under OVRTX vs Newton-Warp

Times `InteractiveScene.write_data_to_sim` (a physics-side call) on the
Franka-Cabinet render-bench scene under the **OVRTX** vs **Newton-Warp**
renderer — the ticket's regression.

**Kitless.** Always Newton physics (`newton_mjwarp`) + a kitless renderer
(`ovrtx` or `newton_renderer`). Isaac Sim / Kit is never launched
(`launch_simulation` is a no-op for kitless backends), so **there is no isaacsim
dependency**. The only wheel-vs-local switch is for **ovrtx** (the renderer
runtime), via `--ovrtx`.

Default metric is `write_data_to_sim`; `--full` also times the whole
`DirectRLEnv.step` frame.

Sources: task from `wahuang-rtx/IsaacLab:wahuang/render-bench-wip`
(`direct/render_bench/`, `18b17b80`); local ovrtx recipe from
`kit/rendering/ovrtx/public/tools/dev-run.sh`. This is also the consolidated
successor of the original ticket repro
(<https://github.com/NVIDIA-dev/peterv_write_data_to_sim_ovrtx_slowdown>): its
Franka-Cabinet scene cfg was verified value-identical to `render_bench_task/`
(camera pose, 256×256 tiles, lights, actuators, animation, physics presets),
its runner semantics are reproduced by
`--num_envs 1024 --num_steps 30 --warmup 12`, and its old preset grammar
(`renderer=ovrtx_renderer`, `fold_preset_tokens`) maps to today's `ovrtx` /
`newton_renderer` preset names.

## Files

| File | What |
|---|---|
| `ovrtx-env.sh` | sourceable toggle: `OVRTX_SOURCE=local\|wheel` → loader paths at the kit ovrtx build. |
| `run_render_bench.sh` | launcher: `--ovrtx`, `--renderer ovrtx\|newton`, `--num_envs/--num_steps/--warmup`, `--full`, `--both` (prints the ratio), `--halt-poll-sweep`. |
| `render_bench_runner.py` | the runner; kitless `launch_simulation`; times `write_data_to_sim` (+ `full_frame` with `--full`); `PT_MARK`/`RESULT` lines. |
| `render_bench_task/` | the vendored task (registered `Repro-RenderBench-Franka-Cabinet-v0`). |
| `render_bench_report.py` | standalone (stdlib-only) chart/stat report over the raw call dumps: pooled tables incl. **totals**, ASCII charts, and SVGs **per `guest_halt_poll_ns`**. |

Runs against `pv2-isaac-lab` by default (`ISAACLAB_DIR`).

## Run (mirrors the ticket)

```bash
# one renderer per process (exactly the ticket commands):
bash scripts/benchmarks/render_bench/run_render_bench.sh --renderer newton --num_envs 1024
bash scripts/benchmarks/render_bench/run_render_bench.sh --renderer ovrtx  --num_envs 1024

# both + slowdown ratio, one command:
bash scripts/benchmarks/render_bench/run_render_bench.sh --both --num_envs 1024

# ovrtx renderer against your LOCAL ovrtx build (after building kit):
bash scripts/benchmarks/render_bench/run_render_bench.sh --renderer ovrtx --num_envs 1024 --ovrtx local

# also time the whole frame:
bash scripts/benchmarks/render_bench/run_render_bench.sh --both --num_envs 1024 --full

# sweep the guest halt-poll window (KVM guests; needs sudo):
bash scripts/benchmarks/render_bench/run_render_bench.sh --both --halt-poll-sweep
```

Use `--warmup >= 10` (default 12). The runner accepts `--enable_cameras --headless`
for exact ticket-command compatibility (no-ops on the kitless path). Prints, e.g.:

```
RESULT metric=write_data_to_sim renderer=ovrtx ovrtx_source=wheel num_envs=16 calls=2000 \
       median_ms=3.06 mean_ms=3.19 p95_ms=3.84 p99_ms=4.65 p999_ms=7.66 \
       min_ms=2.84 max_ms=8.10 total_ms=6380.11 loop_wall_ms=25190.44 loop_share_pct=25.33
```

`total_ms` is the summed time inside the metric for that run and `loop_share_pct`
its share of the measured-loop wall clock. The sum scales with `--num_steps`, so
compare totals only across runs with the same step count — `loop_share_pct` is
the scale-free version of the same number. (It is only reported as a *finding*
for `write_data_to_sim`; for `full_frame`, which **is** the loop, the share is
trivially ~100% and the reports omit it.)

### How many calls do we take, and is p999 worth it?

`--num_steps` defaults to **1000**, and a step records one `write_data_to_sim`
per physics substep — **two on this scene** — so an arm measures **≈2000 calls**
per pass. The factor is scene-dependent: `calls=` on the `RESULT` line is the
authority, not this paragraph.

Percentiles are nearest-rank, no interpolation (`idx = round(q/100 × (n−1))`), so
at n = 2000:

| quantile | which call it actually is | verdict |
|---|---|---|
| `p95` | 101st-worst call | solid |
| `p99` | 21st-worst call | ~20 samples above it — stable enough to quote |
| `p999` | **3rd-worst call** | a `max` in disguise; one hiccup moves it |

Making `p999` a statistic rather than an outlier needs roughly 20 samples past
it, i.e. **~20 000 calls** — `--num_steps 10000`, about 4 min per arm at 16 envs
(and that is per arm *per halt-poll value*, so a full sweep grows fast).

The report does not ask for that. Since the SVG percentile curve already shows
the whole tail, the tables quote **`worst1%`** — the mean of the slowest 1% of
calls (20 calls at the default) — instead of `p999`. It tracks the tail's weight
rather than a single outlier: on the same run `p999` read 3.85× while `worst1%`
read 2.50×. The tables warn when a run is too short for either (`p99` under 52
calls, `worst1%` averaging fewer than 10 calls).

The `RESULT` line still carries `p999_ms`, and `run_render_bench.sh`'s own
summary table still has its `p999_ms` column, for continuity with numbers already
quoted on the ticket. The runner also still prints, when the count is too low:

```
PT_MARK WARNING: write_data_to_sim: p999 == max_ms with calls=400 (nearest-rank
needs >= 52 calls for p99, >= 502 for p999) — raise --num_steps to read the tail
```

At 16 envs a step costs ~23 ms and process startup ~27 s, so an arm runs ~50 s.

## Totals and charts (`render_bench_report.py`)

Median/p95/p99 answer "how slow is a call"; they do not answer "how much time did
the run actually spend in there", nor what the tail looks like. Every run
therefore also dumps **every recorded call duration** and turns them into a
stat/chart report under `logs/run_<timestamp>/`:

```
logs/run_20260821-112108/
  samples/<arm>_n<envs>_p<pass><tag>.tsv     every call, self-describing header
  charts/write_data_to_sim_n16_hp200000.svg  one per guest_halt_poll_ns value
  charts/write_data_to_sim_summary.svg       all halt-poll values, shared axes
  charts/report.txt                          the pooled stat tables (terminal output)
```

Per `guest_halt_poll_ns` value you get a **pooled stats table** — `total_s`,
share `of loop` (the measured-loop wall clock), mean/median/p95/p99/`worst1%`/max
and the `ovrtx/newton` ratio of each, including the total (`+6.57 s over 2000
calls, 3.286 ms/call`) — and one **SVG** holding three views:

- **sorted call duration vs percentile**, on a "nines" x-axis (`-log10(1-p)`), so
  the tail gets real width instead of being squashed into the last pixel — this
  is the "durations per step, sorted" view, and it replaces quoting a p999;
- a **box plot** (p25–p75 box, median, min–max whisker, `worst 1%` and max
  called out), on the same y-axis as the curve beside it;
- a **call-order** chart (per-bucket min/max band with the bucket median), which
  separates one-off spikes from a run that drifts.

All charts in a run share **one log y-range** and use the same color per
renderer, so the per-halt-poll files can be compared side by side;
`*_summary.svg` puts every halt-poll value's percentile curve on identical axes
as small multiples. The SVGs are self-contained (no JS, no external fonts) and
paint with SVG **presentation attributes**, so converters that ignore or reject
CSS (Inkscape, rsvg) still render them correctly; the only CSS is a
`prefers-color-scheme: dark` override, which browsers apply and everything else
harmlessly skips.

Tail resolution is called out rather than implied: the region of the curve where
fewer than 10 calls remain is shaded, and the table warns when the sample count
is too low for `p99` or for `worst1%` to mean anything (see below).

The reporter is standalone and stdlib-only, so an old run can be re-charted (or
re-charted differently) without re-measuring:

```bash
# re-run over a finished run's dumps
uv run --no-sync python scripts/benchmarks/render_bench/render_bench_report.py \
    --samples_dir scripts/benchmarks/render_bench/logs/run_20260821-112108/samples

# the full-frame metric instead (requires the run to have used --full)
uv run --no-sync python scripts/benchmarks/render_bench/render_bench_report.py \
    --samples_dir .../samples --metric full_frame

# stat tables only, no SVGs
... --no_svg
```

`--no-charts` on `run_render_bench.sh` skips the report (the dumps are still
written).

## guest halt-poll sweep

`--halt-poll-sweep` re-runs the whole matrix at each
`/sys/module/haltpoll/parameters/guest_halt_poll_ns` in
`200000,500000,1000000,5000000,10000000` (`--halt-poll a,b,c` for your own list),
then tabulates median/p999, totals and the ovrtx/newton ratio per value — and
`render_bench_report.py` emits its own table plus a full chart set **per value**. This is a KVM
guest knob: the haltpoll cpuidle driver spins this long before a real `HLT`
vmexit, so it directly sets the wake-up latency for the renderer/sim thread
handoffs that the ticket's per-call tax runs through.

Requires `sudo` (writes sysfs) and the `haltpoll` driver
(`cat /sys/devices/system/cpu/cpuidle/current_driver`). The original value is
read at startup and restored on exit, including on Ctrl-C; each write is
verified by readback and the sweep aborts on a mismatch rather than silently
producing N identical runs. Halt-poll values run in a fixed order within a pass
and reverse on even passes, so use `--repeats 2` (or more) before believing a
monotonic trend — at `--repeats 1` the halt axis is confounded with machine
drift.

## ovrtx: wheel ↔ local (validated)

`ovrtx-env.sh` mirrors `dev-run.sh`: `local` puts the ovrtx+ovstage
`public/python` source trees on `PYTHONPATH` and
`rendering/_build/linux-x86_64/release` on `LD_LIBRARY_PATH`, so `import ovrtx`
resolves to the source and loads the freshly built `libovrtx-dynamic.so`.
Requires `kit/rendering/_build/linux-x86_64/release/libovrtx-dynamic.so`
(`cd kit/rendering && ./build.sh -r --no-docker --devrtx`).

Only relevant for `--renderer ovrtx`. Smoke-test the toggle without the benchmark:

```bash
# from the IsaacLab repo root:
OVRTX_SOURCE=local source scripts/benchmarks/render_bench/ovrtx-env.sh
uv run --no-sync python -c "import ovrtx; print(ovrtx.__file__)"   # → under kit/…/public/python
```

## CUPTI GPU memory tracking (the main ovrtx-vs-newton delta)

Constructing an `ovrtx.Renderer` initializes `carb.cudainterop`, whose GPU
memory tracking (`memoryScope/trackAddress`, **on by default**) subscribes
CUPTI activity tracing to every CUDA runtime/driver call in the process —
taxing each of the ~109 host CUDA calls a `write_data_to_sim` makes. With it
disabled, ovrtx ≈ newton on bare metal (measured 1.00x median).

- `--no-cupti` requests the disable via `OVRTX_`-prefixed settings env vars
  (kitless runs never parse `--/` flags: the ovrtx framework startup gets
  `argv=nullptr`).
- **Trust the verification line, not the flag**: every run prints
  `PT_MARK cupti tracking: ACTIVE (...)|inactive (...)`, probed by attempting
  `cuptiSubscribe` on the exact libcupti mapped in the process (CUPTI allows
  one subscriber per library instance). A `WARNING` follows if `--no-cupti`
  did not take.
- Guaranteed fallback if the env vars fail on some build: add to the wheel's
  `ovrtx/bin/ovrtx.config.json`:

  ```json
  "plugins": { "carb.memorytracking.plugin": {
      "memoryScope": { "trackAddress": { "enabled": false } } } }
  ```

- Under Kit (non-kitless runs only), the equivalent flag is
  `--/plugins/carb.memorytracking.plugin/memoryScope/trackAddress/enabled=false`.

## Diagnostics

The runner prints `PT_MARK <stage>` at each step (task-config → gym.make →
reset → warmup → measuring → done), and a `PT_MARK cupti tracking:` verdict
after measurement. If a run produces no `RESULT`, `run_render_bench.sh` prints
the exit code and the last 30 log lines (`logs/write_<renderer>.log`); the
last `PT_MARK` shows where it stopped.
