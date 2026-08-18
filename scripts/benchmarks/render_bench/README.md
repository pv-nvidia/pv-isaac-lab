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
| `run_render_bench.sh` | launcher: `--ovrtx`, `--renderer ovrtx\|newton`, `--num_envs/--num_steps/--warmup`, `--full`, `--both` (prints the ratio). |
| `render_bench_runner.py` | the runner; kitless `launch_simulation`; times `write_data_to_sim` (+ `full_frame` with `--full`); `PT_MARK`/`RESULT` lines. |
| `render_bench_task/` | the vendored task (registered `Repro-RenderBench-Franka-Cabinet-v0`). |

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
```

Use `--warmup >= 10` (default 12). The runner accepts `--enable_cameras --headless`
for exact ticket-command compatibility (no-ops on the kitless path). Prints, e.g.:

```
RESULT metric=write_data_to_sim renderer=ovrtx ovrtx_source=wheel num_envs=1024 calls=30 \
       median_ms=4.71 mean_ms=6.10 p95_ms=12.20 min_ms=2.40 max_ms=38.9
```

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

