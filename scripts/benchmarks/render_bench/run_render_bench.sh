#!/usr/bin/env bash
# NVBug 6431561 repro: time InteractiveScene.write_data_to_sim under the OVRTX vs
# Newton-Warp renderer on an identical scene, with a switch to load either the
# installed ovrtx wheel or your local kit build.
#
# Default metric: write_data_to_sim (the ticket). Add --full to also time the
# whole DirectRLEnv.step frame.
#
# Kitless benchmark: Newton physics + ovrtx|newton renderer. Isaac Sim / Kit is
# never launched (launch_simulation is a no-op for kitless backends), so there is
# no isaacsim dependency.
#
# The OVRTX cost is a fixed host-side per-call tax, so the ratio depends on
# --num_envs: it dominates at 16 envs and is amortised at 1024. Compare a ratio
# only against another ratio taken at the SAME env count.
#
# Usage:
#   bash run_render_bench.sh [--ovrtx wheel|local] [--renderer ovrtx|newton]
#                            [--num_envs N] [--num_steps N] [--warmup N] [--full]
#                            [--both]      # newton and ovrtx, print the slowdown ratio
#                            [--repeats N] # passes; arm order reverses each pass
#                            [--nsys]      # wrap each arm in `nsys profile`; writes
#                                          # logs/nsys_<arm>_*.nsys-rep + .sqlite + .stats.txt
#                            [--nsys-args "..."]  # extra nsys profile options
#                            [--no-cupti]  # disable OVRTX's CUPTI GPU memory tracking
#                            [--pin-sim]   # affinity experiment: renderer threads on CPUs 2..N-1,
#                                          # sim thread alone on CPU 0 (CPU 1 idle) — separates
#                                          # sibling/cache contention from global frequency effects
#
# Examples:
#   bash run_render_bench.sh --both                        # ticket config (16 envs)
#   bash run_render_bench.sh --renderer ovrtx --ovrtx local --num_envs 1024
#   bash run_render_bench.sh --both --nsys                 # profile both arms, export sqlite
#   bash run_render_bench.sh --both --nsys --no-cupti      # ...with CUPTI tracking off
#
# nsys notes (run on the host; needs `nsys` on PATH, override with NSYS_BIN):
#   - Capture starts at cudaProfilerStart (after warmup) and traces cuda,nvtx,osrt;
#     osrt shows blocking OS calls (futex/pthread), the hypothesis space here.
#   - CPU sampling is OFF by default because KVM guests usually lack PMU access;
#     on bare metal add: --nsys-args "--sample=cpu --cpuctxsw=process-tree"
#   - RESULT timings under nsys include tracing overhead: use the profiles for
#     attribution, quote ratios only from un-profiled runs.
#   - Keep --num_steps modest (default 50) or reps get large.
#
# --ovrtx selects the installed ovrtx wheel (default) vs your local kit build
# (only relevant for --renderer ovrtx).
# Env: ISAACLAB_DIR (default: this checkout — three levels up),
#      KIT_ROOT     (default: a sibling ``kit`` checkout next to ISAACLAB_DIR)
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# scripts/benchmarks/render_bench -> repo root is three levels up.
: "${ISAACLAB_DIR:=$(cd "$HERE/../../.." && pwd)}"
export ISAACLAB_DIR
# The kit checkout (for --ovrtx local) is a workspace sibling of ISAACLAB_DIR.
export KIT_ROOT="${KIT_ROOT:-$(cd "$ISAACLAB_DIR/.." && pwd)/kit}"
export OMNI_KIT_ACCEPT_EULA=YES

OVRTX_SOURCE_ARG="wheel"
RENDERER="ovrtx"; NUM_ENVS=16; NUM_STEPS=50; WARMUP=12; BOTH=0; FULL=""; REPEATS=1
NSYS=0; NSYS_ARGS=""; NOCUPTI=0; PINSIM=0; GILPROBE=0; STEPSLEEP=""; DEPRIO=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ovrtx)     OVRTX_SOURCE_ARG="$2"; shift 2 ;;
        --renderer)  RENDERER="$2"; shift 2 ;;
        --num_envs)  NUM_ENVS="$2"; shift 2 ;;
        --num_steps) NUM_STEPS="$2"; shift 2 ;;
        --warmup)    WARMUP="$2"; shift 2 ;;
        --repeats)   REPEATS="$2"; shift 2 ;;
        --full)      FULL="--full"; shift ;;
        --both)      BOTH=1; shift ;;
        --nsys)      NSYS=1; shift ;;
        --nsys-args) NSYS_ARGS="$2"; shift 2 ;;
        --no-cupti)  NOCUPTI=1; shift ;;
        --pin-sim)   PINSIM=1; shift ;;
        --gil-probe) GILPROBE=1; shift ;;
        --step-sleep) STEPSLEEP="$2"; shift 2 ;;
        --deprioritize) DEPRIO=1; shift ;;
        -h|--help)   awk 'NR>1 && !/^#/{exit} NR>1{print}' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

[[ "$REPEATS" -ge 1 ]] || { echo "--repeats must be >= 1" >&2; exit 2; }
NSYS_BIN="${NSYS_BIN:-nsys}"
if (( NSYS )); then
    command -v "$NSYS_BIN" >/dev/null 2>&1 || {
        echo "!!! --nsys: '$NSYS_BIN' not found on PATH (install Nsight Systems or set NSYS_BIN)" >&2; exit 2; }
    echo "!!! nsys attached: RESULT timings include tracing overhead — use profiles for attribution, not ratios." >&2
fi
if [[ "$WARMUP" -lt 10 ]]; then
    echo "!!! --warmup $WARMUP is below 10; the OVRTX RT pipeline is still building." >&2
fi

# Select ovrtx source (exports PYTHONPATH/LD_LIBRARY_PATH for 'local'; no-op for 'wheel').
export OVRTX_SOURCE="$OVRTX_SOURCE_ARG"
# shellcheck disable=SC1091
source "$HERE/ovrtx-env.sh" || exit 1

mkdir -p "$HERE/logs"
SAMPLES="$(mktemp)"
trap 'rm -f "$SAMPLES"' EXIT

# A ratio is only interpretable next to the machine it came from.
print_fingerprint() {
    echo "===================== machine fingerprint =====================" >&2
    printf "  cpu   : %s\n" "$(lscpu 2>/dev/null | sed -n 's/^Model name: *//p' | head -1)" >&2
    printf "  cores : %s\n" "$(nproc 2>/dev/null)" >&2
    printf "  virt  : %s\n" "$(systemd-detect-virt 2>/dev/null || echo unknown)" >&2
    local hv="absent (bare/container)"
    grep -qm1 ' hypervisor' /proc/cpuinfo && hv="flag SET (guest)"
    printf "  hyperv: %s\n" "$hv" >&2
    local gpu
    gpu="$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1)"
    printf "  gpu   : %s\n" "$gpu" >&2
    local pyver='import ovrtx,os;print(getattr(ovrtx,"__version__","?"),os.path.dirname(ovrtx.__file__ or "?"))'
    local ov
    ov="$( ( cd "$ISAACLAB_DIR" && uv run --no-sync python -c "$pyver" ) 2>/dev/null )" || ov="not importable"
    printf "  ovrtx : %s (%s)\n" "$ov" "$OVRTX_SOURCE" >&2
    echo "===============================================================" >&2
}

# Runs one renderer once. Human-facing output goes to stderr; the parsed
# "median p95" pair goes to stdout.
run_one() {
    local renderer="$1" envs="$2" pass="${3:-1}"
    local tag=""
    (( NOCUPTI )) && tag="_nocupti"
    (( PINSIM )) && tag="${tag}_pinsim"
    local log="$HERE/logs/write_${renderer}_n${envs}_p${pass}${tag}.log"
    local nsys_out="$HERE/logs/nsys_${renderer}_n${envs}_p${pass}${tag}"
    local runner_args=()
    (( NOCUPTI )) && runner_args+=(--no_cupti)
    (( PINSIM )) && runner_args+=(--pin_sim)
    (( GILPROBE )) && runner_args+=(--gil_probe)
    [[ -n "$STEPSLEEP" ]] && runner_args+=(--step_sleep "$STEPSLEEP")
    (( DEPRIO )) && runner_args+=(--deprioritize_workers)
    local prefix=()
    if (( NSYS )); then
        runner_args+=(--profile)
        # osrt traces blocking OS calls (futex/pthread) — the hypothesis space for
        # a host-side per-call tax. Capture opens at the runner's cudaProfilerStart
        # (post-warmup) so RT-pipeline build noise stays out of the report. CPU
        # sampling stays off (KVM guests usually lack PMU); see --nsys-args.
        # shellcheck disable=SC2206  # NSYS_ARGS is intentionally word-split
        prefix=("$NSYS_BIN" profile -o "$nsys_out" --force-overwrite=true
                --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none
                --capture-range=cudaProfilerApi --capture-range-end=stop
                $NSYS_ARGS)
    fi
    echo ">>> renderer=$renderer ovrtx=$OVRTX_SOURCE envs=$envs steps=$NUM_STEPS warmup=$WARMUP pass=$pass nsys=$NSYS no_cupti=$NOCUPTI ${FULL}" >&2
    # uv run inside the IsaacLab checkout; --no-sync so uv can't reinstall ovrtx
    # over the PYTHONPATH shadow. --enable_cameras --headless mirror the ticket.
    ( cd "$ISAACLAB_DIR" && "${prefix[@]}" uv run --no-sync python "$HERE/render_bench_runner.py" \
        --renderer "$renderer" --num_envs "$envs" --num_steps "$NUM_STEPS" \
        --warmup "$WARMUP" $FULL --enable_cameras --headless "${runner_args[@]}" ) > "$log" 2>&1
    local rc=$?
    echo "EXIT=$rc" >> "$log"
    if (( NSYS )); then
        if [[ -f "$nsys_out.nsys-rep" ]]; then
            if "$NSYS_BIN" export --type=sqlite --force-overwrite=true \
                    --output="$nsys_out.sqlite" "$nsys_out.nsys-rep" >>"$log" 2>&1; then
                echo "    nsys: $nsys_out.nsys-rep + .sqlite" >&2
            else
                echo "!!! nsys sqlite export failed (see $log)" >&2
            fi
            # Report names differ across nsys versions; try new-style, then old.
            { "$NSYS_BIN" stats --report osrt_sum,cuda_api_sum,nvtx_sum "$nsys_out.nsys-rep" \
              || "$NSYS_BIN" stats --report osrtsum,cudaapisum,nvtxsum "$nsys_out.nsys-rep"; } \
                > "$nsys_out.stats.txt" 2>>"$log" \
                && echo "    nsys stats: $nsys_out.stats.txt" >&2
        else
            echo "!!! nsys produced no $nsys_out.nsys-rep (capture range never opened? see $log)" >&2
        fi
    fi
    grep -E 'cupti tracking:|WARNING: --no_cupti|PT_MARK sched:|PT_MARK pin_sim|pin_sim:|python threads:|gil_probe:|deprioritize_workers:' "$log" >&2
    if grep -qE '^RESULT ' "$log"; then
        grep -E '^RESULT ' "$log" >&2
    else
        echo "!!! renderer=$renderer produced NO RESULT (exit=$rc)." >&2
        echo "    (exit 139 = segfault; often a local-build vs wheel Kit/USD mismatch — try --ovrtx wheel)" >&2
        echo "    last 30 lines of $log:" >&2
        tail -30 "$log" >&2
    fi
    if [[ $rc -ne 0 ]]; then echo "!!! renderer=$renderer FAILED (exit=$rc); full log: $log" >&2; return 1; fi
    grep -E '^RESULT metric=write_data_to_sim ' "$log" | tail -1 \
        | sed -E 's/.*median_ms=([0-9.]+).*p95_ms=([0-9.]+).*/\1 \2/'
}

# Both arms, REPEATS passes, arm order reversed on odd passes so that a machine
# drifting during the run cannot be mistaken for a renderer effect.
run_both() {
    local envs="$1" pass arms pair
    for (( pass=1; pass<=REPEATS; pass++ )); do
        if (( pass % 2 )); then arms=(newton ovrtx); else arms=(ovrtx newton); fi
        (( REPEATS > 1 )) && echo "########## envs=$envs pass $pass/$REPEATS (${arms[0]} first) ##########" >&2
        for arm in "${arms[@]}"; do
            pair="$(run_one "$arm" "$envs" "$pass")" || return 1
            echo "$envs $arm $pair" >> "$SAMPLES"
        done
    done
}

report() {
    # LC_ALL=C: under comma-decimal locales awk parses "1.8668" as 1 and
    # printf emits "1,0000"; the RESULT lines are always dot-decimal.
    LC_ALL=C awk '
    { envs=$1; arm=$2; med=$3; p95=$4
      key=envs SUBSEP arm; n[key]++; smed[key]+=med; sp95[key]+=p95
      if (!(key in lo) || med<lo[key]) lo[key]=med
      if (!(key in hi) || med>hi[key]) hi[key]=med
      seen[envs]=1 }
    END {
      print ""
      print "========== write_data_to_sim: ovrtx vs newton (median) =========="
      printf "  %6s  %-8s %10s %10s %18s\n", "envs", "renderer", "median_ms", "p95_ms", "median spread"
      ns=0; for (e in seen) order[++ns]=e
      for (i=1;i<ns;i++) for (j=i+1;j<=ns;j++) if (order[i]+0>order[j]+0) { t=order[i];order[i]=order[j];order[j]=t }
      for (i=1;i<=ns;i++) {
        e=order[i]
        for (a=1;a<=2;a++) {
          arm=(a==1)?"newton":"ovrtx"; key=e SUBSEP arm
          if (!(key in n)) continue
          am=smed[key]/n[key]; ap=sp95[key]/n[key]
          sp=(am>0)?(hi[key]/lo[key]):0
          printf "  %6s  %-8s %10.4f %10.4f   %8.4f-%.4f (%.2fx)\n", e, arm, am, ap, lo[key], hi[key], sp
          M[arm]=am; P[arm]=ap; S[arm]=sp
        }
        if (("newton" in M) && ("ovrtx" in M) && M["newton"]>0) {
          r=M["ovrtx"]/M["newton"]; rp=(P["newton"]>0)?P["ovrtx"]/P["newton"]:0
          printf "  -> envs=%-6s ovrtx/newton: %.2fx median, %.2fx p95\n", e, r, rp
          worst=(S["newton"]>S["ovrtx"])?S["newton"]:S["ovrtx"]
          if (worst-1 >= 0.5*(r-1) && r>1)
            printf "     !!! NOT QUOTABLE: per-arm spread %.2fx is >=50%% of the effect %.2fx.\n", worst, r
        }
        delete M; delete P; delete S
        print ""
      }
      print "  A ratio is comparable only to another ratio at the SAME env count."
      print "================================================================"
    }' "$SAMPLES" >&2
}

print_fingerprint

if [[ "$BOTH" == "1" ]]; then
    run_both "$NUM_ENVS" || exit 3
    report
else
    run_one "$RENDERER" "$NUM_ENVS" >/dev/null || exit 3
fi
