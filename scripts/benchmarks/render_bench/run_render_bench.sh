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
#                            [--halt-poll-sweep]  # re-run the whole matrix at each guest_halt_poll_ns
#                                                 # in 200000,500000,1000000,5000000,10000000 (needs sudo)
#                            [--halt-poll LIST]   # ...at your own comma-separated ns values
#                            [--no-charts] # skip the per-halt-poll chart/stat report
#
# Every run writes logs/run_<timestamp>/ with:
#   samples/<arm>_n<envs>_p<pass><tag>.tsv   every recorded call duration
#   charts/<metric>_n<envs>_hp<value>.svg    per halt-poll value: sorted-duration
#                                            percentile curve + box plot + call-order view
#   charts/<metric>_summary.svg              the same curves, one panel per halt-poll value
#   charts/report.txt                        the pooled stat tables (terminal output)
# The charts come from render_bench_report.py, which is standalone: re-run it on
# an old run without re-measuring, e.g.
#   uv run --no-sync python render_bench_report.py --samples_dir logs/run_<ts>/samples
#
# Examples:
#   bash run_render_bench.sh --both                        # ticket config (16 envs)
#   bash run_render_bench.sh --renderer ovrtx --ovrtx local --num_envs 1024
#   bash run_render_bench.sh --both --nsys                 # profile both arms, export sqlite
#   bash run_render_bench.sh --both --nsys --no-cupti      # ...with CUPTI tracking off
#   bash run_render_bench.sh --both --halt-poll-sweep      # ratio vs guest halt-poll window
#
# nsys notes (run on the host; needs `nsys` on PATH, override with NSYS_BIN):
#   - Capture starts at cudaProfilerStart (after warmup) and traces cuda,nvtx,osrt;
#     osrt shows blocking OS calls (futex/pthread), the hypothesis space here.
#   - CPU sampling is OFF by default because KVM guests usually lack PMU access;
#     on bare metal add: --nsys-args "--sample=cpu --cpuctxsw=process-tree"
#   - RESULT timings under nsys include tracing overhead: use the profiles for
#     attribution, quote ratios only from un-profiled runs.
#   - Under nsys drop --num_steps well below the default 1000 or reps get huge.
#
# total_s / share: the summed time inside write_data_to_sim for one pass and its
# share of that pass's measured-loop wall clock. The sum scales with --num_steps,
# so compare totals only across runs with the SAME step count; the share is the
# scale-free version of the same number.
#
# Percentiles are nearest-rank over the RECORDED WRITE CALLS, not over steps: a
# step records one write_data_to_sim per physics substep (2 on this scene), so
# calls ~= 2 x --num_steps. p99 needs >= 52 calls and p999 >= 502 to say anything
# max_ms does not already say; the runner prints a WARNING whenever a run lands
# below that, which is the authoritative check. --num_steps defaults to 1000 for
# that reason — lower it only when you do not care about the tail.
#
# guest halt-poll (--halt-poll*): rewrites /sys/module/haltpoll/parameters/
# guest_halt_poll_ns (KVM guests only, via sudo) and re-runs the whole matrix at
# each value, then tabulates the ratio per value. The original value is restored
# on exit, including on Ctrl-C.
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
RENDERER="ovrtx"; NUM_ENVS=16; NUM_STEPS=1000; WARMUP=12; BOTH=0; FULL=""; REPEATS=1
NSYS=0; NSYS_ARGS=""; NOCUPTI=0; PINSIM=0; GILPROBE=0; STEPSLEEP=""; DEPRIO=0; CHARTS=1
HALTPOLL_PATH="${HALTPOLL_PATH:-/sys/module/haltpoll/parameters/guest_halt_poll_ns}"
HALTPOLL_DEFAULT_SET="200000,500000,1000000,5000000,10000000"
HALTPOLL_LIST=""
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
        --no-charts) CHARTS=0; shift ;;
        --halt-poll) HALTPOLL_LIST="$2"; shift 2 ;;
        --halt-poll-sweep) HALTPOLL_LIST="$HALTPOLL_DEFAULT_SET"; shift ;;
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
# Percentile support is checked by the runner (it knows the call count, which is
# a per-substep multiple of --num_steps); its WARNING line is echoed by run_one.

# --halt-poll[-sweep]: validate the list and remember the value to restore.
HALT_VALUES=(); HALT_ORIG=""; HALT_CUR="asis"
if [[ -n "$HALTPOLL_LIST" ]]; then
    IFS=',' read -r -a HALT_VALUES <<< "$HALTPOLL_LIST"
    (( ${#HALT_VALUES[@]} )) || { echo "--halt-poll: empty list" >&2; exit 2; }
    for v in "${HALT_VALUES[@]}"; do
        [[ "$v" =~ ^[0-9]+$ ]] || { echo "--halt-poll: '$v' is not a nanosecond integer" >&2; exit 2; }
    done
    [[ -r "$HALTPOLL_PATH" ]] || {
        echo "!!! --halt-poll: $HALTPOLL_PATH is not readable — the haltpoll cpuidle driver" >&2
        echo "    is only present in KVM guests (check: cat /sys/devices/system/cpu/cpuidle/current_driver)" >&2
        exit 2; }
    HALT_ORIG="$(cat "$HALTPOLL_PATH")"
    sudo -n true 2>/dev/null || echo "!!! --halt-poll needs sudo to write $HALTPOLL_PATH (you may be prompted)." >&2
    echo "### halt-poll sweep: ${HALT_VALUES[*]} ns (was $HALT_ORIG; restored on exit)" >&2
fi

# Select ovrtx source (exports PYTHONPATH/LD_LIBRARY_PATH for 'local'; no-op for 'wheel').
export OVRTX_SOURCE="$OVRTX_SOURCE_ARG"
# shellcheck disable=SC1091
source "$HERE/ovrtx-env.sh" || exit 1

mkdir -p "$HERE/logs"
# Raw per-call dumps and charts live in a per-invocation directory so a report
# can never silently mix in samples from an earlier run.
RUNDIR="$HERE/logs/run_$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RUNDIR/samples"
SAMPLE_FILES=()
SAMPLES="$(mktemp)"
# Restore the machine's halt-poll window whatever happens: an interrupted sweep
# must not strand the guest at the last value written.
cleanup() {
    rm -f "$SAMPLES"
    local now=""
    [[ -n "$HALT_ORIG" ]] && now="$(cat "$HALTPOLL_PATH" 2>/dev/null)"
    if [[ -n "$HALT_ORIG" && "$now" != "$HALT_ORIG" ]]; then
        if echo "$HALT_ORIG" | sudo tee "$HALTPOLL_PATH" >/dev/null 2>&1; then
            echo "### guest_halt_poll_ns restored to $HALT_ORIG" >&2
        else
            echo "!!! could not restore guest_halt_poll_ns=$HALT_ORIG — set it by hand:" >&2
            echo "    echo $HALT_ORIG | sudo tee $HALTPOLL_PATH" >&2
        fi
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Writes one halt-poll value and verifies the readback: a silently rejected write
# would turn the sweep into N identical runs that read as a null result.
set_halt_poll() {
    local want="$1" got
    echo "$want" | sudo tee "$HALTPOLL_PATH" >/dev/null || {
        echo "!!! failed to write $want to $HALTPOLL_PATH" >&2; return 1; }
    got="$(cat "$HALTPOLL_PATH")"
    [[ "$got" == "$want" ]] || {
        echo "!!! $HALTPOLL_PATH readback=$got after writing $want — aborting the sweep" >&2; return 1; }
    echo "### guest_halt_poll_ns = $want ns" >&2
}

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
# "median p95 p99 p999" tuple goes to stdout.
run_tag() {
    local tag=""
    (( NOCUPTI )) && tag="_nocupti"
    (( PINSIM )) && tag="${tag}_pinsim"
    [[ "$HALT_CUR" != "asis" ]] && tag="${tag}_hp${HALT_CUR}"
    printf '%s' "$tag"
}

run_one() {
    local renderer="$1" envs="$2" pass="${3:-1}" samples="${4:-}" tag="${5:-$(run_tag)}"
    local log="$HERE/logs/write_${renderer}_n${envs}_p${pass}${tag}.log"
    local nsys_out="$HERE/logs/nsys_${renderer}_n${envs}_p${pass}${tag}"
    local runner_args=()
    (( NOCUPTI )) && runner_args+=(--no_cupti)
    (( PINSIM )) && runner_args+=(--pin_sim)
    (( GILPROBE )) && runner_args+=(--gil_probe)
    [[ -n "$STEPSLEEP" ]] && runner_args+=(--step_sleep "$STEPSLEEP")
    (( DEPRIO )) && runner_args+=(--deprioritize_workers)
    # Raw per-call durations, self-describing: the chart report never has to
    # re-derive halt_poll/pass from a file name.
    [[ -n "$samples" ]] && runner_args+=(--samples_out "$samples"
                                         --run_meta "halt_poll=$HALT_CUR,pass=$pass,tag=${tag#_}")
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
    echo ">>> renderer=$renderer ovrtx=$OVRTX_SOURCE envs=$envs steps=$NUM_STEPS warmup=$WARMUP pass=$pass nsys=$NSYS no_cupti=$NOCUPTI halt_poll=$HALT_CUR ${FULL}" >&2
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
    grep -E 'cupti tracking:|WARNING:|PT_MARK sched:|PT_MARK pin_sim|pin_sim:|python threads:|gil_probe:|deprioritize_workers:|PT_MARK samples:' "$log" >&2
    if grep -qE '^RESULT ' "$log"; then
        grep -E '^RESULT ' "$log" >&2
    else
        echo "!!! renderer=$renderer produced NO RESULT (exit=$rc)." >&2
        echo "    (exit 139 = segfault; often a local-build vs wheel Kit/USD mismatch — try --ovrtx wheel)" >&2
        echo "    last 30 lines of $log:" >&2
        tail -30 "$log" >&2
    fi
    if [[ $rc -ne 0 ]]; then echo "!!! renderer=$renderer FAILED (exit=$rc); full log: $log" >&2; return 1; fi
    # -n/p: on a miss print nothing rather than passing the RESULT line through
    # (a pass-through would silently feed junk fields to the awk report).
    local stats
    stats="$(grep -E '^RESULT metric=write_data_to_sim ' "$log" | tail -1 \
        | sed -nE 's/.*median_ms=([0-9.]+).*p95_ms=([0-9.]+).*p99_ms=([0-9.]+).*p999_ms=([0-9.]+).*/\1 \2 \3 \4/p')"
    if [[ -z "$stats" ]]; then
        echo "!!! could not parse median/p95/p99/p999 from the RESULT line in $log" >&2
        return 1
    fi
    # Totals are parsed separately (and default to 0) so a runner that predates
    # total_ms/wall_ms still yields a usable table instead of aborting the sweep.
    local extra
    extra="$(grep -E '^RESULT metric=write_data_to_sim ' "$log" | tail -1 \
        | sed -nE 's/.*total_ms=([0-9.]+).*loop_wall_ms=([0-9.]+).*/\1 \2/p')"
    [[ -n "$extra" ]] || extra="0 0"
    printf '%s %s\n' "$stats" "$extra"
}

# One pass at the current halt-poll value. With --both the arm order reverses on
# even passes so that a machine drifting during the run cannot be mistaken for a
# renderer effect.
run_pass() {
    local envs="$1" pass="$2" arms stats
    if (( BOTH )); then
        if (( pass % 2 )); then arms=(newton ovrtx); else arms=(ovrtx newton); fi
    else
        arms=("$RENDERER")
    fi
    local tag; tag="$(run_tag)"
    for arm in "${arms[@]}"; do
        # run_one is captured in a subshell, so the sample path is chosen HERE
        # (the array append has to happen in this shell to survive).
        local dump="$RUNDIR/samples/${arm}_n${envs}_p${pass}${tag}.tsv"
        stats="$(run_one "$arm" "$envs" "$pass" "$dump" "$tag")" || return 1
        echo "$envs $HALT_CUR $arm $stats" >> "$SAMPLES"
        if [[ -s "$dump" ]]; then
            SAMPLE_FILES+=("$dump")
        else
            # A measured arm with no dump would silently drop out of the charts.
            echo "!!! renderer=$arm produced a RESULT but no sample dump ($dump) — it will be" >&2
            echo "    missing from the chart report; the RESULT table is unaffected." >&2
        fi
    done
}

# The measurement matrix: passes x halt-poll values x arms. The halt-poll order
# reverses on even passes for the same reason the arm order does — with
# --repeats 1 the halt axis and the time axis are confounded, so a monotonic
# "trend" over halt values may just be the machine warming up.
run_matrix() {
    local envs="$1" pass hv i halts=()
    for (( pass=1; pass<=REPEATS; pass++ )); do
        if (( ${#HALT_VALUES[@]} )); then
            halts=("${HALT_VALUES[@]}")
            if (( pass % 2 == 0 )); then
                local rev=()
                for (( i=${#halts[@]}-1; i>=0; i-- )); do rev+=("${halts[i]}"); done
                halts=("${rev[@]}")
            fi
        else
            halts=(asis)
        fi
        for hv in "${halts[@]}"; do
            if [[ "$hv" != "asis" ]]; then
                HALT_CUR="$hv"
                set_halt_poll "$hv" || return 4
            fi
            (( REPEATS > 1 || ${#HALT_VALUES[@]} )) \
                && echo "########## envs=$envs pass $pass/$REPEATS halt_poll=$HALT_CUR ##########" >&2
            run_pass "$envs" "$pass" || return 1
        done
    done
}

report() {
    # LC_ALL=C: under comma-decimal locales awk parses "1.8668" as 1 and
    # printf emits "1,0000"; the RESULT lines are always dot-decimal.
    LC_ALL=C awk '
    function fnum(x) { return x+0 }
    { envs=$1; halt=$2; arm=$3; med=$4; p95=$5; p99=$6; p999=$7; tot=$8; wall=$9
      key=envs SUBSEP halt SUBSEP arm; n[key]++
      smed[key]+=med; sp95[key]+=p95; sp99[key]+=p99; sp999[key]+=p999
      stot[key]+=tot; swall[key]+=wall
      if (!(key in lo) || med<lo[key]) lo[key]=med
      if (!(key in hi) || med>hi[key]) hi[key]=med
      seen[envs]=1; halts[halt]=1 }
    END {
      ne=0; for (e in seen) eo[++ne]=e
      for (i=1;i<ne;i++) for (j=i+1;j<=ne;j++) if (fnum(eo[i])>fnum(eo[j])) { t=eo[i];eo[i]=eo[j];eo[j]=t }
      nh=0; for (h in halts) ho[++nh]=h
      for (i=1;i<nh;i++) for (j=i+1;j<=nh;j++) if (fnum(ho[i])>fnum(ho[j])) { t=ho[i];ho[i]=ho[j];ho[j]=t }
      print ""
      print "========== write_data_to_sim: ovrtx vs newton (median) =========="
      printf "  %6s %12s  %-8s %10s %10s %10s %10s %9s %7s %18s\n", \
             "envs", "halt_poll", "renderer", "median_ms", "p95_ms", "p99_ms", "p999_ms", \
             "total_s", "share", "median spread"
      for (i=1;i<=ne;i++) {
        e=eo[i]
        for (k=1;k<=nh;k++) {
          h=ho[k]; any=0
          for (a=1;a<=2;a++) {
            arm=(a==1)?"newton":"ovrtx"; key=e SUBSEP h SUBSEP arm
            if (!(key in n)) continue
            any=1
            am=smed[key]/n[key]; ap=sp95[key]/n[key]
            a99=sp99[key]/n[key]; a999=sp999[key]/n[key]
            at=stot[key]/n[key]; aw=swall[key]/n[key]
            sh=(aw>0)?(100*at/aw):0
            sp=(am>0)?(hi[key]/lo[key]):0
            printf "  %6s %12s  %-8s %10.4f %10.4f %10.4f %10.4f %9.3f %6.1f%%   %8.4f-%.4f (%.2fx)\n", \
                   e, h, arm, am, ap, a99, a999, at/1000, sh, lo[key], hi[key], sp
            M[arm]=am; P[arm]=ap; P99[arm]=a99; P999[arm]=a999; S[arm]=sp; T[arm]=at
            SM[e,h,arm]=am; S999[e,h,arm]=a999; STO[e,h,arm]=at
          }
          if (!any) continue
          if (("newton" in M) && ("ovrtx" in M) && M["newton"]>0) {
            r=M["ovrtx"]/M["newton"]; rp=(P["newton"]>0)?P["ovrtx"]/P["newton"]:0
            r99=(P99["newton"]>0)?P99["ovrtx"]/P99["newton"]:0
            r999=(P999["newton"]>0)?P999["ovrtx"]/P999["newton"]:0
            rt=(T["newton"]>0)?T["ovrtx"]/T["newton"]:0
            printf "  -> envs=%-6s halt_poll=%-10s ovrtx/newton: %.2fx median, %.2fx p95, %.2fx p99, %.2fx p999, %.2fx total\n", \
                   e, h, r, rp, r99, r999, rt
            printf "     total time in write_data_to_sim: newton %.2fs vs ovrtx %.2fs (+%.2fs)\n", \
                   T["newton"]/1000, T["ovrtx"]/1000, (T["ovrtx"]-T["newton"])/1000
            RAT[e,h]=r; RATT[e,h]=rt
            worst=(S["newton"]>S["ovrtx"])?S["newton"]:S["ovrtx"]
            if (worst-1 >= 0.5*(r-1) && r>1)
              printf "     !!! NOT QUOTABLE: per-arm spread %.2fx is >=50%% of the effect %.2fx.\n", worst, r
          }
          delete M; delete P; delete P99; delete P999; delete S; delete T
          print ""
        }
      }
      if (nh>1) {
        for (i=1;i<=ne;i++) {
          e=eo[i]
          printf "  ---------- guest_halt_poll_ns sweep (envs=%s) ----------\n", e
          printf "  %12s %12s %12s %8s %12s %12s %10s %10s %8s\n", \
                 "halt_poll_ns", "newton_med", "ovrtx_med", "ratio", "newton_p999", "ovrtx_p999", \
                 "newton_s", "ovrtx_s", "total"
          for (k=1;k<=nh;k++) {
            h=ho[k]
            printf "  %12s %12s %12s %8s %12s %12s %10s %10s %8s\n", h, \
                   ((e,h,"newton") in SM) ? sprintf("%.4f", SM[e,h,"newton"]) : "-", \
                   ((e,h,"ovrtx")  in SM) ? sprintf("%.4f", SM[e,h,"ovrtx"])  : "-", \
                   ((e,h) in RAT)         ? sprintf("%.2fx", RAT[e,h])        : "-", \
                   ((e,h,"newton") in S999) ? sprintf("%.4f", S999[e,h,"newton"]) : "-", \
                   ((e,h,"ovrtx")  in S999) ? sprintf("%.4f", S999[e,h,"ovrtx"])  : "-", \
                   ((e,h,"newton") in STO) ? sprintf("%.3f", STO[e,h,"newton"]/1000) : "-", \
                   ((e,h,"ovrtx")  in STO) ? sprintf("%.3f", STO[e,h,"ovrtx"]/1000)  : "-", \
                   ((e,h) in RATT)        ? sprintf("%.2fx", RATT[e,h])       : "-"
          }
          print ""
        }
        print "  Halt-poll values run in a fixed order within a pass (reversed on even"
        print "  passes): at --repeats 1 a monotonic trend may just be machine drift."
      }
      print "  Columns are the mean over passes of each pass\47s own statistic; total_s is one"
      print "  pass\47s summed write time (share = its % of the measured-loop wall clock) and"
      print "  compares only at equal --num_steps. The chart report below pools all calls."
      print "  Tail quantiles are nearest-rank over recorded write calls (~2 per step):"
      print "  p99 needs >= 52 calls, p999 >= 502, else they just repeat max_ms."
      print "  A ratio is comparable only to another ratio at the SAME env count."
      print "================================================================"
    }' "$SAMPLES" >&2
}

# Pooled stats + ASCII/SVG charts per guest_halt_poll_ns, from the raw dumps.
# A failure here must not lose a finished measurement run: the samples stay on
# disk and render_bench_report.py can be re-run against them at any time.
charts() {
    (( CHARTS )) || { echo "### charts skipped (--no-charts); samples: $RUNDIR/samples" >&2; return 0; }
    (( ${#SAMPLE_FILES[@]} )) || { echo "!!! no sample dumps were produced — skipping charts" >&2; return 0; }
    ( cd "$ISAACLAB_DIR" && uv run --no-sync python "$HERE/render_bench_report.py" \
        --samples "${SAMPLE_FILES[@]}" --svg_dir "$RUNDIR/charts" \
        --text_out "$RUNDIR/charts/report.txt" ) >&2 \
        || echo "!!! chart report failed; raw samples kept in $RUNDIR/samples" >&2
}

print_fingerprint

run_matrix "$NUM_ENVS" || exit 3
report
charts
