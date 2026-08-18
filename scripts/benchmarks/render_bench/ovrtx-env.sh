# shellcheck shell=bash
# Sourceable helper: select which `ovrtx` (and `ovstage`) Python + native libs a
# subsequent `python`/`uv run` will import.
#
#   OVRTX_SOURCE=wheel  (default) -> use whatever is installed in the venv
#                                    (the Artifactory ovrtx wheel). No-op.
#   OVRTX_SOURCE=local            -> use the locally-built ovrtx from the kit
#                                    tree: its Python source trees on PYTHONPATH
#                                    and rendering/_build/.../release on
#                                    LD_LIBRARY_PATH. This mirrors exactly what
#                                    kit/rendering/ovrtx/public/tools/dev-run.sh
#                                    does, so `import ovrtx` resolves to the
#                                    source tree (whose <pkg>/bin is absent, so
#                                    the loader falls through to LD_LIBRARY_PATH
#                                    and picks up the freshly built .so).
#
# Usage:  source ovrtx-env.sh        # reads $OVRTX_SOURCE
#         OVRTX_SOURCE=local source ovrtx-env.sh
#
# Override the kit checkout location with KIT_ROOT if it is not the sibling
# default. Nothing here launches a process; it only exports env vars.

# Resolve the kit rendering checkout (holds the ovrtx/ovstage source + _build).
# Default: a workspace sibling ``kit`` next to this IsaacLab checkout (this file
# lives at scripts/benchmarks/render_bench/, i.e. four levels below the workspace).
: "${KIT_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)/kit}"
_rendering="${KIT_ROOT}/rendering"
_ovrtx_py="${_rendering}/ovrtx/public/python"
_ovstage_py="${_rendering}/ovstage/public/python"
_build_root="${_rendering}/_build/linux-x86_64/release"

case "${OVRTX_SOURCE:-wheel}" in
    local)
        if [ ! -f "${_build_root}/libovrtx-dynamic.so" ]; then
            echo "ovrtx-env: OVRTX_SOURCE=local but ${_build_root}/libovrtx-dynamic.so is missing." >&2
            echo "ovrtx-env: build it first:  cd ${_rendering} && ./build.sh -r --no-docker --devrtx" >&2
            return 1 2>/dev/null || exit 1
        fi
        export PYTHONPATH="${_ovrtx_py}:${_ovstage_py}${PYTHONPATH:+:${PYTHONPATH}}"
        export LD_LIBRARY_PATH="${_build_root}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
        # Belt-and-suspenders: an explicit hint the ovrtx loader also honors.
        export OVRTX_LIBRARY_PATH_HINT="${_build_root}"
        echo "ovrtx-env: LOCAL build -> ${_build_root}"
        echo "ovrtx-env:   PYTHONPATH prepended: ${_ovrtx_py}"
        ;;
    wheel|"")
        echo "ovrtx-env: WHEEL (venv-installed ovrtx; no overrides)"
        ;;
    *)
        echo "ovrtx-env: unknown OVRTX_SOURCE='${OVRTX_SOURCE}' (expected 'local' or 'wheel')" >&2
        return 1 2>/dev/null || exit 1
        ;;
esac
