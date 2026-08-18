
import ctypes
import glob
import os
import site
import sys

_KEEP = {}


def _report(message):
    path = os.environ.get("CUPTI_HOG_STATUS_FILE")
    if path:
        with open(path, "a") as handle:
            handle.write(message + "\n")


def _candidates():
    """CUPTI is not on the loader path at interpreter start, so locate the wheel copy."""
    override = os.environ.get("CUPTI_LIBRARY")
    if override:
        yield override
    roots = list(site.getsitepackages())
    roots.append(
        os.path.join(sys.prefix, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages")
    )
    for root in roots:
        for pattern in ("nvidia/cuda_cupti/lib/libcupti.so*", "triton/backends/nvidia/lib/cupti/libcupti.so*"):
            for path in sorted(glob.glob(os.path.join(root, pattern))):
                if not path.endswith(".debug"):
                    yield path
    yield from ("libcupti.so.12", "libcupti.so", "libcupti.so.13")


def _hog():
    for name in _candidates():
        try:
            lib = ctypes.CDLL(name)
        except OSError:
            continue
        cb = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p)(
            lambda *a: None
        )
        handle = ctypes.c_void_p()
        rc = lib.cuptiSubscribe(ctypes.byref(handle), cb, None)
        _report(f"cuptiSubscribe rc={rc} via {name}")
        if rc == 0:
            # Hold references so the subscription outlives this function.
            _KEEP["lib"], _KEEP["cb"], _KEEP["handle"] = lib, cb, handle
            return True
        return False
    _report("libcupti not loadable; NOT hogging (set CUPTI_LIBRARY)")
    return False


_hog()
