"""
Optional GPU-accelerated evaluation for CTRNN and Izhikevich spiking networks.

This module requires CuPy. Install via: pip install 'neat-python[gpu]'

All CuPy imports are lazy — ``import neat`` never triggers a CuPy import.
"""


def _import_cupy():
    """Import and return the CuPy module, or raise an informative error."""
    try:
        import cupy
        return cupy
    except ImportError:
        raise ImportError(
            "CuPy is required for GPU evaluation but is not installed.\n"
            "Install it with: pip install 'neat-python[gpu]'\n"
            "Or install CuPy directly: pip install cupy-cuda12x"
        ) from None


def _import_numpy():
    """Import and return NumPy, or raise an informative error."""
    try:
        import numpy
        return numpy
    except ImportError:
        raise ImportError(
            "NumPy is required for GPU evaluation but is not installed.\n"
            "Install it with: pip install numpy"
        ) from None


def gpu_available():
    """Return True if CuPy is installed and a GPU device is accessible."""
    try:
        cp = _import_cupy()
        cp.cuda.Device(0).compute_capability
        return True
    except Exception:
        return False
