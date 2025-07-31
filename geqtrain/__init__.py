try:
    import torch
except ImportError:
    raise ImportError(
        "PyTorch is not installed. Please install it for your specific system (CPU/CUDA) "
        "by following the instructions at https://pytorch.org/get-started/locally/ "
        "before using GEqTrain."
    )

from ._version import __version__  # noqa: F401