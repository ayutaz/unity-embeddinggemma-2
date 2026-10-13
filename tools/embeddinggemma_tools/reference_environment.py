"""Record numerical execution settings without changing the reference math."""

import platform

import torch


def cpu_math_environment():
    """Build and CPU dispatch may differ even with the same package versions."""
    return dict(
        system=platform.system(), machine=platform.machine(),
        torch_build_version=torch.__version__,
        cpu_capability=torch.backends.cpu.get_cpu_capability(),
        intraop_threads=torch.get_num_threads(),
        interop_threads=torch.get_num_interop_threads(),
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        mkl_available=torch.backends.mkl.is_available(),
        mkldnn_enabled=torch.backends.mkldnn.enabled,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        torch_build_configuration=torch.__config__.show(),
    )
