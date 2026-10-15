"""Standard-FP32 SiLU candidate for explicit CPU/ONNX diagnosis only.

The expression follows Runtime/Resources/EmbeddingGemmaNativeSilu.compute.
Exp polynomial adapted from SLEEF 5a1d179, src/libm/sleefsimdsp.c xexpf.
Copyright Naoki Shibata and contributors 2010 - 2024; Boost Software License
1.0, included in Packages/com.ayutaz.embeddinggemma/SLEEF-LICENSE.txt.
This does not replace the default export or qualify a complete encoder/GPU.
"""

import torch

from .export import fp32_fma


def fp32_native_silu(values):
    if not torch.compiler.is_compiling() and not torch.jit.is_tracing():
        if (values.dtype != torch.float32 or values.device.type != "cpu" or not values.numel()
                or not torch.isfinite(values).all()):
            raise ValueError("SiLU candidate requires nonempty finite CPU FP32 inputs")
    # The existing shader qualifies its polynomial only inside [-80,80].
    # Clamp only the unused polynomial branch; preserve the shader's original
    # mathematical expression for actual out-of-range values, not a clipped SiLU.
    core = values.clamp(-80, 80)
    negative = -core
    q = torch.round(negative * 1.4426950408889634073599246810018921374)
    s = fp32_fma(q, torch.full_like(values, -0.693145751953125), negative)
    s = fp32_fma(q, torch.full_like(values, -1.428606765330187045e-6), s)
    u = torch.full_like(values, 0.000198527617612853646278381)
    for coefficient in (0.00139304355252534151077271, 0.00833336077630519866943359,
                        0.0416664853692054748535156, 0.166666671633720397949219, 0.5):
        u = fp32_fma(u, s, torch.full_like(values, coefficient))
    u = 1.0 + fp32_fma(s * s, u, s)
    exponential = u * torch.pow(torch.full_like(values, 2.0), q)
    # CPU / standard ONNX Div already rounds the division. The extra GPU
    # approximate-division refinement is unnecessary for this CPU candidate.
    result = core / (1.0 + exponential)
    tail = values / (1.0 + torch.exp(-values))
    selected = torch.where(values.abs() > 80, tail, result)
    return torch.where(values == 0, values, selected)
