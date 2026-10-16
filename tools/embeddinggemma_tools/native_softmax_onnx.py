"""Opt-in FP32/ONNX port of the pinned audio eight-lane GPU Softmax.

Exp expression adapted from SLEEF 5a1d179 xexpf, copyright Naoki Shibata
and contributors 2010 - 2024, Boost Software License 1.0. See
Packages/com.ayutaz.embeddinggemma/SLEEF-LICENSE.txt. The original expression
and reduction order are in EmbeddingGemmaNativeSoftmax.compute.
This standalone diagnostic does not replace default export or qualify a model.
"""
import torch

from .export import fp32_fma


def _nonpositive_exp(values):
    # Softmax subtracts a finite maximum, so the exponent is nonpositive.
    # Keep the unused polynomial branch bounded for masks and extreme tails.
    core = values.clamp(-104, 0)
    q = torch.round(core * 1.4426950408889634073599246810018921374)
    reduced = fp32_fma(q, torch.full_like(core, -0.693145751953125), core)
    reduced = fp32_fma(q, torch.full_like(core, -1.428606765330187045e-6), reduced)
    polynomial = torch.full_like(core, 0.000198527617612853646278381)
    for coefficient in (0.00139304355252534151077271, 0.00833336077630519866943359,
                        0.0416664853692054748535156, 0.166666671633720397949219, 0.5):
        polynomial = fp32_fma(polynomial, reduced, torch.full_like(core, coefficient))
    polynomial = 1.0 + fp32_fma(reduced * reduced, polynomial, reduced)
    # q <= 0, so ceil(q/2) matches the original integer truncation toward0.
    first = torch.ceil(q * 0.5)
    two = torch.full_like(core, 2.0)
    result = (polynomial * torch.pow(two, first)) * torch.pow(two, q - first)
    return torch.where(values < -104, torch.zeros_like(values), result)


def fp32_native_softmax24(values):
    if not torch.compiler.is_compiling() and not torch.jit.is_tracing():
        if (values.dtype != torch.float32 or values.device.type != 'cpu' or not values.numel()
                or values.ndim < 1 or values.shape[-1] != 24 or torch.isnan(values).any()
                or torch.isposinf(values).any() or not torch.isfinite(values).any(dim=-1).all()):
            raise ValueError('Native Softmax diagnostic requires CPU FP32 width24 with a finite maximum per row')
    exponentials = _nonpositive_exp(values - values.amax(dim=-1, keepdim=True))
    lanes = (exponentials[..., :8] + exponentials[..., 8:16]) + exponentials[..., 16:24]
    left = (lanes[..., 0:1] + lanes[..., 4:5]) + (lanes[..., 2:3] + lanes[..., 6:7])
    right = (lanes[..., 1:2] + lanes[..., 5:6]) + (lanes[..., 3:4] + lanes[..., 7:8])
    # CPU/ONNX FP32 division is already rounded. The GPU shader refines its
    # approximate reciprocal; its independent port needs no approximate divide.
    inverse = 1.0 / (left + right)
    return exponentials * inverse
