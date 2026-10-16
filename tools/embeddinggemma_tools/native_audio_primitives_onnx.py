"""Opt-in standard-FP32 GLU and depthwise candidates, not model acceptance.

GLU Exp follows EmbeddingGemmaNativeGlu.compute / SLEEF 5a1d179 xexpf,
copyright Naoki Shibata and contributors 2010 - 2024, BSL-1.0. See
Packages/com.ayutaz.embeddinggemma/SLEEF-LICENSE.txt.
"""
import torch

from .export import fp32_fma


def fp32_native_glu(values):
    if not torch.compiler.is_compiling() and not torch.jit.is_tracing():
        if (values.dtype != torch.float32 or values.device.type != 'cpu' or values.ndim < 1
                or not values.numel() or values.shape[-1] % 2 or not torch.isfinite(values).all()):
            raise ValueError('Native GLU diagnostic requires finite CPU FP32 even-width tensors')
    left, gate = values.chunk(2, dim=-1)
    return left * fp32_native_sigmoid(gate)


def fp32_native_sigmoid(values):
    if not torch.compiler.is_compiling() and not torch.jit.is_tracing():
        if (values.dtype != torch.float32 or values.device.type != 'cpu'
                or not values.numel() or not torch.isfinite(values).all()):
            raise ValueError('Native sigmoid diagnostic requires nonempty finite CPU FP32 values')
    core = (-values).clamp(-80, 80)
    q = torch.round(core * 1.4426950408889634073599246810018921374)
    reduced = fp32_fma(q, torch.full_like(core, -0.693145751953125), core)
    reduced = fp32_fma(q, torch.full_like(core, -1.428606765330187045e-6), reduced)
    polynomial = torch.full_like(core, 0.000198527617612853646278381)
    for coefficient in (0.00139304355252534151077271, 0.00833336077630519866943359,
                        0.0416664853692054748535156, 0.166666671633720397949219, 0.5):
        polynomial = fp32_fma(polynomial, reduced, torch.full_like(core, coefficient))
    polynomial = 1.0 + fp32_fma(reduced * reduced, polynomial, reduced)
    exponential = polynomial * torch.pow(torch.full_like(core, 2.0), q)
    # Retain original mathematical tails; only the unused polynomial is clipped.
    exponential = torch.where(values.abs() > 80, torch.exp(-values), exponential)
    inverse = 1.0 / (1.0 + exponential)
    return inverse


def fp32_native_depthwise5(values, weights):
    if not torch.compiler.is_compiling() and not torch.jit.is_tracing():
        if (values.dtype != torch.float32 or weights.dtype != torch.float32
                or values.device.type != 'cpu' or weights.device.type != 'cpu'
                or values.ndim != 3 or weights.ndim != 3 or tuple(weights.shape[1:]) != (1, 5)
                or values.shape[1] != weights.shape[0] or values.shape[2] < 5
                or not values.numel() or not weights.numel()
                or not torch.isfinite(values).all() or not torch.isfinite(weights).all()
                or torch.any(values.abs() > 1e6) or torch.any(weights.abs() > 100)):
            raise ValueError('Depthwise diagnostic requires bounded CPU FP32 independent channels /kernel5')
    length = values.shape[2] - 4
    total = torch.zeros_like(values[:, :, :length])
    for index in range(5):
        total = fp32_fma(values[:, :, index:index + length], weights[:, 0, index][None, :, None], total)
    return total
