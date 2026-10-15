"""Opt-in ONNX-only block192 lowering without eager replay of its FMA graph.

Native forward is the ORIGINAL MatMul passthrough, not an execution of the
fixed-order candidate. The exported graph must independently pass the original
reference. Lowering follows fp32_block192_matmul and its bounded Dekker/TwoSum
expression; it adds no FP64 operators and never changes the source weights.
"""

import torch
from torch.onnx import symbolic_helper


def fma_graph(g, a, b, c, splitter):
    mul = lambda x, y: g.op("Mul", x, y)
    add = lambda x, y: g.op("Add", x, y)
    sub = lambda x, y: g.op("Sub", x, y)
    product = mul(a, b)
    a_split, b_split = mul(a, splitter), mul(b, splitter)
    a_high, b_high = sub(a_split, sub(a_split, a)), sub(b_split, sub(b_split, b))
    a_low, b_low = sub(a, a_high), sub(b, b_high)
    error = add(add(add(sub(mul(a_high, b_high), product), mul(a_high, b_low)),
                    mul(a_low, b_high)), mul(a_low, b_low))
    total = add(product, c)
    recovered = sub(total, product)
    sum_error = add(sub(product, sub(total, recovered)), sub(c, recovered))
    return add(total, add(error, sum_error))


class _StaticBlock192(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, weights):
        # Avoid executing thousands of explicit FMA steps before/during export.
        # This native control proves passthrough only; ONNX qualification is separate.
        return torch.mm(values, weights)

    @staticmethod
    def symbolic(g, values, weights):
        left, right = (symbolic_helper._get_tensor_sizes(x) for x in (values, weights))
        if (left is None or right is None or len(left) != 2 or len(right) != 2
                or any(type(d) is not int or d < 1 for d in (*left, *right)) or left[1] != right[0]):
            raise ValueError("Static block192 lowering requires fixed positive matrix shapes")
        rows, width, columns = left[0], left[1], right[1]
        blocks = (width + 191) // 192
        padding = blocks * 192 - width

        def ints(value):
            return g.op("Constant", value_t=torch.tensor(value, dtype=torch.int64))

        zero = g.op("Constant", value_t=torch.tensor(0.0, dtype=torch.float32))
        if padding:
            values = g.op("Pad", values, ints([0, 0, 0, padding]), zero, mode_s="constant")
            weights = g.op("Pad", weights, ints([0, 0, padding, 0]), zero, mode_s="constant")
        chunks = g.op("Transpose", g.op("Reshape", values, ints([rows, blocks, 192])), perm_i=[1, 0, 2])
        coefficients = g.op("Reshape", weights, ints([blocks, 192, columns]))
        partial = g.op("ConstantOfShape", ints([blocks, rows, 1]), value_t=torch.zeros(1, dtype=torch.float32))
        splitter = g.op("Constant", value_t=torch.tensor(4097.0, dtype=torch.float32))
        axis = ints([2])
        right_shape = ints([blocks, 1, columns])
        for index in range(192):
            a = g.op("Slice", chunks, ints([index]), ints([index + 1]), axis)
            b = g.op("Reshape", g.op("Gather", coefficients, ints(index), axis_i=1), right_shape)
            partial = fma_graph(g, a, b, partial, splitter)
        total = g.op("Gather", partial, ints(0), axis_i=0)
        for index in range(1, blocks):
            total = g.op("Add", total, g.op("Gather", partial, ints(index), axis_i=0))
        return total.setType(values.type().with_sizes([rows, columns]))


def static_block192_export(values, weights):
    if (values.dtype != torch.float32 or weights.dtype != torch.float32
            or values.device.type != "cpu" or weights.device.type != "cpu"
            or values.ndim != 2 or weights.ndim != 2 or values.shape[1] != weights.shape[0]):
        raise ValueError("Static block192 export requires matching CPU FP32 matrices")
    if not torch.jit.is_tracing() and not torch.compiler.is_compiling():
        if (not values.numel() or not weights.numel() or not torch.isfinite(values).all()
                or not torch.isfinite(weights).all() or torch.any(values.abs() > 1e6)
                or torch.any(weights.abs() > 100)):
            raise ValueError("Static block192 export requires qualified finite bounded matrices")
    return _StaticBlock192.apply(values, weights)
