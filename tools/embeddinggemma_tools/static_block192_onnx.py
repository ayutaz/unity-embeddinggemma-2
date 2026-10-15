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
        return _StaticBlock192.graph(g, values, weights)

    @staticmethod
    def graph(g, values, weights, bias=None):
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
        if bias is not None:
            total = g.op("Add", total, bias)
        for index in range(1, blocks):
            total = g.op("Add", total, g.op("Gather", partial, ints(index), axis_i=0))
        return total.setType(values.type().with_sizes([rows, columns]))


def validate_matrices(values, weights):
    if (values.dtype != torch.float32 or weights.dtype != torch.float32
            or values.device.type != "cpu" or weights.device.type != "cpu"
            or values.ndim != 2 or weights.ndim != 2 or values.shape[1] != weights.shape[0]):
        raise ValueError("Static block192 export requires matching CPU FP32 matrices")
    if not torch.jit.is_tracing() and not torch.compiler.is_compiling():
        if (not values.numel() or not weights.numel() or not torch.isfinite(values).all()
                or not torch.isfinite(weights).all() or torch.any(values.abs() > 1e6)
                or torch.any(weights.abs() > 100)):
            raise ValueError("Static block192 export requires qualified finite bounded matrices")


def static_block192_export(values, weights):
    validate_matrices(values, weights)
    return _StaticBlock192.apply(values, weights)


class _StaticBlock192Bias(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bias, values, weights):
        return torch.addmm(bias, values, weights)

    @staticmethod
    def symbolic(g, bias, values, weights):
        return _StaticBlock192.graph(g, values, weights, bias)


def static_block192_addmm_export(bias, values, weights):
    validate_matrices(values, weights)
    if (bias.dtype != torch.float32 or bias.device.type != "cpu" or bias.ndim != 1
            or bias.shape[0] != weights.shape[1]):
        raise ValueError("Static biased export requires the original matching CPU FP32 bias vector")
    if not torch.jit.is_tracing() and not torch.compiler.is_compiling():
        if not torch.isfinite(bias).all() or torch.any(bias.abs() > 1e6):
            raise ValueError("Static biased export requires a finite bounded bias")
    return _StaticBlock192Bias.apply(bias, values, weights)


class _StaticBlock192Bmm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, weights):
        return torch.bmm(values, weights)

    @staticmethod
    def symbolic(g, values, weights):
        left, right = (symbolic_helper._get_tensor_sizes(x) for x in (values, weights))
        if (left is None or right is None or len(left) != 3 or len(right) != 3
                or any(type(d) is not int or d < 1 for d in (*left, *right))
                or left[0] != right[0] or left[2] != right[1]):
            raise ValueError("Static block192 BMM requires fixed positive batched matrix shapes")
        batch, rows, width, columns = left[0], left[1], left[2], right[2]
        chunk = min(192, width)
        blocks = (width + chunk - 1) // chunk
        padding = blocks * chunk - width

        def ints(value):
            return g.op("Constant", value_t=torch.tensor(value, dtype=torch.int64))

        zero = g.op("Constant", value_t=torch.tensor(0.0, dtype=torch.float32))
        if padding:
            values = g.op("Pad", values, ints([0, 0, 0, 0, 0, padding]), zero, mode_s="constant")
            weights = g.op("Pad", weights, ints([0, 0, 0, 0, padding, 0]), zero, mode_s="constant")
        chunks = g.op("Transpose", g.op("Reshape", values, ints([batch, rows, blocks, chunk])), perm_i=[2, 0, 1, 3])
        coefficients = g.op("Transpose", g.op("Reshape", weights, ints([batch, blocks, chunk, columns])), perm_i=[1, 0, 2, 3])
        partial = g.op("ConstantOfShape", ints([blocks, batch, rows, 1]), value_t=torch.zeros(1, dtype=torch.float32))
        splitter = g.op("Constant", value_t=torch.tensor(4097.0, dtype=torch.float32))
        axis = ints([3])
        for index in range(chunk):
            a = g.op("Slice", chunks, ints([index]), ints([index + 1]), axis)
            b = g.op("Reshape", g.op("Gather", coefficients, ints(index), axis_i=2), ints([blocks, batch, 1, columns]))
            partial = fma_graph(g, a, b, partial, splitter)
        total = g.op("Gather", partial, ints(0), axis_i=0)
        for index in range(1, blocks):
            total = g.op("Add", total, g.op("Gather", partial, ints(index), axis_i=0))
        return total.setType(values.type().with_sizes([batch, rows, columns]))


def static_block192_bmm_export(values, weights):
    if (values.dtype != torch.float32 or weights.dtype != torch.float32
            or values.device.type != "cpu" or weights.device.type != "cpu"
            or values.ndim != 3 or weights.ndim != 3 or values.shape[0] != weights.shape[0]
            or values.shape[2] != weights.shape[1]):
        raise ValueError("Static block192 BMM requires matching CPU FP32 batched matrices")
    if not torch.jit.is_tracing() and not torch.compiler.is_compiling():
        if (not values.numel() or not weights.numel() or not torch.isfinite(values).all()
                or not torch.isfinite(weights).all() or torch.any(values.abs() > 1e6)
                or torch.any(weights.abs() > 1e6)):
            raise ValueError("Static block192 BMM requires qualified finite bounded matrices")
    return _StaticBlock192Bmm.apply(values, weights)


def audit_static_linear_nodes(segment, *, rows=280):
    def parameter_matrix(node):
        if node.op == "get_attr":
            return True
        return (node.op == "call_function" and node.target == torch.ops.aten.permute.default
                and len(node.args) == 2 and tuple(node.args[1]) == (1, 0)
                and isinstance(node.args[0], torch.fx.Node) and node.args[0].op == "get_attr")
    selected = set()
    for node in segment.graph.nodes:
        if node.op != "call_function" or node.target not in (torch.ops.aten.mm.default, torch.ops.aten.addmm.default):
            continue
        biased = node.target == torch.ops.aten.addmm.default
        if node.kwargs or len(node.args) != (3 if biased else 2):
            continue
        left, right = node.args[-2:]
        a, b = (value.meta.get("val") for value in (left, right))
        if (not all(isinstance(value, torch.Tensor) and value.dtype == torch.float32 and value.ndim == 2
                    for value in (a, b)) or a.shape[0] != rows or a.shape[1] != b.shape[0]
                or not parameter_matrix(right)):
            continue
        if biased:
            bias = node.args[0]
            value = bias.meta.get("val")
            if (bias.op != "get_attr" or not isinstance(value, torch.Tensor) or value.dtype != torch.float32
                    or tuple(value.shape) != (b.shape[1],)):
                continue
        selected.add(node)
    return selected


def audit_relative_position_linear_nodes(segment):
    return {node for node in audit_static_linear_nodes(segment, rows=13)
            if node.target == torch.ops.aten.mm.default
            and tuple(node.args[0].meta['val'].shape) == (13, 1024)
            and tuple(node.args[1].meta['val'].shape) == (1024, 1024)}


def audit_attention_bmm_nodes(segment):
    # These three shape pairs have actual ONNX evidence against the original
    # qualified layer0/1 captures. The 1x280x1024 projection is not qualified here.
    qualified = {((192, 12, 128), (192, 128, 24)),
                 ((8, 288, 128), (8, 128, 13)),
                 ((192, 12, 24), (192, 24, 128))}
    selected = set()
    for node in segment.graph.nodes:
        if (node.op != "call_function" or node.target != torch.ops.aten.bmm.default
                or node.kwargs or len(node.args) != 2):
            continue
        values = [arg.meta.get("val") for arg in node.args]
        if (all(isinstance(x, torch.Tensor) and x.dtype == torch.float32 and x.device.type == "cpu" for x in values)
                and tuple(tuple(x.shape) for x in values) in qualified):
            selected.add(node)
    return selected


def audit_parameter_bmm_nodes(segment):
    def parameter_view(node):
        if node.op == "get_attr":
            return True
        if node.op != "call_function" or node.kwargs or len(node.args) != 2:
            return False
        if node.target == torch.ops.aten.permute.default:
            valid = tuple(node.args[1]) == (1, 0)
        elif node.target in (torch.ops.aten.view.default, torch.ops.aten.expand.default):
            valid = tuple(node.args[1]) == (1, 1024, 1024)
        else:
            return False
        return valid and isinstance(node.args[0], torch.fx.Node) and parameter_view(node.args[0])

    selected = set()
    for node in segment.graph.nodes:
        if (node.op != "call_function" or node.target != torch.ops.aten.bmm.default
                or node.kwargs or len(node.args) != 2):
            continue
        values = [arg.meta.get("val") for arg in node.args]
        if (all(isinstance(x, torch.Tensor) and x.dtype == torch.float32 and x.device.type == "cpu" for x in values)
                and tuple(tuple(x.shape) for x in values) == ((1, 280, 1024), (1, 1024, 1024))
                and parameter_view(node.args[1])):
            selected.add(node)
    return selected
