"""ONNX-only fixed stem convolution order without expensive eager FMA replay.

Native forward is original torch.conv2d passthrough only. Its output does not
qualify the exported FMA order; actual ONNX must be evaluated independently.
"""
import torch
from torch.onnx import symbolic_helper

from .compensated_conv import validate_native_stem_conv
from .static_block192_onnx import fma_graph


class _StaticStemConv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, weights):
        return torch.nn.functional.conv2d(values, weights, stride=2, padding=1)

    @staticmethod
    def symbolic(g, values, weights):
        shape, coefficient_shape = (symbolic_helper._get_tensor_sizes(x) for x in (values, weights))
        if (shape is None or coefficient_shape is None or len(shape) != 4 or len(coefficient_shape) != 4
                or any(type(x) is not int or x < 1 for x in (*shape, *coefficient_shape))
                or coefficient_shape[2:] != [3, 3] or shape[1] != coefficient_shape[1]
                or (shape[1] != 1 and shape[1] % 16)):
            raise ValueError('Static stem export requires fixed matching Cin1 or16-block3x3 tensors')
        batch, channels, height, width = shape
        outputs = coefficient_shape[0]
        out_height, out_width = (height+1)//2, (width+1)//2
        locations = out_height * out_width
        chunk = 1 if channels == 1 else 16
        blocks = channels // chunk

        def ints(value): return g.op('Constant', value_t=torch.tensor(value, dtype=torch.int64))

        zero = g.op('Constant', value_t=torch.tensor(0.0, dtype=torch.float32))
        padded = g.op('Pad', values, ints([0, 0, 1, 1, 0, 0, 1, 1]), zero, mode_s='constant')
        coefficients = g.op('Reshape', weights, ints([outputs, blocks, chunk, 9]))
        partial = g.op('ConstantOfShape', ints([batch, blocks, locations, 1]), value_t=torch.zeros(1, dtype=torch.float32))
        splitter = g.op('Constant', value_t=torch.tensor(4097.0, dtype=torch.float32))
        for kernel in range(9):
            row, column = divmod(kernel, 3)
            plane = g.op('Slice', padded, ints([row, column]),
                         ints([row+2*out_height, column+2*out_width]), ints([2, 3]), ints([2, 2]))
            plane = g.op('Transpose', g.op('Reshape', plane, ints([batch, blocks, chunk, locations])), perm_i=[0, 1, 3, 2])
            coefficient = g.op('Transpose', g.op('Gather', coefficients, ints(kernel), axis_i=3), perm_i=[1, 2, 0])
            for channel in range(chunk):
                a = g.op('Slice', plane, ints([channel]), ints([channel+1]), ints([3]))
                b = g.op('Reshape', g.op('Gather', coefficient, ints(channel), axis_i=1), ints([1, blocks, 1, outputs]))
                partial = fma_graph(g, a, b, partial, splitter)
        total = g.op('Gather', partial, ints(0), axis_i=1)
        for block in range(1, blocks):
            total = g.op('Add', total, g.op('Gather', partial, ints(block), axis_i=1))
        output = g.op('Transpose', g.op('Reshape', total, ints([batch, out_height, out_width, outputs])), perm_i=[0, 3, 1, 2])
        return output.setType(values.type().with_sizes([batch, outputs, out_height, out_width]))


def static_stem_conv_export(values, weights):
    validate_native_stem_conv(values, weights)
    return _StaticStemConv.apply(values, weights)
