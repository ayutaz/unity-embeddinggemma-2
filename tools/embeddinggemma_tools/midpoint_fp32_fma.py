"""Bounded normal FP32 FMA with a retained midpoint sticky residual.

Only the explicit query-key midpoint candidate uses this in connected exports.
Like the original Dekker helper, it excludes overflow and product underflow.
There are no FP64 operations, reinterpret casts or changed model coefficients.
"""
import torch
from .static_block192_onnx import _StaticBlock192Bmm, validate_batched_matrices, static_block192_bmm_export


def _two_sum(a,b):
    high=a+b
    recovered=high-a
    return high,(a-(high-recovered))+(b-recovered)


def fp32_midpoint_fma(a,b,c):
    product=a*b
    a_split,b_split=a*4097.0,b*4097.0
    a_high,b_high=a_split-(a_split-a),b_split-(b_split-b)
    a_low,b_low=a-a_high,b-b_high
    product_error=((a_high*b_high-product)+a_high*b_low+a_low*b_high)+a_low*b_low
    total,sum_error=_two_sum(product,c)
    error,sticky=_two_sum(product_error,sum_error)

    # For noncancelling normal additions the combined residual is bounded by
    # one ulp(total). A lost low residual can affect rounding only at a half
    # step. For exact cancellation TwoSum's low addition is exact instead.
    magnitude=total.abs()
    half_scale=magnitude*(2.0**-24)
    upward=(magnitude+half_scale)-magnitude
    upward=torch.where(upward==0,half_scale+half_scale,upward)
    downward=magnitude-(magnitude-half_scale)
    step=torch.where((total<0)==(error<0),upward,downward)
    step=torch.maximum(step,torch.full_like(step,2.0**-149))
    midpoint=(error.abs()==step*0.5)&(sticky!=0)
    # A one-residual-ulp nudge specifies the correct side of the midpoint.
    # It is used only when the high residual is exactly half the output step.
    nudge=error.abs()*(2.0**-23)
    nudge=torch.where(sticky<0,-nudge,nudge)
    corrected=torch.where(midpoint,error+nudge,error)
    return total+corrected


def midpoint_fma_graph(g,a,b,c,splitter):
    add=lambda x,y:g.op('Add',x,y)
    sub=lambda x,y:g.op('Sub',x,y)
    mul=lambda x,y:g.op('Mul',x,y)
    absolute=lambda x:g.op('Abs',x)
    constant=lambda x:g.op('Constant',value_t=torch.tensor(x,dtype=torch.float32))
    def two_sum(x,y):
        high=add(x,y);recovered=sub(high,x)
        return high,add(sub(x,sub(high,recovered)),sub(y,recovered))
    product=mul(a,b)
    a_split,b_split=mul(a,splitter),mul(b,splitter)
    a_high,b_high=sub(a_split,sub(a_split,a)),sub(b_split,sub(b_split,b))
    a_low,b_low=sub(a,a_high),sub(b,b_high)
    error=add(add(add(sub(mul(a_high,b_high),product),mul(a_high,b_low)),mul(a_low,b_high)),mul(a_low,b_low))
    total,sum_error=two_sum(product,c)
    residual,sticky=two_sum(error,sum_error)
    magnitude=absolute(total);half_scale=mul(magnitude,constant(2.**-24))
    zero=constant(0.)
    upward=sub(add(magnitude,half_scale),magnitude)
    upward=g.op('Where',g.op('Equal',upward,zero),add(half_scale,half_scale),upward)
    downward=sub(magnitude,sub(magnitude,half_scale))
    same_sign=g.op('Equal',g.op('Less',total,zero),g.op('Less',residual,zero))
    step=g.op('Max',g.op('Where',same_sign,upward,downward),constant(2.**-149))
    midpoint=g.op('And',g.op('Equal',absolute(residual),mul(step,constant(.5))),g.op('Not',g.op('Equal',sticky,zero)))
    nudge=mul(absolute(residual),constant(2.**-23))
    nudge=g.op('Where',g.op('Less',sticky,zero),g.op('Neg',nudge),nudge)
    return add(total,g.op('Where',midpoint,add(residual,nudge),residual))


class _StaticMidpointBmm(_StaticBlock192Bmm):
    @staticmethod
    def symbolic(g,values,weights):
        return _StaticBlock192Bmm.graph(g,values,weights,fma=midpoint_fma_graph)


def static_midpoint_bmm_export(values,weights):
    validate_batched_matrices(values,weights)
    return _StaticMidpointBmm.apply(values,weights)


def midpoint_query_key_bmm_export(values,weights):
    if tuple(values.shape)==(192,12,128) and tuple(weights.shape)==(192,128,24):
        return static_midpoint_bmm_export(values,weights)
    return static_block192_bmm_export(values,weights)
