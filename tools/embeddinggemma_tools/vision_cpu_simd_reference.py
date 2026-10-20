"""Explicit sixteen-lane CPU reference for the observed AVX512 image context.

This portable FP32 order does not require AVX512 instructions and never selects
itself from hardware. It adds no Unity runtime or changed model coefficients.
For widths64/768, row_sum's four accumulators need fewer than16 iterations:
https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/cpu/SumKernel.cpp
"""
import torch


def mean16(values):
    if (not isinstance(values,torch.Tensor) or values.device.type!='cpu' or values.dtype!=torch.float32
            or values.ndim<2 or values.shape[-1] not in (64,768) or not values.numel() or not torch.isfinite(values).all()):
        raise ValueError('The explicit image reference requires finite CPU FP32 width64 or768 rows')
    width=values.shape[-1]
    grouped=values.contiguous().reshape(-1,width//64,4,16)
    partial=torch.zeros_like(grouped[:,0])
    for index in range(width//64):partial=partial+grouped[:,index]
    lanes=partial[:,0]
    for index in range(1,4):lanes=lanes+partial[:,index]
    total=torch.zeros_like(lanes[:,:1])
    for index in range(16):total=total+lanes[:,index:index+1]
    return (total/width).reshape(*values.shape[:-1],1)


class VisionCpuSimdRms(torch.nn.Module):
    """Borrow original scale and epsilon; change only the declared sum order."""
    def __init__(self,width,scale,epsilon):
        super().__init__()
        if width not in (64,768) or epsilon!=1e-6:
            raise ValueError('An original image RMS width and epsilon are required')
        if scale is not None and (not isinstance(scale,torch.Tensor) or scale.device.type!='cpu'
                or scale.dtype!=torch.float32 or tuple(scale.shape)!=(width,) or not torch.isfinite(scale).all()):
            raise ValueError('The original finite CPU FP32 image scale is required')
        self.width=width;self.epsilon=epsilon;self.register_buffer('scale',scale)

    def forward(self,values):
        if not isinstance(values,torch.Tensor) or values.shape[-1]!=self.width:
            raise ValueError('The declared image RMS width differs')
        statistic=mean16(values.square())+self.epsilon
        result=values*torch.pow(statistic,-.5)
        return result if self.scale is None else result*self.scale
