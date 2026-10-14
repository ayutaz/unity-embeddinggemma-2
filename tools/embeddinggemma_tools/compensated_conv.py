"""Opt-in standard FP32 unfolded convolution; original modules stay unchanged."""
import torch
import torch.nn.functional as F


def fp32_compensated_conv2d(values, weight, *, chunk_size=64):
    """3x3 / stride2 / pad1 convolution with compensated block products."""
    if type(chunk_size) is not int or chunk_size<=0:
        raise ValueError('Convolution chunk size must be a positive integer')
    if not torch.compiler.is_compiling() and not torch.jit.is_tracing():
        if (weight.dtype!=torch.float32 or weight.ndim!=4 or tuple(weight.shape[-2:])!=(3,3)
                or values.dtype!=torch.float32 or values.ndim!=4
                or values.device!=weight.device or values.shape[1]!=weight.shape[1]
                or not values.numel() or not weight.numel()
                or not torch.isfinite(values).all() or not torch.isfinite(weight).all()):
            raise ValueError('Convolution requires finite matching FP32 input and 3x3 weights')
    patches=F.unfold(values,kernel_size=3,padding=1,stride=2).transpose(1,2)
    flat=weight.reshape(weight.shape[0],-1)
    total=values.new_zeros((values.shape[0],patches.shape[1],weight.shape[0]))
    correction=torch.zeros_like(total)
    for start in range(0,flat.shape[1],chunk_size):
        part=F.linear(patches[...,start:start+chunk_size],flat[:,start:start+chunk_size])
        updated=total+part
        correction=correction+torch.where(total.abs()>=part.abs(),(total-updated)+part,(part-updated)+total)
        total=updated
    output=total+correction
    return output.transpose(1,2).reshape(values.shape[0],weight.shape[0],(values.shape[2]+1)//2,(values.shape[3]+1)//2)


class CompensatedConv2d(torch.nn.Module):
    """Borrow one unbiased FP32 Conv2d weight without changing its parameters."""
    def __init__(self,source,*,chunk_size=64):
        super().__init__()
        if (not isinstance(source,torch.nn.Conv2d) or source.bias is not None
                or source.kernel_size!=(3,3) or source.stride!=(2,2) or source.padding!=(1,1)
                or source.dilation!=(1,1) or source.groups!=1 or source.padding_mode!='zeros'
                or source.weight.dtype!=torch.float32 or not torch.isfinite(source.weight).all()
                or type(chunk_size) is not int or chunk_size<=0):
            raise ValueError('Candidate requires an unbiased finite FP32 3x3 / stride2 / pad1 convolution')
        self.weight=source.weight
        self.chunk_size=chunk_size

    def forward(self,values):
        return fp32_compensated_conv2d(values,self.weight,chunk_size=self.chunk_size)
