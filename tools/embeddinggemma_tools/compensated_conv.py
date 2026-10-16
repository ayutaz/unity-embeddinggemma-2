"""Opt-in standard FP32 unfolded convolution; original modules stay unchanged."""
import torch
import torch.nn.functional as F


def validate_native_stem_conv(values, weight):
    """Audit bounded CPU tensors before an explicit stem diagnostic."""
    if not torch.compiler.is_compiling() and not torch.jit.is_tracing():
        if (values.dtype != torch.float32 or weight.dtype != torch.float32
                or values.device.type != 'cpu' or weight.device.type != 'cpu'
                or values.ndim != 4 or weight.ndim != 4 or tuple(weight.shape[-2:]) != (3, 3)
                or values.shape[1] != weight.shape[1] or (weight.shape[1] != 1 and weight.shape[1] % 16)
                or not values.numel() or not weight.numel() or not torch.isfinite(values).all()
                or not torch.isfinite(weight).all() or torch.any(values.abs() > 1e6)
                or torch.any(weight.abs() > 100)):
            raise ValueError('Native stem convolution requires bounded CPU FP32 Cin1 or16-block3x3 tensors')


def fp32_native_stem_conv2d(values, weight):
    """Opt-in original kernel order: Cin1 FMA9 or existing Cin16-block order."""
    from .export import fp32_fma
    validate_native_stem_conv(values, weight)
    if weight.shape[1] != 1:
        return fp32_block16_conv2d(values, weight)
    patches = F.unfold(values, 3, padding=1, stride=2).transpose(1, 2)
    coefficients = weight.reshape(weight.shape[0], 9)
    total = values.new_zeros((values.shape[0], patches.shape[1], weight.shape[0]))
    for kernel in range(9):
        total = fp32_fma(patches[..., kernel:kernel+1], coefficients[:, kernel][None, None, :], total)
    return total.transpose(1, 2).reshape(values.shape[0], weight.shape[0], (values.shape[2]+1)//2, (values.shape[3]+1)//2)


def fp32_block16_conv2d(values,weight):
    """Pinned oneDNN AVX512 block16 FMA/reset/store order, standard FP32 only.

    Each input-channel block starts at zero. Kernel positions then channels are
    fused in order; the rounded block is added to the previously stored result.
    Uses the bounded-normal FMA candidate, not general IEEE exceptional values.
    Source: oneDNN 80afa710... jit_avx512_common_conv_kernel.cpp,
    prepare_output / compute_loop_fma_core / store_output.
    """
    from .export import fp32_fma
    if not torch.compiler.is_compiling() and not torch.jit.is_tracing():
        if (values.dtype!=torch.float32 or weight.dtype!=torch.float32 or values.ndim!=4 or weight.ndim!=4
                or tuple(weight.shape[-2:])!=(3,3) or values.shape[1]!=weight.shape[1]
                or weight.shape[1]%16 or not values.numel() or not weight.numel()
                or values.device!=weight.device or not torch.isfinite(values).all() or not torch.isfinite(weight).all()
                or torch.any(values.abs()>1e6) or torch.any(weight.abs()>100)):
            raise ValueError('Pinned block16 convolution requires bounded finite matching FP32 3x3 tensors')
    blocks=weight.shape[1]//16
    patches=F.unfold(values,3,padding=1,stride=2).transpose(1,2)
    patches=patches.reshape(values.shape[0],-1,blocks,16,9)
    coefficients=weight.reshape(weight.shape[0],blocks,16,9).permute(1,0,2,3)
    partial=values.new_zeros((values.shape[0],patches.shape[1],blocks,weight.shape[0]))
    for kernel in range(9):
        for channel in range(16):
            a=patches[...,channel,kernel].unsqueeze(-1)
            b=coefficients[...,channel,kernel].unsqueeze(0).unsqueeze(0)
            partial=fp32_fma(a,b,partial)
    total=partial[:,:,0,:]
    for block in range(1,blocks):total=partial[:,:,block,:]+total
    return total.transpose(1,2).reshape(values.shape[0],weight.shape[0],(values.shape[2]+1)//2,(values.shape[3]+1)//2)


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
