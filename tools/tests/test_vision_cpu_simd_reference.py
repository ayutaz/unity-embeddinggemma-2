"""A fixed sixteen-lane reference order, not an AVX512 device or model pass."""
import importlib
import numpy as np
import pytest
import torch


def scalar_mean(values):
    width=values.shape[-1];out=[]
    for row in values.reshape(-1,width):
        partial=np.zeros((4,16),np.float32)
        for index in range(width//64):
            for accumulator in range(4):
                for lane in range(16):
                    k=index*64+accumulator*16+lane
                    partial[accumulator,lane]=np.float32(partial[accumulator,lane]+row[k])
        total=np.float32(0)
        for lane in range(16):
            value=partial[0,lane]
            for accumulator in range(1,4):value=np.float32(value+partial[accumulator,lane])
            total=np.float32(total+value)
        out.append(np.float32(total/np.float32(width)))
    return np.array(out,np.float32).reshape(*values.shape[:-1],1)


@pytest.mark.parametrize('shape',[(3,64),(2,768),(1,2,64),(1,2,768)])
def test_sixteen_lane_order_matches_independent_scalar_source_oracle(shape):
    module=importlib.import_module('embeddinggemma_tools.vision_cpu_simd_reference')
    rng=np.random.default_rng(sum(shape))
    values=(rng.normal(size=shape)*2.**rng.integers(-8,8,size=shape)).astype(np.float32)
    before=values.tobytes();actual=module.mean16(torch.from_numpy(values))
    np.testing.assert_array_equal(actual.numpy().view('u4'),scalar_mean(values).view('u4'))
    assert values.tobytes()==before and actual.dtype==torch.float32


@pytest.mark.parametrize('width,scaled',[(64,True),(64,False),(768,True),(768,False)])
def test_RMS_borrows_original_scale_and_keeps_exact_mean_order(width,scaled):
    module=importlib.import_module('embeddinggemma_tools.vision_cpu_simd_reference')
    values=torch.linspace(-2,3,3*width).reshape(3,width)
    scale=torch.nn.Parameter(torch.linspace(.5,1.5,width)) if scaled else None
    reference=module.VisionCpuSimdRms(width,scale,1e-6)
    before=None if scale is None else scale.detach().clone()
    with torch.no_grad():actual=reference(values)
    squared=values.numpy()*values.numpy();mean=scalar_mean(squared)
    wanted=values.numpy()*(mean+np.float32(1e-6))**np.float32(-.5)
    if scaled:wanted=wanted*scale.detach().numpy()
    np.testing.assert_allclose(actual.numpy(),wanted,rtol=2e-7,atol=2e-7)
    assert reference.scale is scale and (scale is None or torch.equal(scale,before))


@pytest.mark.parametrize('fault',['width','dtype','nan','rank'])
def test_unqualified_statistics_are_rejected(fault):
    module=importlib.import_module('embeddinggemma_tools.vision_cpu_simd_reference')
    values=torch.ones(2,64)
    if fault=='width':values=values[:,:32]
    elif fault=='dtype':values=values.double()
    elif fault=='nan':values[0,0]=float('nan')
    else:values=values[0]
    with pytest.raises(ValueError):module.mean16(values)
