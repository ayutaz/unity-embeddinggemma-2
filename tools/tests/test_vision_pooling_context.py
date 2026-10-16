import numpy as np
import pytest
import torch


def layouts(*args):
    from embeddinggemma_tools import vision_pooling_context
    return vision_pooling_context.pooling_layouts(*args)


def fixture():
    positions = torch.tensor([[[x,y] for y in range(3) for x in range(3)]+[[-1,-1]]*9])
    hidden = torch.arange(36,dtype=torch.float32).reshape(1,18,2)
    weights = torch.zeros(1,1,18)
    weights[:,:,:9] = 1/9
    return hidden,positions,weights


def test_pooling_layout_probe_matches_official_padding_mask_and_preserves_input_weights():
    hidden,positions,weights=fixture()
    before=[value.clone() for value in (hidden,positions,weights)]
    outputs,metadata=layouts(hidden,positions,weights)
    assert metadata['official_pool_shape']==[1,2,18]
    assert metadata['saved_pool_shape']==[1,1,18]
    target=hidden[:,:9].mean(1)*(2**.5)
    for actual in outputs.values():
        assert actual.shape==(1,2) and actual.dtype==torch.float32
        np.testing.assert_allclose(actual.numpy(),target.numpy(),rtol=1e-6,atol=1e-6)
    assert all(torch.equal(actual,expected) for actual,expected in zip((hidden,positions,weights),before,strict=True))
    assert metadata['official_pool_stride'][-2]==1
    assert metadata['saved_pool_stride'][-1]==1


@pytest.mark.parametrize('failure',['dtype','rank','finite'])
def test_pooling_probe_rejects_unproven_inputs(failure):
    hidden,positions,weights=fixture()
    if failure=='dtype':hidden=hidden.half()
    elif failure=='rank':hidden=hidden[0]
    else:hidden[0,0,0]=float('nan')
    with pytest.raises(ValueError):layouts(hidden,positions,weights)
