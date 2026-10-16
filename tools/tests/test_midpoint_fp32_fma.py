from fractions import Fraction

import numpy as np
import torch


def candidate():
    from embeddinggemma_tools import midpoint_fp32_fma
    return midpoint_fp32_fma.fp32_midpoint_fma


def exact_oracle(a,b,c):
    outputs=[]
    for x,y,z in zip(a.numpy().ravel(),b.numpy().ravel(),c.numpy().ravel(),strict=True):
        exact=Fraction.from_float(float(x))*Fraction.from_float(float(y))+Fraction.from_float(float(z))
        near=np.float32(float(exact))
        neighbors=(near,np.nextafter(near,np.float32(-np.inf)),np.nextafter(near,np.float32(np.inf)))
        chosen=min(neighbors,key=lambda v:(abs(Fraction.from_float(float(v))-exact),int(v.view('u4'))&1))
        outputs.append(chosen)
    return np.asarray(outputs,dtype=np.float32).reshape(a.shape)


def fixtures():
    rows=[]
    for exponent in (-20,-4,0,7,20):
        unit=2.**(-12+exponent)
        for sign in (-1,1):
            rows.extend(((sign*(1+2.**-23)*unit,(1-2.**-24)*unit,sign*2.**(2*exponent)),
                         (sign*(1+2.**-23)*unit,(1-2.**-23)*unit,sign*(1+2.**-23)*2.**(2*exponent)),
                         (sign*(1+2.**-23)*unit*.5,(1-2.**-23)*unit,sign*(1-2.**-24)*2.**(2*exponent))))
    return tuple(torch.tensor(v,dtype=torch.float32) for v in zip(*rows))


def test_midpoint_sticky_residual_preserves_true_single_rounding_for_both_signs_and_exponents():
    args=fixtures();actual=candidate()(*args).numpy();expected=exact_oracle(*args)
    np.testing.assert_array_equal(actual.view('u4'),expected.view('u4'))


def test_matches_independent_exact_rational_rounding_for_random_normal_and_cancelling_triples():
    rng=np.random.default_rng(91016)
    a=torch.from_numpy(np.ldexp(rng.uniform(-2,2,2000),rng.integers(-20,11,2000)).astype(np.float32))
    b=torch.from_numpy(np.ldexp(rng.uniform(-2,2,2000),rng.integers(-20,11,2000)).astype(np.float32))
    c=torch.from_numpy(np.ldexp(rng.uniform(-2,2,2000),rng.integers(-30,21,2000)).astype(np.float32))
    c[:500]=-(a[:500]*b[:500])
    expected=exact_oracle(a,b,c)
    np.testing.assert_array_equal(candidate()(a,b,c).numpy().view('u4'),expected.view('u4'))


def test_actual_standard_fp32_onnx_retains_midpoint_correction_and_no_fp64(tmp_path):
    from embeddinggemma_tools.audio_candidate_primitives import run_candidate
    args=fixtures();actual,metadata=run_candidate(candidate(),args,tmp_path)
    np.testing.assert_array_equal(actual.numpy().view('u4'),exact_oracle(*args).view('u4'))
    assert metadata['execution']=='actual_standard_fp32_onnx_on_same_native_inputs'
    assert not metadata['native_forward_used_as_candidate_evidence']


def test_actual_static_bmm_graph_uses_corrected_fma_without_native_forward_evidence(tmp_path):
    from embeddinggemma_tools import midpoint_fp32_fma
    from embeddinggemma_tools.audio_candidate_primitives import run_candidate
    a,b,c=fixtures()
    left=torch.stack((c,a),dim=-1).unsqueeze(1)
    right=torch.stack((torch.ones_like(b),b),dim=-1).unsqueeze(2)
    actual,metadata=run_candidate(midpoint_fp32_fma.static_midpoint_bmm_export,(left,right),tmp_path)
    np.testing.assert_array_equal(actual.numpy().ravel().view('u4'),exact_oracle(a,b,c).view('u4'))
    assert not metadata['native_forward_used_as_candidate_evidence']
