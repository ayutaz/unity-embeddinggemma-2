"""Small fixed-artifact experiment for the first required-CI audio ONNX failure."""
import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np
import onnx
from onnx import numpy_helper
from onnx.reference import ReferenceEvaluator
import torch

from . import export
from .refine_layernorm_node_onnx import refine_layernorm_node
from .refine_matmul_onnx import refine_static_matmul

PINNED_FILES={
    "audio-onnx-subsample-stages.onnx":(4357883,"581f6562bbf0e4f6ac4ad7703ebd7a86c47dc55f43404932a9ae21b67a95ae27"),
    "audio-subsample-native_input.bin":(572928,"366de71cb4e45e4feda58a605babe4ea9d8693a7e37e84eac0735a7c42d0ad24"),
    "audio-subsample-frame_mask.bin":(4476,"d1d720a6b7013a4b396f2a8595d52f9648132b665f40f899b09bc62918e57c95"),
    "audio-subsample-native_subsample.bin":(1146880,"ef8ed757c419086c1e0b680ae2815b61e6c8b36cc1ff89e0d0209c63a1e8d42f")}


def digest(path):
    with Path(path).open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()


class CompensatedMatmul(torch.nn.Module):
    def forward(self,values,weights):
        total=values.new_zeros((*values.shape[:-1],weights.shape[1]));correction=torch.zeros_like(total)
        for start in range(0,weights.shape[0],64):
            part=values[...,start:start+64]@weights[start:start+64,:]
            updated=total+part
            correction=correction+torch.where(total.abs()>=part.abs(),(total-updated)+part,(part-updated)+total)
            total=updated
        return total+correction


class NormTemplate(torch.nn.Module):
    def __init__(self,width,epsilon):
        super().__init__()
        self.norm=export.TorchWelfordLayerNorm(torch.nn.LayerNorm(width,eps=epsilon,bias=False),refine_inverse=True,fuse_scalar_moment=True)
    def forward(self,values,scale):return self.norm(values)*scale


def template(module,inputs,path):
    with torch.no_grad():torch.onnx.export(module.eval(),inputs,path,dynamo=False,opset_version=20,input_names=["values","coefficients"],output_names=["result"])
    return onnx.version_converter.convert_version(onnx.load(path),24)


def probe(directory,output,*,source_commit):
    directory,output=Path(directory),Path(output);output.mkdir(parents=True,exist_ok=True)
    receipt=output/"subsample-probe.json";receipt.unlink(missing_ok=True)
    if re.fullmatch(r"[0-9a-f]{40}",source_commit or "") is None:raise ValueError("Pinned probe requires immutable source provenance")
    for name,(size,sha) in PINNED_FILES.items():
        path=directory/name
        if not path.is_file() or path.stat().st_size!=size or digest(path)!=sha:raise ValueError("Pinned subsample input/graph hash mismatch: "+name)
    graph=onnx.shape_inference.infer_shapes(onnx.load(directory/"audio-onnx-subsample-stages.onnx"))
    original_initializers=[weight.SerializeToString() for weight in graph.graph.initializer]
    features=np.fromfile(directory/"audio-subsample-native_input.bin",dtype="<f4").reshape(1,1119,128)
    expected=np.fromfile(directory/"audio-subsample-native_subsample.bin",dtype="<f4").reshape(1,280,1024)
    mask=np.fromfile(directory/"audio-subsample-frame_mask.bin",dtype="<i4")
    if not np.isfinite(features).all() or not np.isfinite(expected).all() or mask.shape!=(1119,) or not np.all(mask==1):raise ValueError("Pinned subsample finite/profile contract differs")
    report=dict(source_commit=source_commit,reference_producer="040d4e91dc6d5fe862b4e6fb9b24f56f7e830faa",artifact_id=11986081282,
        scope="fixed_required_ci_silence_subsample_only",success=False,complete=False,cases=[],
        original_coefficients_or_reference_changed=False,full_encoder_validated=False,gpu_validated=False,end_to_end_validated=False,
        pinned_files={name:dict(bytes=size,sha256=sha) for name,(size,sha) in PINNED_FILES.items()})
    def save():receipt.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    save()
    def evaluate(model,name,rewrites):
        onnx.checker.check_model(model)
        assert [weight.SerializeToString() for weight in model.graph.initializer[:len(original_initializers)]]==original_initializers
        values=ReferenceEvaluator(model).run(None,{"input":features})[-1]
        if values.dtype!=np.float32 or values.shape!=expected.shape or not np.isfinite(values).all():raise ValueError("Subsample candidate output contract differs")
        differences=np.abs(values.astype(np.float64)-expected.astype(np.float64));allowed=1e-5+1e-4*np.abs(expected.astype(np.float64))
        mismatches=int((differences>allowed).sum());capture=output/(name+"-fp32.bin");values.astype("<f4").tofile(capture)
        row=dict(name=name,passed=mismatches==0,mismatched_elements=mismatches,maximum_absolute_error=float(differences.max()),
            maximum_tolerance_ratio=float((differences/allowed).max()),capture_sha256=digest(capture),rewrites=rewrites,
            existing_initializer_bytes_preserved=True)
        report["cases"].append(row);save();del values;gc.collect()
    evaluate(graph,"original_control",[])
    candidate=copy.deepcopy(graph);rewrites=[]
    for node in [node for node in candidate.graph.node if node.op_type=="LayerNormalization"]:
        infos={value.name:value for value in (*candidate.graph.input,*candidate.graph.value_info,*candidate.graph.output)}
        shape=[dimension.dim_value for dimension in infos[node.input[0]].type.tensor_type.shape.dim]
        if len(shape)!=4 or shape[-1] not in (32,128):raise ValueError("Pinned subsample LayerNorm scope changed")
        epsilon=next(attribute.f for attribute in node.attribute if attribute.name=="epsilon")
        weight=next(weight for weight in candidate.graph.initializer if weight.name==node.input[1]);scale=numpy_helper.to_array(weight)
        built=template(NormTemplate(shape[-1],epsilon),(torch.zeros(shape),torch.from_numpy(scale.copy())),output/("norm-template-"+str(shape[-1])+".onnx"))
        rewrites.append(refine_layernorm_node(candidate,built,node_name=node.name,expected_scale_sha256=hashlib.sha256(scale.astype("<f4").tobytes()).hexdigest(),expected_epsilon=epsilon))
    if len(rewrites)!=2:raise ValueError("Pinned subsample must contain exactly two LayerNorm nodes")
    evaluate(candidate,"refined_welford_moment",list(rewrites))
    projection=next(node for node in candidate.graph.node if node.name=="/input_proj_linear/MatMul")
    info=next(value for value in candidate.graph.value_info if value.name==projection.input[0]);shape=[dimension.dim_value for dimension in info.type.tensor_type.shape.dim]
    weight=next(weight for weight in candidate.graph.initializer if weight.name==projection.input[1]);values=numpy_helper.to_array(weight)
    built=template(CompensatedMatmul(),(torch.zeros(shape),torch.from_numpy(values.copy())),output/"projection-template.onnx")
    rewrites.append(refine_static_matmul(candidate,built,node_name=projection.name,expected_weight_sha256=hashlib.sha256(values.astype("<f4").tobytes()).hexdigest()))
    evaluate(candidate,"refined_welford_moment_compensated64",rewrites)
    report.update(success=report["cases"][-1]["passed"],complete=True);save();return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--input",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();report=probe(args.input,args.output,source_commit=os.environ.get("GITHUB_SHA","local"));print(json.dumps(report,indent=2))
    if not report["success"]:raise SystemExit(1)


if __name__=="__main__":main()
