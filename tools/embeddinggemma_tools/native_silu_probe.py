"""Small native SiLU references and standard ONNX Swish import probes."""

import hashlib
import json
from pathlib import Path
import re

import onnx
from onnx import helper, TensorProto
import torch
from .export import save_program_without_debug_traces


class NativeSiLU(torch.nn.Module):
    def forward(self, values):
        return torch.nn.functional.silu(values)


def prepare_silu_probe(destination, *, source_commit):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    receipt=destination/"silu-validation.json";receipt.unlink(missing_ok=True)
    if not isinstance(source_commit,str) or re.fullmatch(r"[0-9a-f]{40}",source_commit) is None:
        raise ValueError("Native SiLU probe requires an immutable source SHA")
    cases=[];files={}
    def record(path):
        files[path.name]=dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    for index,shape in enumerate(((17,),(1,2,4096))):
        values=torch.randn(shape,generator=torch.Generator().manual_seed(412))*2
        with torch.no_grad():
            native=torch.nn.functional.silu(values)
            div_exp=values/(1+torch.exp(-values))
            program=torch.export.export(NativeSiLU().eval(),(values,))
        native_path=destination/f"native-{index}.pt2"
        # No decomposition: the reference must retain the actual aten.silu kernel.
        save_program_without_debug_traces(program,native_path)
        saved=torch.export.load(native_path).module()(values)
        if not torch.equal(saved,native):raise ValueError("Native SiLU saved graph changed the kernel")
        onnx_path=destination/f"swish-{index}.onnx"
        graph=helper.make_graph([helper.make_node("Swish",["input"],["output"],alpha=1.0)],f"native-silu-{index}",
            [helper.make_tensor_value_info("input",TensorProto.FLOAT,list(shape))],
            [helper.make_tensor_value_info("output",TensorProto.FLOAT,list(shape))])
        model=helper.make_model(graph,opset_imports=[helper.make_opsetid("",24)],ir_version=10)
        onnx.checker.check_model(model);onnx.save_model(model,onnx_path)
        record(native_path);record(onnx_path)
        cases.append(dict(id=f"case-{index}",shape=list(shape),input=values.flatten().tolist(),
                          expected=native.flatten().tolist(),native_graph=native_path.name,onnx_graph=onnx_path.name,
                          native_saved_exact=True,div_exp_exact_mismatches=int((div_exp!=native).sum()),
                          div_exp_maximum_absolute_error=(div_exp-native).abs().max().item()))
    reference=destination/"silu-reference.json"
    reference.write_text(json.dumps(dict(metadata=dict(source_commit=source_commit,torch_build_version=torch.__version__,
        cpu_capability=torch.backends.cpu.get_cpu_capability(),intraop_threads=torch.get_num_threads(),
        interop_threads=torch.get_num_interop_threads(),torch_build_configuration=torch.__config__.show(),
        feature_atol=1e-5,feature_rtol=1e-4),cases=cases),indent=2,allow_nan=False)+"\n",encoding="utf-8")
    record(reference)
    report=dict(success=True,stage="native_silu_swish_probe_only",case_count=len(cases),source_commit=source_commit,
                unity_validated=False,audio_encoder_validated=False,files=files)
    receipt.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    return report
