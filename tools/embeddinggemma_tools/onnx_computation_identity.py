"""Bind embedded ONNX computation while excluding only optional debug metadata.

This reads and fingerprints models; it never executes, rewrites, or qualifies
them. Operator routing, attributes, tensors, types, shapes, IR and opsets remain
bound. Metadata identity is separate from real numeric and GPU acceptance.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path
import re

import onnx


def fingerprint(path, *, expected_sha256):
    with Path(path).open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if re.fullmatch('[0-9a-f]{64}', expected_sha256 or '') is None or actual != expected_sha256:
        raise ValueError('Pinned ONNX source hash differs')
    model = onnx.load_model(path, load_external_data=False)
    counts = dict(metadata_fields_removed=0, node_count=0, tensor_count=0)
    def visit(message):
        kind = message.DESCRIPTOR.full_name
        if kind == 'onnx.NodeProto': counts['node_count'] += 1
        if kind == 'onnx.TensorProto':
            counts['tensor_count'] += 1
            if message.data_location == onnx.TensorProto.EXTERNAL or message.external_data:
                raise ValueError('Only fully embedded tensor data can be fingerprinted')
        for field, value in message.ListFields():
            if field.name in ('doc_string', 'metadata_props'):
                message.ClearField(field.name); counts['metadata_fields_removed'] += 1
            elif field.type == field.TYPE_MESSAGE:
                if field.is_repeated:
                    for child in value: visit(child)
                else:
                    visit(value)
    visit(model)
    digest = hashlib.sha256(model.SerializeToString(deterministic=True)).hexdigest()
    def checksum(value):
        return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    entries=[];order=[]
    for weight in model.graph.initializer:
        name=weight.name;order.append(name);weight.ClearField('name')
        value_sha=hashlib.sha256(weight.SerializeToString(deterministic=True)).hexdigest()
        weight.name=name;entries.append((name,value_sha))
    operations=dict(Counter((node.domain or 'ai.onnx')+'::'+node.op_type for node in model.graph.node))
    components=dict(initializer_count=len(entries),initializer_order_sha256=checksum(order),
        named_initializer_values_sha256=checksum(sorted(entries)),
        initializer_values_sha256=checksum(sorted(value for _,value in entries)),node_operations=operations)
    # These extra digests classify differences; they never replace whole identity.
    del model.graph.initializer[:]
    components['graph_without_initializers_sha256']=hashlib.sha256(model.graph.SerializeToString(deterministic=True)).hexdigest()
    model.ClearField('graph')
    components['model_header_sha256']=hashlib.sha256(model.SerializeToString(deterministic=True)).hexdigest()
    return dict(schema=1, scope='embedded_onnx_computation_identity_excluding_optional_debug_metadata_only',
        source_sha256=actual, computation_sha256=digest,
        ignored_metadata_fields=['doc_string','metadata_props'], components=components, **counts,
        model_executed=False, gpu_validated=False, generation_qualified=False)
