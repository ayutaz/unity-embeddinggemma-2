"""Small inverse-only ONNX rewrite; never re-export or copy the model's weights."""
import copy
import hashlib

import onnx
import numpy as np
from onnx import numpy_helper


def _initializer_digest(values):
    digest = hashlib.sha256()
    for value in values:
        data = value.SerializeToString()
        digest.update(len(data).to_bytes(8, 'little'))
        digest.update(data)
    return digest.hexdigest()


def refine_welford_scalar_means(model, template, *, prefix, expected_count):
    """Inline FP32 FMA at the eight audited scalar AddMoments mean updates.

    Preserve lane order, variance arithmetic, weights and original references.
    This is an opt-in diagnostic candidate, not full encoder acceptance.
    """
    unsupported = {onnx.TensorProto.DOUBLE, onnx.TensorProto.FLOAT16, onnx.TensorProto.BFLOAT16}
    if (not isinstance(prefix, str) or not prefix or expected_count != 8
            or len(template.graph.input) != 3 or len(template.graph.output) != 1
            or any(v.type.tensor_type.elem_type != onnx.TensorProto.FLOAT
                   for v in (*template.graph.input, *template.graph.output))
            or any(n.domain not in ('', 'ai.onnx') for n in template.graph.node)
            or any(w.data_type in unsupported for w in template.graph.initializer)
            or any((n.op_type == 'Cast' and a.name == 'to' and a.i in unsupported)
                   or (a.type == onnx.AttributeProto.TENSOR and a.t.data_type in unsupported)
                   for n in template.graph.node for a in n.attribute)):
        raise ValueError('Scalar mean requires an explicit eight-lane scope and three FP32 template inputs')
    producers = {v: n for n in model.graph.node for v in n.output}
    weights = {w.name:w for w in model.graph.initializer}
    infos = {v.name:v for v in (*model.graph.input, *model.graph.output, *model.graph.value_info)}
    def constant(name):
        if name in weights: return numpy_helper.to_array(weights[name])
        node = producers.get(name)
        if node is None or node.op_type != 'Constant': return None
        values = [a.t for a in node.attribute if a.name == 'value']
        return numpy_helper.to_array(values[0]) if len(values) == 1 else None
    targets = []
    lane_source = None
    previous_output = None
    for node in model.graph.node:
        if not node.name.startswith(prefix) or node.op_type != 'Add' or len(node.input) != 2: continue
        previous, product_name = node.input
        product = producers.get(product_name)
        if product is None: continue
        if product.op_type == 'Mul' and len(product.input) == 2:
            coefficient_name, delta_name = product.input
            coefficient = constant(coefficient_name)
        elif product.op_type == 'Sub' and not targets:
            # Standard ONNX constant folding removes multiplication by one.
            coefficient_name, delta_name = None, product_name
            coefficient = np.array(1, dtype=np.float32)
        else: continue
        delta = producers.get(delta_name)
        if (coefficient is None or coefficient.size != 1 or delta is None
                or delta.op_type != 'Sub' or len(delta.input) != 2): continue
        if delta.input[1] != previous:
            # Export may emit duplicate, bit-identical initial zero constants.
            left, right = constant(delta.input[1]), constant(previous)
            if (targets or left is None or right is None or left.dtype != np.float32
                    or right.dtype != np.float32 or not np.array_equal(left, right)
                    or not np.all(left == 0)): continue
        sliced = producers.get(delta.input[0])
        if sliced is None or sliced.op_type != 'Slice' or len(sliced.input) != 5: continue
        bounds = [constant(v) for v in sliced.input[1:]]
        lane = len(targets)
        info = infos.get(sliced.input[0])
        shape = info.type.tensor_type.shape.dim if info is not None else []
        if (lane >= 8 or coefficient.dtype != np.float32 or not np.isfinite(coefficient).all()
                or coefficient.item() != np.float32(1 / (lane+1))
                or any(v is None or v.size != 1 or v.dtype != np.int64 for v in bounds)
                or not shape or shape[-1].dim_value != 8
                or int(bounds[2].item()) not in (-1, len(shape)-1)
                or int(bounds[0].item()) != lane or int(bounds[3].item()) != 1
                or int(bounds[1].item()) != lane+1):
            raise ValueError(f'Scalar mean lane bounds or reciprocal coefficient differ at {node.name}: lane {lane}, coefficient {coefficient}, bounds {bounds}')
        if lane == 0:
            zeros = constant(previous)
            if zeros is None or zeros.dtype != np.float32 or not np.all(zeros == 0):
                raise ValueError('Scalar mean initial accumulator differs')
            lane_source = sliced.input[0]
        elif previous != previous_output or sliced.input[0] != lane_source:
            raise ValueError('Scalar mean ordered lane chain differs')
        previous_output = node.output[0]
        targets.append((node, (coefficient_name, delta_name, previous)))
    if len(targets) != expected_count: raise ValueError('Scalar mean target count differs')
    reserved = {v.name for v in (*model.graph.input, *model.graph.output, *model.graph.initializer, *model.graph.value_info)}
    reserved.update(v for n in model.graph.node for v in (*n.input, *n.output))
    def fresh(value):
        while value in reserved: value += '_'
        reserved.add(value)
        return value
    replacements, extra = {}, []
    unit_name = fresh('__eg_scalar_mean_unit')
    if any(inputs[0] is None for _, inputs in targets):
        extra.append(numpy_helper.from_array(np.array(1, dtype=np.float32), name=unit_name))
    for index, (target, inputs) in enumerate(targets):
        inputs = (inputs[0] or unit_name, *inputs[1:])
        names = dict(zip((v.name for v in template.graph.input), inputs))
        names[template.graph.output[0].name] = target.output[0]
        def rename(value):
            if not value: return value
            if value not in names: names[value] = fresh(f'__eg_scalar_mean_{index}_'+value)
            return names[value]
        for original in template.graph.initializer:
            value = copy.deepcopy(original); value.name = rename(value.name); extra.append(value)
        nodes = []
        for original in template.graph.node:
            value = copy.deepcopy(original); value.name = fresh(f'__eg_scalar_mean_{index}_node_'+value.name)
            value.input[:] = [rename(v) for v in original.input]
            value.output[:] = [rename(v) for v in original.output]
            nodes.append(value)
        replacements[target.name] = nodes
    before = _initializer_digest(model.graph.initializer); count = len(model.graph.initializer)
    rewritten = []
    for node in model.graph.node: rewritten.extend(replacements.get(node.name, [node]))
    del model.graph.node[:]; model.graph.node.extend(rewritten); model.graph.initializer.extend(extra)
    if _initializer_digest(model.graph.initializer[:count]) != before:
        raise RuntimeError('Existing scalar mean initializer bytes changed')
    return dict(replaced_scalar_mean_count=len(targets), prefix=prefix,
                existing_initializer_bytes_preserved=True, existing_initializer_sha256=before,
                node_count=len(model.graph.node), full_encoder_validated=False)


def refine_layernorm_inverses(model, template, *, prefixes=(
        '/subsample_conv_projection/layer0/norm', '/subsample_conv_projection/layer1/norm')):
    """Inline a bounded FP32 refinement at explicitly named Sqrt/Reciprocal pairs.

    Audit every target before mutation. Existing initializers, inputs, outputs,
    native references and all other operations remain intact. The caller must
    establish the positive statistic domain of the experimental template.
    """
    if (not prefixes or len(set(prefixes)) != len(prefixes)
            or any(not isinstance(p, str) or not p for p in prefixes)):
        raise ValueError('Select unique explicit inverse prefixes')
    if (len(template.graph.input) != 1 or len(template.graph.output) != 1
            or any(v.type.tensor_type.elem_type != onnx.TensorProto.FLOAT
                   for v in (*template.graph.input, *template.graph.output))
            or any(n.domain not in ('', 'ai.onnx') for n in template.graph.node)
            or any(n.op_type == 'Cast' and a.name == 'to' and a.i == onnx.TensorProto.DOUBLE
                   for n in template.graph.node for a in n.attribute)
            or any(w.data_type == onnx.TensorProto.DOUBLE for w in template.graph.initializer)
            or any(a.type == onnx.AttributeProto.TENSOR and a.t.data_type == onnx.TensorProto.DOUBLE
                   for n in template.graph.node for a in n.attribute)):
        raise ValueError('Template requires one FP32 standard-op input/output')
    targets = []
    for prefix in prefixes:
        inverse = [n for n in model.graph.node if n.name == prefix+'/Reciprocal']
        roots = [n for n in model.graph.node if n.name == prefix+'/Sqrt']
        if (len(inverse) != 1 or len(roots) != 1 or inverse[0].op_type != 'Reciprocal'
                or roots[0].op_type != 'Sqrt' or len(roots[0].input) != 1
                or len(roots[0].output) != 1 or len(inverse[0].input) != 1
                or len(inverse[0].output) != 1 or inverse[0].input[0] != roots[0].output[0]):
            raise ValueError('Inverse pair is missing, ambiguous or structurally different')
        users = [n for n in model.graph.node if roots[0].output[0] in n.input]
        if len(users) != 1 or any(o.name == roots[0].output[0] for o in model.graph.output):
            raise ValueError('Inverse square root seed has another consumer')
        targets.append((prefix, roots[0], inverse[0]))
    reserved = {v.name for v in (*model.graph.input, *model.graph.output, *model.graph.initializer, *model.graph.value_info)}
    reserved.update(v for n in model.graph.node for v in (*n.input, *n.output))
    def fresh(value):
        while value in reserved:value += '_'
        reserved.add(value)
        return value
    replacements, extra_weights = {}, []
    for prefix, root, inverse in targets:
        namespace = '__eg_refined_'+prefix.replace('/', '_')+'_'
        names = {template.graph.input[0].name: root.input[0], template.graph.output[0].name: inverse.output[0]}
        def rename(value):
            if not value:return value
            if value not in names:names[value] = fresh(namespace+value)
            return names[value]
        for initializer in template.graph.initializer:
            weight = copy.deepcopy(initializer);weight.name = rename(initializer.name);extra_weights.append(weight)
        nodes = []
        for original in template.graph.node:
            node = copy.deepcopy(original)
            node.name = fresh(namespace+'node_'+original.name)
            node.input[:] = [rename(v) for v in original.input]
            node.output[:] = [rename(v) for v in original.output]
            nodes.append(node)
        replacements[inverse.name] = nodes
    existing_count = len(model.graph.initializer)
    before = _initializer_digest(model.graph.initializer)
    removed = {root.name for _, root, _ in targets}
    rewritten = []
    for node in model.graph.node:
        if node.name in replacements:rewritten.extend(replacements[node.name])
        elif node.name not in removed:rewritten.append(node)
    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    model.graph.initializer.extend(extra_weights)
    after = _initializer_digest(model.graph.initializer[:existing_count])
    if before != after:raise RuntimeError('Existing initializer bytes changed')
    return dict(replaced_inverse_count=len(targets), prefixes=list(prefixes),
                existing_initializer_bytes_preserved=True, existing_initializer_sha256=before,
                node_count=len(model.graph.node), full_encoder_validated=False)
