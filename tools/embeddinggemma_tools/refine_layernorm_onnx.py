"""Small inverse-only ONNX rewrite; never re-export or copy the model's weights."""
import copy
import hashlib

import onnx


def _initializer_digest(values):
    digest = hashlib.sha256()
    for value in values:
        data = value.SerializeToString()
        digest.update(len(data).to_bytes(8, 'little'))
        digest.update(data)
    return digest.hexdigest()


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
            or any(n.domain not in ('', 'ai.onnx') for n in template.graph.node)
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
