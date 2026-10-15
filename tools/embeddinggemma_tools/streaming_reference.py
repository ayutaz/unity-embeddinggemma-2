"""Release dead values while running the same ONNX reference kernels in order.

Restricted to the flat graphs used by the explicit numerical diagnostics.
No graph rewriting, fusion, dtype change, or replacement operator is performed.
The default repository reference evaluator and its gates remain unchanged.
"""

from collections import Counter

from onnx.reference import ReferenceEvaluator


def run_flat_reference(evaluator, feed_inputs, *, output_names=None, observer=None):
    if not isinstance(evaluator, ReferenceEvaluator):
        raise ValueError("Flat execution requires the ONNX ReferenceEvaluator")
    if evaluator.check_shape_annotations_:
        raise ValueError("Flat diagnostic does not implement symbolic shape bindings")
    if any(node.need_context() or node.has_linked_attribute for node in evaluator.rt_nodes_):
        raise ValueError("Flat diagnostic cannot release values used by implicit context or linked attributes")
    requested = list(evaluator.output_names if output_names is None else output_names)
    keep = set(requested)
    uses = Counter(name for node in evaluator.rt_nodes_ for name in node.input if name)
    values = {"": None, **evaluator.rt_inits_, **feed_inputs}
    for node in evaluator.rt_nodes_:
        missing = [name for name in node.input if name not in values]
        if missing:
            raise RuntimeError(f"Unable to find flat reference input {missing}")
        inputs = [values[name] for name in node.input]
        outputs = node.run(*inputs)
        if len(outputs) != len(node.output):
            raise RuntimeError("Flat reference operator output arity differs")
        for name, value in zip(node.output, outputs, strict=True):
            values[name] = value
        for name in node.input:
            if not name:
                continue
            uses[name] -= 1
            if not uses[name] and name not in keep:
                values.pop(name, None)
        for name in node.output:
            if not uses[name] and name not in keep and name:
                values.pop(name, None)
        if outputs:
            del value
        del inputs, outputs
        if observer is not None:
            observer(tuple(values))
    missing = keep.difference(values)
    if missing:
        raise RuntimeError(f"Unable to find flat reference output {sorted(missing)}")
    return [values[name] for name in requested]
