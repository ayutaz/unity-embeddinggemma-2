using System;
using System.Collections;
using System.Linq;
using System.Reflection;
using NUnit.Framework;
using Unity.InferenceEngine;

namespace EmbeddingGemma.Tests
{
    public sealed class WorkerObservationTests
    {
        [TestCase(BackendType.CPU)] [TestCase(BackendType.GPUCompute)]
        public void LayerObservationPreservesNormalFinalBitsAndCopiesLiveOutputs(BackendType backend)
        {
            var type = AppDomain.CurrentDomain.GetAssemblies().Select(assembly => assembly.GetType("EmbeddingGemma.Tests.WorkerObservation"))
                .FirstOrDefault(value => value != null);
            Assert.That(type, Is.Not.Null, "Unchanged-model layer observer is missing.");
            var shape = new TensorShape(1, 2);
            var graph = new FunctionalGraph(); var x = graph.AddInput<float>(shape, "input");
            var model = graph.Compile(x + Functional.Constant(shape, new[] { 1f, 1f }));
            var layers = model.layers.Select(layer => layer.GetType().Name + ":" + string.Join(",", layer.outputs)).ToArray();
            using var input = new Tensor<float>(shape, new[] { 1f, 2f });
            if (backend == BackendType.GPUCompute) ComputeTensorData.Pin(input);
            var result = type.GetMethod("Observe").Invoke(null, new object[] { model, backend, input, shape, 4 });
            var resultType = result.GetType();
            Assert.That((float[])resultType.GetField("Baseline").GetValue(result), Is.EqualTo(new[] { 2f, 3f }));
            Assert.That((float[])resultType.GetField("Final").GetValue(result), Is.EqualTo(new[] { 2f, 3f }));
            var snapshots = (IList)resultType.GetField("Snapshots").GetValue(result);
            Assert.That(snapshots.Count, Is.EqualTo(1));
            var snapshot = snapshots[0]; var snapshotType = snapshot.GetType();
            Assert.That((float[])snapshotType.GetField("Values").GetValue(snapshot), Is.EqualTo(new[] { 2f, 3f }));
            Assert.That(snapshotType.GetField("Backend").GetValue(snapshot), Is.EqualTo(backend));
            Assert.That(model.layers.Select(layer => layer.GetType().Name + ":" + string.Join(",", layer.outputs)), Is.EqualTo(layers));
            Assert.That(model.outputs.Count, Is.EqualTo(1), "Do not add debug outputs or change optimization.");
        }
    }
}
