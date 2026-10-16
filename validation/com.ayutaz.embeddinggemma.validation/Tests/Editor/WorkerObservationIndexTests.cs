using System;
using System.Collections;
using System.Collections.Generic;
using System.Reflection;
using NUnit.Framework;
using Unity.InferenceEngine;

namespace EmbeddingGemma.Tests
{
    public sealed class WorkerObservationIndexTests
    {
        [TestCase(BackendType.CPU)] [TestCase(BackendType.GPUCompute)]
        public void OnlyRequestedTensorIsCapturedAndFinalBitsStayUnchanged(BackendType backend)
        {
            var method = typeof(WorkerObservation).GetMethod("ObserveIndexes");
            Assert.That(method, Is.Not.Null, "Selective layer observation is missing.");
            var graph = new FunctionalGraph(); var x = graph.AddInput<float>(new TensorShape(1, 2), "input");
            var first = x + 1f; var model = graph.Compile(first, first * 2f);
            var requested = model.outputs[1].index;
            using var input = new Tensor<float>(new TensorShape(1, 2), new[] { 1f, 2f });
            if (backend == BackendType.GPUCompute) ComputeTensorData.Pin(input);
            var result = (WorkerObservation.Result)method.Invoke(null, new object[] { model, backend, input, new HashSet<int> { requested } });
            Assert.That(result.FinalMatchesBaselineBits, Is.True);
            Assert.That(result.Snapshots.Count, Is.EqualTo(1));
            Assert.That(result.Snapshots[0].TensorIndex, Is.EqualTo(requested));
            Assert.That(result.Snapshots[0].Values, Is.EqualTo(new[] { 4f, 6f }));
            var shape=typeof(WorkerObservation.Snapshot).GetField("Shape");
            Assert.That(shape,Is.Not.Null,"Captured tensor shape is missing.");
            Assert.That(shape.GetValue(result.Snapshots[0]),Is.EqualTo(new TensorShape(1,2)));
        }
    }
}
