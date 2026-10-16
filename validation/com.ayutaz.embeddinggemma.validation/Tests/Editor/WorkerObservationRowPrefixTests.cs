using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using NUnit.Framework;
using Unity.InferenceEngine;

namespace EmbeddingGemma.Tests
{
    public sealed class WorkerObservationRowPrefixTests
    {
        static MethodInfo RequiredMethod()
        {
            var method = typeof(WorkerObservation).GetMethod("ObserveRowPrefixes");
            Assert.That(method, Is.Not.Null, "Bounded per-group GPU row observation is missing.");
            return method;
        }

        [TestCase(2)] [TestCase(8)]
        public void OnlyFirstRowsOfEveryGroupAreCopiedWithoutChangingNormalOutput(int rows)
        {
            var method = RequiredMethod();
            var shape = new TensorShape(1, 2, 4, 5);
            var values = Enumerable.Range(0, shape.length).Select(i => i * .25f).ToArray();
            var graph = new FunctionalGraph(); var x = graph.AddInput<float>(shape, "input");
            var first = x + 1f; var model = graph.Compile(first, first * 2f);
            var descriptors = model.layers.Select(layer => layer.GetType().Name + ":" + string.Join(",", layer.outputs)).ToArray();
            using var input = new Tensor<float>(shape, values); ComputeTensorData.Pin(input);
            var selected = new HashSet<int>(model.outputs.Select(value => value.index));
            var result = (WorkerObservation.Result)method.Invoke(null, new object[] { model, input, selected, rows });
            Assert.That(result.FinalMatchesBaselineBits, Is.True);
            Assert.That(result.Final, Is.EqualTo(values.Select(v => v + 1f).ToArray()));
            Assert.That(result.Snapshots.Select(value => value.TensorIndex), Is.EquivalentTo(selected));
            foreach (var snapshot in result.Snapshots)
            {
                Assert.That(snapshot.Backend, Is.EqualTo(BackendType.GPUCompute));
                Assert.That(snapshot.Shape, Is.EqualTo(new TensorShape(1, 2, Math.Min(rows, 4), 5)));
                var original = typeof(WorkerObservation.Snapshot).GetField("OriginalShape");
                Assert.That(original, Is.Not.Null); Assert.That(original.GetValue(snapshot), Is.EqualTo(shape));
                var multiplier = snapshot.TensorIndex == model.outputs[0].index ? 1f : 2f;
                var expected = Enumerable.Range(0, 2).SelectMany(group => Enumerable.Range(0, Math.Min(rows, 4) * 5)
                    .Select(i => (values[group * 20 + i] + 1f) * multiplier)).ToArray();
                Assert.That(snapshot.Values, Is.EqualTo(expected), "Each group must retain its own row prefix, not one flat prefix.");
            }
            Assert.That(input.DownloadToArray(), Is.EqualTo(values));
            Assert.That(model.layers.Select(layer => layer.GetType().Name + ":" + string.Join(",", layer.outputs)), Is.EqualTo(descriptors));
            Assert.That(model.outputs.Count, Is.EqualTo(2));
        }

        [Test]
        public void ZeroRowLimitIsRejectedBeforeScheduling()
        {
            var method = RequiredMethod();
            var graph = new FunctionalGraph(); var x = graph.AddInput<float>(new TensorShape(1, 2), "input");
            var model = graph.Compile(x + 1f); using var input = new Tensor<float>(new TensorShape(1, 2), new[] { 1f, 2f });
            ComputeTensorData.Pin(input);
            var error = Assert.Throws<TargetInvocationException>(() => method.Invoke(null,
                new object[] { model, input, new HashSet<int> { model.outputs[0].index }, 0 }));
            Assert.That(error.InnerException, Is.TypeOf<ArgumentOutOfRangeException>());
        }
    }
}
