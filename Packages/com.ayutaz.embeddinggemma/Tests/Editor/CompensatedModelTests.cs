using System;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.ExceptionServices;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEngine;

namespace EmbeddingGemma.Tests
{
    public sealed class CompensatedModelTests
    {
        static Model Create(Model source)
        {
            var type = AppDomain.CurrentDomain.GetAssemblies().Select(assembly => assembly.GetType("EmbeddingGemma.Tests.CompensatedModel")).FirstOrDefault(value => value != null);
            Assert.That(type, Is.Not.Null, "Immutable saved-model compensation is missing.");
            try { return (Model)type.GetMethod("Create").Invoke(null, new object[] { source, 1 }); }
            catch (TargetInvocationException error) { ExceptionDispatchInfo.Capture(error.InnerException).Throw(); throw; }
        }
        static Model Fixture()
        {
            var graph = new FunctionalGraph(); var input = graph.AddInput<float>(new TensorShape(1, 3), "input");
            var weight = Functional.Constant(new TensorShape(3, 1), new[] { 1f, 1f, 1f });
            return graph.Compile(Functional.MatMul(input, weight) * 2f + 5f);
        }
        [TestCase(BackendType.CPU)] [TestCase(BackendType.GPUCompute)]
        public void CompensatedGraphPreservesDownstreamRoutingAndSourceBytes(BackendType backend)
        {
            var source = Fixture(); using var before = new MemoryStream(); ModelWriter.Save(before, source);
            var candidate = Create(source);
            Assert.That(candidate.inputs.Select(value => value.index), Is.EqualTo(source.inputs.Select(value => value.index)));
            Assert.That(candidate.outputs.Select(value => value.index), Is.EqualTo(source.outputs.Select(value => value.index)));
            Assert.That(candidate.outputs.Select(value => value.name), Is.EqualTo(source.outputs.Select(value => value.name)));
            using var input = new Tensor<float>(new TensorShape(1, 3), new[] { 1e8f, 1f, -1e8f });
            if (backend == BackendType.GPUCompute) { Assert.That(SystemInfo.supportsComputeShaders, Is.True); ComputeTensorData.Pin(input); }
            using var worker = new Worker(candidate, backend); worker.Schedule(input);
            var output = (Tensor<float>)worker.PeekOutput(); Assert.That(output.backendType, Is.EqualTo(backend));
            Assert.That(output.DownloadToArray(), Is.EqualTo(new[] { 7f }));
            using var after = new MemoryStream(); ModelWriter.Save(after, source); Assert.That(after.ToArray(), Is.EqualTo(before.ToArray()));
        }
        [Test] public void DynamicOperandProjectionIsBorrowedWithoutChangingItsIdentity()
        {
            var graph = new FunctionalGraph(); var a = graph.AddInput<float>(new TensorShape(1, 3), "a");
            var b = graph.AddInput<float>(new TensorShape(3, 1), "b"); var source = graph.Compile(Functional.MatMul(a, b));
            var candidate = Create(source);
            Assert.That(candidate.layers, Is.EqualTo(source.layers)); Assert.That(candidate.inputs.Count, Is.EqualTo(2));
        }
        [Test] public void BroadcastConstantMatmulRetainsItsBatchAndLayerIdentity()
        {
            var graph = new FunctionalGraph(); var input = graph.AddInput<float>(new TensorShape(2, 1, 3), "input");
            var weight = Functional.Constant(new TensorShape(2, 3, 1), new[] { 1f, 2f, 3f, 4f, 5f, 6f });
            var source = graph.Compile(Functional.MatMul(input, weight));
            Assert.That(source.constants.Any(value => value.shape.rank == 3), Is.True);
            var candidate = Create(source);
            Assert.That(candidate.layers, Is.EqualTo(source.layers));
            Assert.That(candidate.constants.Select(value => value.shape), Is.EqualTo(source.constants.Select(value => value.shape)));
        }
    }
}
