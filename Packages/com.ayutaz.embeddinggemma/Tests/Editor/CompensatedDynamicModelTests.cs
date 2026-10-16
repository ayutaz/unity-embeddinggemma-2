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
    public sealed class CompensatedDynamicModelTests
    {
        [TestCase(BackendType.CPU)] [TestCase(BackendType.GPUCompute)]
        public void SavedDynamicMatmulRetainsSourceBytesAndFinalOutputIdentity(BackendType backend)
        {
            var graph = new FunctionalGraph();
            var a = graph.AddInput<float>(new TensorShape(2, 1, 3), "a");
            var b = graph.AddInput<float>(new TensorShape(1, 3, 1), "b");
            var source = graph.Compile(Functional.MatMul(a, b) + 2f);
            using var before = new MemoryStream(); ModelWriter.Save(before, source);
            var method = typeof(CompensatedModel).GetMethod("CreateWithDynamicMatMuls");
            Assert.That(method, Is.Not.Null, "Immutable dynamic attention compensation transform is missing.");
            Model candidate;
            try { candidate = (Model)method.Invoke(null, new object[] { source, 1 }); }
            catch (TargetInvocationException error) { ExceptionDispatchInfo.Capture(error.InnerException).Throw(); throw; }
            Assert.That(candidate.inputs.Select(value => value.index), Is.EqualTo(source.inputs.Select(value => value.index)));
            Assert.That(candidate.outputs.Select(value => value.index), Is.EqualTo(source.outputs.Select(value => value.index)));
            Assert.That(candidate.outputs.Select(value => value.name), Is.EqualTo(source.outputs.Select(value => value.name)));
            using var left = new Tensor<float>(new TensorShape(2, 1, 3), new[] { 1e8f, 1f, -1e8f, 1f, 1e8f, -1e8f });
            using var right = new Tensor<float>(new TensorShape(1, 3, 1), new[] { 1f, 1f, 1f });
            if (backend == BackendType.GPUCompute)
            {
                Assert.That(SystemInfo.supportsComputeShaders, Is.True); ComputeTensorData.Pin(left); ComputeTensorData.Pin(right);
            }
            using var worker = new Worker(candidate, backend); worker.Schedule(left, right);
            var output = (Tensor<float>)worker.PeekOutput();
            Assert.That(output.backendType, Is.EqualTo(backend));
            Assert.That(output.DownloadToArray(), Is.EqualTo(new[] { 3f, 3f }));
            using var after = new MemoryStream(); ModelWriter.Save(after, source);
            Assert.That(after.ToArray(), Is.EqualTo(before.ToArray()));
        }
    }
}
