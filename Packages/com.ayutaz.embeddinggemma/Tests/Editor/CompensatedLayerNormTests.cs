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
    public sealed class CompensatedLayerNormTests
    {
        static Model Build(float[] scale, float[] bias, int chunk, int[] shape, float epsilon)
        {
            var type = AppDomain.CurrentDomain.GetAssemblies().Select(value => value.GetType("EmbeddingGemma.Tests.CompensatedLayerNorm")).FirstOrDefault(value => value != null);
            var method = type?.GetMethod("Build");
            Assert.That(method, Is.Not.Null, "Centered FP32 compensated LayerNorm candidate is missing.");
            try { return (Model)method.Invoke(null, new object[] { scale, bias, chunk, shape, epsilon }); }
            catch (TargetInvocationException error) { ExceptionDispatchInfo.Capture(error.InnerException).Throw(); throw; }
        }
        static Model Transform(Model source)
        {
            var method = typeof(CompensatedModel).GetMethod("CreateWithCompensatedNorms");
            Assert.That(method, Is.Not.Null, "Immutable LayerNorm graph transform is missing.");
            try { return (Model)method.Invoke(null, new object[] { source, 2, 2 }); }
            catch (TargetInvocationException error) { ExceptionDispatchInfo.Capture(error.InnerException).Throw(); throw; }
        }
        [TestCase(BackendType.CPU, 2)] [TestCase(BackendType.GPUCompute, 2)]
        [TestCase(BackendType.CPU, 3)] [TestCase(BackendType.GPUCompute, 3)]
        [TestCase(BackendType.CPU, 4)] [TestCase(BackendType.GPUCompute, 4)]
        public void CenteredVarianceRetainsSmallSpreadAroundLargeMean(BackendType backend, int rank)
        {
            var shape = rank == 2 ? new[] { 2, 4 } : rank == 3 ? new[] { 1, 2, 4 } : new[] { 1, 1, 2, 4 };
            var scale = new[] { 1f, 2f, 3f, 4f }; var bias = new[] { .1f, .2f, .3f, .4f };
            var original = (float[])scale.Clone(); var model = Build(scale, bias, 2, shape, .25f);
            using var input = new Tensor<float>(new TensorShape(shape), new[] { 10000f, 10001f, 10002f, 10003f, 10003f, 10002f, 10001f, 10000f });
            var spread = new[] { -1.5, -.5, .5, 1.5, 1.5, .5, -.5, -1.5 };
            var expected = spread.Select((value, index) => (float)(value / Math.Sqrt(1.5) * scale[index % 4] + bias[index % 4])).ToArray();
            if (backend == BackendType.GPUCompute) { Assert.That(SystemInfo.supportsComputeShaders, Is.True); ComputeTensorData.Pin(input); }
            using var worker = new Worker(model, backend); worker.Schedule(input);
            var output = (Tensor<float>)worker.PeekOutput(); Assert.That(output.shape, Is.EqualTo(input.shape)); Assert.That(output.backendType, Is.EqualTo(backend));
            Assert.That(output.DownloadToArray(), Is.EqualTo(expected).Within(2e-5)); Assert.That(scale, Is.EqualTo(original));
            if (backend == BackendType.GPUCompute)
            {
                var fallback = (System.Collections.Generic.HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallback", BindingFlags.Instance | BindingFlags.NonPublic).GetValue(worker);
                Assert.That(model.layers.Count(layer => fallback.Contains(layer.outputs[0])), Is.EqualTo(0));
            }
        }
        [Test] public void InvalidNormContextIsRejected()
        {
            Assert.Throws<ArgumentException>(() => Build(new[] { 1f }, new[] { 0f }, 2, new[] { 1, 4 }, .25f));
            Assert.Throws<ArgumentException>(() => Build(new[] { 1f }, new[] { float.NaN }, 2, new[] { 1, 1 }, .25f));
            Assert.Throws<ArgumentOutOfRangeException>(() => Build(new[] { 1f }, new[] { 0f }, 0, new[] { 1, 1 }, .25f));
            Assert.Throws<ArgumentOutOfRangeException>(() => Build(new[] { 1f }, new[] { 0f }, 2, new[] { 1, 1 }, 0f));
        }
        [Test] public void LargeRowCountUsesOnlySmallScalarAndCoefficientInitializers()
        {
            var model = Build(Enumerable.Repeat(1f, 128).ToArray(), new float[128], 8, new[] { 1, 560, 64, 128 }, 1e-6f);
            Assert.That(model.constants.Max(value => value.lengthBytes), Is.LessThanOrEqualTo(128 * 4));
        }
        [TestCase(BackendType.CPU)] [TestCase(BackendType.GPUCompute)]
        public void NormTransformKeepsDownstreamEdgesAndSourceSerializedBytes(BackendType backend)
        {
            var graph = new FunctionalGraph(); var x = graph.AddInput<float>(new TensorShape(1, 4), "source-input");
            var source = graph.Compile(Functional.LayerNorm(x, Functional.Constant(new[] { 1f, 1f, 1f, 1f }), Functional.Constant(new float[4]), .25f) + 3f);
            using var before = new MemoryStream(); ModelWriter.Save(before, source);
            var candidate = Transform(source);
            Assert.That(candidate.outputs.Select(value => value.index), Is.EqualTo(source.outputs.Select(value => value.index)));
            Assert.That(candidate.outputs.Select(value => value.name), Is.EqualTo(source.outputs.Select(value => value.name)));
            using var input = new Tensor<float>(new TensorShape(1, 4), new[] { 10000f, 10001f, 10002f, 10003f });
            if (backend == BackendType.GPUCompute) ComputeTensorData.Pin(input);
            using var worker = new Worker(candidate, backend); worker.Schedule(input);
            Assert.That(((Tensor<float>)worker.PeekOutput()).DownloadToArray(), Is.EqualTo(new[] { (float)(3 - 1.5 / Math.Sqrt(1.5)), (float)(3 - .5 / Math.Sqrt(1.5)), (float)(3 + .5 / Math.Sqrt(1.5)), (float)(3 + 1.5 / Math.Sqrt(1.5)) }).Within(2e-5));
            using var after = new MemoryStream(); ModelWriter.Save(after, source); Assert.That(after.ToArray(), Is.EqualTo(before.ToArray()));
        }
    }
}
