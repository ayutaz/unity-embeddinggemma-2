using System;
using System.Linq;
using System.Reflection;
using System.Runtime.ExceptionServices;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEngine;

namespace EmbeddingGemma.Tests
{
    public sealed class CompensatedProjectionTests
    {
        static Model Build(float[] weights, int width, int columns, int chunk)
        {
            var type = AppDomain.CurrentDomain.GetAssemblies().Select(assembly => assembly.GetType("EmbeddingGemma.Tests.CompensatedProjection")).FirstOrDefault(value => value != null);
            Assert.That(type, Is.Not.Null, "Sentis FP32 compensated projection is missing.");
            try { return (Model)type.GetMethod("Build").Invoke(null, new object[] { weights, width, columns, chunk }); }
            catch (TargetInvocationException error) { ExceptionDispatchInfo.Capture(error.InnerException).Throw(); throw; }
        }

        [TestCase(BackendType.CPU)] [TestCase(BackendType.GPUCompute)]
        public void CancellationKeepsUnitTermWithoutDoubleOrReadback(BackendType backend)
        {
            var model = Build(new[] { 1f, 1f, 1f }, 3, 1, 1);
            using var input = new Tensor<float>(new TensorShape(1, 3), new[] { 1e8f, 1f, -1e8f });
            if (backend == BackendType.GPUCompute) { Assert.That(SystemInfo.supportsComputeShaders, Is.True); ComputeTensorData.Pin(input); }
            using var worker = new Worker(model, backend); worker.Schedule(input);
            var output = (Tensor<float>)worker.PeekOutput();
            Assert.That(output.backendType, Is.EqualTo(backend));
            if (backend == BackendType.GPUCompute)
            {
                var fallback = (System.Collections.Generic.HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallback", BindingFlags.Instance | BindingFlags.NonPublic).GetValue(worker);
                Assert.That(model.layers.Count(layer => fallback.Contains(layer.outputs[0])), Is.EqualTo(0));
            }
            Assert.That(output.DownloadToArray(), Is.EqualTo(new[] { 1f }));
        }

        [TestCase(BackendType.CPU)] [TestCase(BackendType.GPUCompute)]
        public void OddTailAndMultipleColumnsMatchIndependentIntegerProducts(BackendType backend)
        {
            var weights = new[] { 1f, -2f, 3f, -4f, 5f, -6f, 7f, -8f, 9f, -10f };
            var before = (float[])weights.Clone(); var model = Build(weights, 5, 2, 2);
            using var input = new Tensor<float>(new TensorShape(1, 5), new[] { 1f, 2f, 3f, 4f, 5f });
            if (backend == BackendType.GPUCompute) { Assert.That(SystemInfo.supportsComputeShaders, Is.True); ComputeTensorData.Pin(input); }
            using var worker = new Worker(model, backend); worker.Schedule(input);
            Assert.That(((Tensor<float>)worker.PeekOutput()).DownloadToArray(), Is.EqualTo(new[] { 95f, -110f }));
            Assert.That(weights, Is.EqualTo(before));
        }

        [Test] public void InvalidSizesAndWeightsAreRejectedBeforeCompilation()
        {
            Assert.Throws<ArgumentException>(() => Build(new[] { 1f }, 3, 1, 1));
            Assert.Throws<ArgumentOutOfRangeException>(() => Build(new[] { 1f }, 1, 1, 0));
            Assert.Throws<ArgumentOutOfRangeException>(() => Build(new float[0], 0, 1, 1));
            Assert.Throws<ArgumentException>(() => Build(new[] { float.NaN }, 1, 1, 1));
        }
    }
}
