using System;
using System.Linq;
using System.Reflection;
using System.Runtime.ExceptionServices;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEngine;

namespace EmbeddingGemma.Tests
{
    public sealed class CompensatedMatMulTests
    {
        static Model Build(int[] left, int[] right, int chunk)
        {
            var type = AppDomain.CurrentDomain.GetAssemblies().Select(value => value.GetType("EmbeddingGemma.Tests.CompensatedMatMul")).FirstOrDefault(value => value != null);
            var method = type?.GetMethod("Build");
            Assert.That(method, Is.Not.Null, "Dynamic batched FP32 compensated matmul is missing.");
            try { return (Model)method.Invoke(null, new object[] { left, right, chunk }); }
            catch (TargetInvocationException error) { ExceptionDispatchInfo.Capture(error.InnerException).Throw(); throw; }
        }

        [TestCase(BackendType.CPU, 3)] [TestCase(BackendType.GPUCompute, 3)]
        [TestCase(BackendType.CPU, 5)] [TestCase(BackendType.GPUCompute, 5)]
        public void DynamicBatchedOperandsKeepCancellationTermAndBroadcastShape(BackendType backend, int rank)
        {
            var left = rank == 3 ? new[] { 2, 1, 3 } : new[] { 1, 2, 1, 1, 3 };
            var right = rank == 3 ? new[] { 1, 3, 1 } : new[] { 1, 1, 4, 3, 1 };
            var expectedShape = rank == 3 ? new[] { 2, 1, 1 } : new[] { 1, 2, 4, 1, 1 };
            var model = Build(left, right, 1);
            using var a = new Tensor<float>(new TensorShape(left), new[] { 1e8f, 1f, -1e8f, 1f, 1e8f, -1e8f });
            using var b = new Tensor<float>(new TensorShape(right), Enumerable.Repeat(1f, right.Aggregate(1, (x, y) => x * y)).ToArray());
            if (backend == BackendType.GPUCompute)
            {
                Assert.That(SystemInfo.supportsComputeShaders, Is.True, "Actual GPU required.");
                ComputeTensorData.Pin(a); ComputeTensorData.Pin(b);
            }
            using var worker = new Worker(model, backend); worker.Schedule(a, b);
            var output = (Tensor<float>)worker.PeekOutput();
            Assert.That(output.backendType, Is.EqualTo(backend));
            Assert.That(output.shape, Is.EqualTo(new TensorShape(expectedShape)));
            Assert.That(output.DownloadToArray(), Is.EqualTo(Enumerable.Repeat(1f, output.shape.length).ToArray()));
            Assert.That(model.constants.Max(value => value.lengthBytes), Is.LessThanOrEqualTo(64), "No output-sized zeros.");
            if (backend == BackendType.GPUCompute)
            {
                var fallback = (System.Collections.Generic.HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallback", BindingFlags.Instance | BindingFlags.NonPublic).GetValue(worker);
                Assert.That(model.layers.SelectMany(value => value.outputs).Count(fallback.Contains), Is.Zero);
            }
        }

        [Test] public void InvalidContractionAndBroadcastDimensionsAreRejected()
        {
            Assert.Throws<ArgumentException>(() => Build(new[] { 1, 3 }, new[] { 2, 1 }, 1));
            Assert.Throws<ArgumentException>(() => Build(new[] { 2, 1, 3 }, new[] { 3, 3, 1 }, 1));
        }

        [Test] public void EmptyDimensionsAndNonpositiveChunksAreRejected()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => Build(new[] { 1, 0 }, new[] { 0, 1 }, 1));
            Assert.Throws<ArgumentOutOfRangeException>(() => Build(new[] { 1, 3 }, new[] { 3, 1 }, 0));
        }
    }
}
