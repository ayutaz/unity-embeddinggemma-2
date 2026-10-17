using System;
using System.Linq;
using System.Reflection;
using System.Runtime.ExceptionServices;
using NUnit.Framework;
using Unity.InferenceEngine;

namespace EmbeddingGemma.Tests
{
    public sealed class GpuSoftmaxDispatchTests
    {
        static GpuSoftmax Create(int lanes, int rows)
        {
            var constructor = typeof(GpuSoftmax).GetConstructor(new[] { typeof(int), typeof(int) });
            Assert.That(constructor, Is.Not.Null, "Softmax needs an explicit bound on rows per GPU dispatch.");
            try { return (GpuSoftmax)constructor.Invoke(new object[] { lanes, rows }); }
            catch (TargetInvocationException error) { ExceptionDispatchInfo.Capture(error.InnerException).Throw(); throw; }
        }
        static int Property(GpuSoftmax operation, string name)
        {
            var property = typeof(GpuSoftmax).GetProperty(name); Assert.That(property, Is.Not.Null, name);
            return (int)property.GetValue(operation);
        }
        [TestCase(8)] [TestCase(16)]
        public void RowTilesKeepEveryOutputBitAndBorrowedInputAcrossPartialFinalTileAndReuse(int lanes)
        {
            var values = Enumerable.Range(0, 5 * 2520).Select(i => i % 2520 < 8 ? 0f : -16.635532f + i % 7 * .125f).ToArray();
            using var input = new Tensor<float>(new TensorShape(1, 5, 2520), values); ComputeTensorData.Pin(input);
            using var tiled = Create(lanes, 2); using var whole = Create(lanes, 5);
            using var actual = tiled.Execute(input); using var expected = whole.Execute(input); using var again = tiled.Execute(input);
            Assert.That(Property(tiled, "LastDispatchCount"), Is.EqualTo(3)); Assert.That(Property(whole, "LastDispatchCount"), Is.EqualTo(1));
            var bits = actual.DownloadToArray().Select(BitConverter.SingleToInt32Bits).ToArray();
            Assert.That(bits, Is.EqualTo(expected.DownloadToArray().Select(BitConverter.SingleToInt32Bits)));
            Assert.That(bits, Is.EqualTo(again.DownloadToArray().Select(BitConverter.SingleToInt32Bits)));
            Assert.That(input.DownloadToArray(), Is.EqualTo(values));
            tiled.Dispose(); Assert.That(actual.DownloadToArray().Select(BitConverter.SingleToInt32Bits), Is.EqualTo(bits));
        }
        [Test] public void ExistingConstructorBoundsLargeRowCountWithoutChangingBackend()
        {
            using var operation = new GpuSoftmax(16);
            Assert.That(Property(operation, "MaximumRowsPerDispatch"), Is.EqualTo(1024));
            var values = Enumerable.Range(0, 1025 * 17).Select(i => (float)(i % 17 - 8)).ToArray();
            using var input = new Tensor<float>(new TensorShape(1025, 17), values); ComputeTensorData.Pin(input);
            using var result = operation.Execute(input);
            Assert.That(Property(operation, "LastDispatchCount"), Is.EqualTo(2));
            Assert.That(result.backendType, Is.EqualTo(BackendType.GPUCompute));
            var actual = result.DownloadToArray();
            for (var row = 0; row < 1025; row++)
                Assert.That(actual.Skip(row * 17).Take(17).Sum(v => (double)v), Is.EqualTo(1).Within(1e-6));
        }
        [TestCase(0)] [TestCase(-1)] public void NonpositiveRowBudgetCannotCreateAnUnboundedSubmissionLoop(int budget)
            => Assert.Throws<ArgumentOutOfRangeException>(() => Create(16, budget));
    }
}
