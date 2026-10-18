using System;
using System.Reflection;
using System.Runtime.ExceptionServices;
using NUnit.Framework;
using Unity.InferenceEngine;
namespace EmbeddingGemma.Tests
{
    public sealed class GpuOutputTensorTests
    {
        static Tensor<float> Allocate(TensorShape shape)
        {
            var type = typeof(GpuSoftmax).Assembly.GetType("EmbeddingGemma.GpuOutputTensor", true);
            try { return (Tensor<float>)type.GetMethod("Allocate", BindingFlags.Static | BindingFlags.Public).Invoke(null, new object[] { shape }); }
            catch (TargetInvocationException error) when (error.InnerException != null)
            { ExceptionDispatchInfo.Capture(error.InnerException).Throw(); throw; }
        }
        [Test] public void WriteOnlyOutputHasNoCpuStorageToZeroInitializeOrTransfer()
        {
            var shape = new TensorShape(1, 4, 32, 32);
            using var output = Allocate(shape);
            Assert.That(output.shape, Is.EqualTo(shape)); Assert.That(output.dataType, Is.EqualTo(DataType.Float));
            Assert.That(output.dataOnBackend, Is.Null, "The GPU pin must allocate fresh GPU storage, not convert a zero-filled CPU output.");
        }
        [Test] public void SeparateOutputsAreOwnedIndependentlyBeforeGpuStorageIsAttached()
        {
            using var first = Allocate(new TensorShape(2, 3)); using var second = Allocate(new TensorShape(2, 3));
            Assert.That(first, Is.Not.SameAs(second)); first.Dispose();
            Assert.That(second.shape, Is.EqualTo(new TensorShape(2, 3)));
            Assert.That(second.dataOnBackend, Is.Null);
        }
        [Test] public void EmptyOutputIsRejectedBeforeAllocatingStorage()
        { Assert.Throws<ArgumentException>(() => Allocate(new TensorShape(0))); }
    }
}
