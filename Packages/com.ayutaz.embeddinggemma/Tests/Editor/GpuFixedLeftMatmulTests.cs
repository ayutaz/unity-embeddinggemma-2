using System;
using System.Reflection;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEngine;

namespace EmbeddingGemma.Tests
{
    public sealed class GpuFixedLeftMatmulTests
    {
        static Type OperationType()
        {
            var type=typeof(IGpuTensorOperation).Assembly.GetType("EmbeddingGemma.GpuFixedLeftMatmul");
            Assert.That(type,Is.Not.Null,"Owned fixed-left native GPU matmul is missing.");return type;
        }
        static IGpuTensorOperation Create(TensorShape shape,float[] coefficients)
            =>(IGpuTensorOperation)Activator.CreateInstance(OperationType(),shape,coefficients);

        [TestCase(2)] [TestCase(3)]
        public void OwnsCoefficientsBorrowsInputAndReturnsIndependentReusableOutput(int rank)
        {
            Assert.That(SystemInfo.supportsComputeShaders,Is.True,"Actual GPU required; no skip.");
            var leftShape=rank==2?new TensorShape(2,3):new TensorShape(1,2,3);
            var inputShape=rank==2?new TensorShape(3,2):new TensorShape(1,3,2);
            var expectedShape=rank==2?new TensorShape(2,2):new TensorShape(1,2,2);
            var coefficients=new[]{1f,0f,2f,-1f,3f,.5f};
            using var operation=Create(leftShape,coefficients);Array.Clear(coefficients,0,coefficients.Length);
            var values=new[]{1f,2f,3f,4f,5f,6f};using var input=new Tensor<float>(inputShape,values);ComputeTensorData.Pin(input);
            using var first=operation.Execute(input);using var second=operation.Execute(input);
            Assert.That(first.backendType,Is.EqualTo(BackendType.GPUCompute));Assert.That(first.shape,Is.EqualTo(expectedShape));
            Assert.That(first.DownloadToArray(),Is.EqualTo(new[]{11f,14f,10.5f,13f}));
            Assert.That(second.DownloadToArray(),Is.EqualTo(first.DownloadToArray()));Assert.That(first,Is.Not.SameAs(second));
            Assert.That(input.DownloadToArray(),Is.EqualTo(values));operation.Dispose();operation.Dispose();
            Assert.That(first.DownloadToArray(),Is.EqualTo(new[]{11f,14f,10.5f,13f}));
            Assert.Throws<ObjectDisposedException>(()=>operation.Execute(input));
        }

        [Test]
        public void InvalidInputDoesNotPoisonTheOwnedOperationOrBorrowedTensors()
        {
            using var operation=Create(new TensorShape(2,3),new[]{1f,0f,2f,-1f,3f,.5f});
            using var cpu=new Tensor<float>(new TensorShape(3,2),new[]{1f,2f,3f,4f,5f,6f});
            Assert.Throws<ArgumentException>(()=>operation.Execute(cpu));Assert.Throws<ArgumentNullException>(()=>operation.Execute(null));
            using var wrong=new Tensor<float>(new TensorShape(2,2),new[]{1f,2f,3f,4f});ComputeTensorData.Pin(wrong);
            Assert.Throws<ArgumentException>(()=>operation.Execute(wrong));
            using var valid=new Tensor<float>(new TensorShape(3,2),new[]{1f,2f,3f,4f,5f,6f});ComputeTensorData.Pin(valid);
            using var result=operation.Execute(valid);Assert.That(result.DownloadToArray(),Is.EqualTo(new[]{11f,14f,10.5f,13f}));
            Assert.That(cpu.DownloadToArray(),Is.EqualTo(new[]{1f,2f,3f,4f,5f,6f}));
        }

        [TestCase("rank")] [TestCase("batch")] [TestCase("length")] [TestCase("nan")]
        public void RejectsInvalidFixedCoefficientsBeforeAllocating(string problem)
        {
            var shape=problem=="rank"?new TensorShape(3):problem=="batch"?new TensorShape(2,1,3):new TensorShape(1,1,3);
            var values=problem=="length"?new[]{1f}:problem=="nan"?new[]{1f,float.NaN,2f}:new[]{1f,2f,3f};
            var error=Assert.Throws<TargetInvocationException>(()=>Create(shape,values));Assert.That(error.InnerException,Is.InstanceOf<ArgumentException>());
        }
    }
}
