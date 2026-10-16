using System;
using System.Linq;
using System.Reflection;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEngine;

namespace EmbeddingGemma.Tests
{
    public sealed class GpuBatchedMatmulTests
    {
        static Type Required()
        {
            var type=typeof(GpuSilu).Assembly.GetType("EmbeddingGemma.GpuBatchedMatmul");
            Assert.That(type,Is.Not.Null,"Native-order GPU batched matmul is missing.");return type;
        }
        static Tensor<float> Execute(IDisposable operation,Tensor<float> left,Tensor<float> right)
            =>(Tensor<float>)Required().GetMethod("Execute",new[]{typeof(Tensor<float>),typeof(Tensor<float>)}).Invoke(operation,new object[]{left,right});

        [Test]
        public void BroadcastGpuBatchesReturnIndependentOutputsAndPreserveBorrowedInputs()
        {
            Assert.That(SystemInfo.supportsComputeShaders,Is.True,"Actual GPU required; no skip.");
            var a=Enumerable.Range(0,12).Select(i=>(float)(i-5)).ToArray();var b=Enumerable.Range(0,24).Select(i=>(float)(i%7-3)).ToArray();
            using var left=new Tensor<float>(new TensorShape(2,1,2,3),a);using var right=new Tensor<float>(new TensorShape(1,4,3,2),b);
            ComputeTensorData.Pin(left);ComputeTensorData.Pin(right);var operation=(IDisposable)Activator.CreateInstance(Required());
            using var first=Execute(operation,left,right);using var second=Execute(operation,left,right);operation.Dispose();operation.Dispose();
            Assert.That(first.backendType,Is.EqualTo(BackendType.GPUCompute));Assert.That(first.shape,Is.EqualTo(new TensorShape(2,4,2,2)));
            var expected=new float[32];for(var batchA=0;batchA<2;batchA++)for(var batchB=0;batchB<4;batchB++)for(var row=0;row<2;row++)for(var column=0;column<2;column++)
                for(var k=0;k<3;k++)expected[((batchA*4+batchB)*2+row)*2+column]+=a[(batchA*2+row)*3+k]*b[(batchB*3+k)*2+column];
            Assert.That(first.DownloadToArray(),Is.EqualTo(expected));Assert.That(second.DownloadToArray(),Is.EqualTo(expected));
            Assert.That(left.DownloadToArray(),Is.EqualTo(a));Assert.That(right.DownloadToArray(),Is.EqualTo(b));
        }
        [Test]
        public void InvalidDeviceAndBatchShapeDoNotPoisonLaterGpuExecution()
        {
            Assert.That(SystemInfo.supportsComputeShaders,Is.True,"Actual GPU required; no skip.");
            using var operation=(IDisposable)Activator.CreateInstance(Required());
            using var left=new Tensor<float>(new TensorShape(2,2,3),new float[12]);using var right=new Tensor<float>(new TensorShape(1,3,2),new float[6]);
            ComputeTensorData.Pin(left);
            var error=Assert.Throws<TargetInvocationException>(()=>Execute(operation,left,right));Assert.That(error.InnerException,Is.InstanceOf<ArgumentException>());
            ComputeTensorData.Pin(right);using var invalid=new Tensor<float>(new TensorShape(3,3,2),new float[18]);ComputeTensorData.Pin(invalid);
            error=Assert.Throws<TargetInvocationException>(()=>Execute(operation,left,invalid));Assert.That(error.InnerException,Is.InstanceOf<ArgumentException>());
            using var output=Execute(operation,left,right);Assert.That(output.shape,Is.EqualTo(new TensorShape(2,2,2)));Assert.That(output.DownloadToArray(),Is.All.EqualTo(0f));
            operation.Dispose();error=Assert.Throws<TargetInvocationException>(()=>Execute(operation,left,right));Assert.That(error.InnerException,Is.InstanceOf<ObjectDisposedException>());
        }
    }
}
