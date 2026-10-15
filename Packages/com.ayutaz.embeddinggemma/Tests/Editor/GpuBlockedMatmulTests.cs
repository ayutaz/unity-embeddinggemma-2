using System;
using System.Reflection;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEngine;

namespace EmbeddingGemma.Tests
{
    public sealed class GpuBlockedMatmulTests
    {
        static Type RequiredType()
        {
            var type=typeof(AudioFeatureEncoder).Assembly.GetType("EmbeddingGemma.GpuBlockedMatmul");
            Assert.That(type,Is.Not.Null,"A direct GPU block192 kernel is required; expanded FMA graph exceeded the resource bound.");
            return type;
        }

        [TestCase(128,128,0f)] [TestCase(192,128,1f)] [TestCase(192,192,0f)]
        [TestCase(256,192,1f)] [TestCase(384,384,0f)] [TestCase(768,384,1f)]
        public void ExplicitAccumulationBlockPreservesItsOwnResetBoundary(int block,int boundary,float expected)
        {
            Assert.That(SystemInfo.supportsComputeShaders,Is.True,"Actual GPU required; never skip.");
            var coefficients=new float[512];for(var i=0;i<512;i++)coefficients[i]=1;
            var values=new float[512];values[0]=1e8f;values[boundary]=-1e8f;values[boundary+1]=1;
            using var operation=(IDisposable)Activator.CreateInstance(RequiredType(),512,1,coefficients,block);
            using var input=new Tensor<float>(new TensorShape(1,512),values);ComputeTensorData.Pin(input);
            using var output=Execute(operation,input);Assert.That(output.DownloadToArray(),Is.EqualTo(new[]{expected}));
        }
        [TestCase(-1)] [TestCase(0)] [TestCase(4097)]
        public void ExplicitAccumulationBlockRejectsUnboundedDispatchLoops(int block)
        {
            var error=Assert.Throws<TargetInvocationException>(()=>Activator.CreateInstance(RequiredType(),1,1,new[]{1f},block));
            Assert.That(error.InnerException,Is.InstanceOf<ArgumentOutOfRangeException>());
        }

        [Test]
        public void DirectGpuKernelMustExposeADeviceOnlyOwnedOutputContract()
        {
            var type=RequiredType();
            Assert.That(type.GetConstructor(new[]{typeof(int),typeof(int),typeof(float[])}),Is.Not.Null);
            var method=type.GetMethod("Execute",BindingFlags.Public|BindingFlags.Instance,null,new[]{typeof(Tensor<float>)},null);
            Assert.That(method,Is.Not.Null);
            Assert.That(method.ReturnType,Is.EqualTo(typeof(Tensor<float>)));
            Assert.That(typeof(IDisposable).IsAssignableFrom(type),Is.True);
        }

        [Test]
        public void DirectKernelMustPreserveFusedRoundingBlockResetAndTailOnGpu()
        {
            Assert.That(SystemInfo.supportsComputeShaders,Is.True,"An actual GPU is required; never skip.");
            const int rows=2,width=200,columns=3;
            var values=new float[rows*width];var weights=new float[width*columns];
            for(var i=0;i<values.Length;i++)values[i]=(i%23-11)/16f;
            for(var i=0;i<weights.Length;i++)weights[i]=(i%17-8)/32f;
            // Separate multiply/add and continuous accumulation both give wrong answers here.
            values[0]=-1000000f;values[1]=1000000.0625f;
            weights[0]=33f;weights[columns]=33.000003814697265625f;
            for(var k=2;k<width;k++)weights[k*columns]=0;
            for(var k=0;k<width;k++){values[width+k]=1;weights[k*columns+1]=1;}
            values[width]=1000000;weights[1]=100;
            var expected=Oracle(values,weights,rows,width,columns);
            using var operation=Create(width,columns,weights);
            using var input=new Tensor<float>(new TensorShape(1,rows,width),values);
            ComputeTensorData.Pin(input);
            using var output=Execute(operation,input);
            Assert.That(output.backendType,Is.EqualTo(BackendType.GPUCompute));
            Assert.That(output.shape,Is.EqualTo(new TensorShape(1,rows,columns)));
            Assert.That(output.DownloadToArray(),Is.EqualTo(expected));
        }

        [Test]
        public void CpuInputIsRejectedBeforeAnyImplicitTransfer()
        {
            Assert.That(SystemInfo.supportsComputeShaders,Is.True,"An actual GPU is required; never skip.");
            using var operation=Create(2,1,new[]{1f,2f});
            using var cpu=new Tensor<float>(new TensorShape(1,1,2),new[]{3f,4f});
            var error=Assert.Throws<TargetInvocationException>(()=>Execute(operation,cpu));
            Assert.That(error.InnerException,Is.InstanceOf<ArgumentException>());
        }

        [TestCase("shape")]
        [TestCase("null")]
        [TestCase("nonfinite")]
        [TestCase("bound")]
        public void InvalidCoefficientContractsAreRejectedBeforeAllocation(string problem)
        {
            var values=problem=="null"?null:new[]{1f,2f};
            if(problem=="nonfinite")values[0]=float.NaN;
            if(problem=="bound")values[0]=101;
            var error=Assert.Throws<TargetInvocationException>(()=>Create(problem=="shape"?3:2,1,values));
            Assert.That(error.InnerException,Is.InstanceOf<ArgumentException>());
        }

        [Test]
        public void OutputsStayIndependentAfterInvalidInputReuseAndOwnerDispose()
        {
            Assert.That(SystemInfo.supportsComputeShaders,Is.True,"An actual GPU is required; never skip.");
            using var operation=Create(2,1,new[]{1f,2f});
            using var a=new Tensor<float>(new TensorShape(1,1,2),new[]{3f,4f});
            using var b=new Tensor<float>(new TensorShape(1,1,2),new[]{7f,13f});
            using var bad=new Tensor<float>(new TensorShape(1,1,3),new[]{1f,2f,3f});
            ComputeTensorData.Pin(a);ComputeTensorData.Pin(b);ComputeTensorData.Pin(bad);
            using var first=Execute(operation,a);
            var error=Assert.Throws<TargetInvocationException>(()=>Execute(operation,bad));
            Assert.That(error.InnerException,Is.InstanceOf<ArgumentException>());
            using var second=Execute(operation,b);
            operation.Dispose();operation.Dispose();
            Assert.That(first.DownloadToArray(),Is.EqualTo(new[]{11f}));
            Assert.That(second.DownloadToArray(),Is.EqualTo(new[]{33f}));
            var disposed=Assert.Throws<TargetInvocationException>(()=>Execute(operation,a));
            Assert.That(disposed.InnerException,Is.InstanceOf<ObjectDisposedException>());
        }

        static IDisposable Create(int width,int columns,float[] weights)
            => (IDisposable)Activator.CreateInstance(RequiredType(),width,columns,weights);
        static Tensor<float> Execute(IDisposable operation,Tensor<float> input)
            => (Tensor<float>)RequiredType().GetMethod("Execute").Invoke(operation,new object[]{input});

        static float[] Oracle(float[] input,float[] weights,int rows,int width,int columns)
        {
            var result=new float[rows*columns];
            for(var row=0;row<rows;row++)for(var column=0;column<columns;column++)
            {
                float total=0;
                for(var start=0;start<width;start+=192)
                {
                    float partial=0;
                    for(var k=start;k<Math.Min(start+192,width);k++)
                        partial=(float)((double)input[row*width+k]*weights[k*columns+column]+partial);
                    total=start==0?partial:total+partial;
                }
                result[row*columns+column]=total;
            }
            return result;
        }
    }
}
