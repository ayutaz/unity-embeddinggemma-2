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
