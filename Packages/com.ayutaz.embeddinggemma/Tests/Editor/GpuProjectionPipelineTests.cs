using System;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEngine;

namespace EmbeddingGemma.Tests
{
    public sealed class GpuProjectionPipelineTests
    {
        static Type Required()
        {
            var type=typeof(GpuBlockedMatmul).Assembly.GetType("EmbeddingGemma.GpuProjectionPipeline");
            Assert.That(type,Is.Not.Null,"GPU segment/projection composition is missing.");return type;
        }
        static IDisposable Create(Model[] models,GpuBlockedMatmul[] projections)
            =>(IDisposable)Activator.CreateInstance(Required(),models,new[]{"input","b"},new[]{"a","c"},projections);
        static Tensor<float> Execute(IDisposable pipeline,Tensor<float> input)
            =>(Tensor<float>)Required().GetMethod("Execute").Invoke(pipeline,new object[]{input});
        static Model Rename(Model model,params string[] names)
        {
            for(var i=0;i<names.Length;i++){var output=model.outputs[i];output.name=names[i];model.outputs[i]=output;}return model;
        }
        static Model[] Models()
        {
            var shape=new TensorShape(1,2);
            var a=new FunctionalGraph();var x=a.AddInput<float>(shape,"input");var prefix=Rename(a.Compile(x),"input");
            var b=new FunctionalGraph();var residual=b.AddInput<float>(shape,"input");var hidden=b.AddInput<float>(shape,"a");
            var middle=Rename(b.Compile(residual,Functional.Relu(hidden)),"input","b");
            var c=new FunctionalGraph();var carry=c.AddInput<float>(shape,"input");var projected=c.AddInput<float>(shape,"c");
            var tail=Rename(c.Compile(carry+projected),"final");return new[]{prefix,middle,tail};
        }
        [Test]
        public void GpuSegmentsPreserveResidualAndOwnOutputsAcrossReuseAndDispose()
        {
            Assert.That(SystemInfo.supportsComputeShaders,Is.True,"Actual GPU required; no skip.");
            using var first=new GpuBlockedMatmul(2,2,new[]{1f,0f,0f,2f});
            using var second=new GpuBlockedMatmul(2,2,new[]{3f,0f,0f,4f});
            var pipeline=Create(Models(),new[]{first,second});
            using var input=new Tensor<float>(new TensorShape(1,2),new[]{-3f,4f});ComputeTensorData.Pin(input);
            using var result=Execute(pipeline,input);using var other=Execute(pipeline,input);
            pipeline.Dispose();pipeline.Dispose();
            Assert.That(result.backendType,Is.EqualTo(BackendType.GPUCompute));
            Assert.That(result.DownloadToArray(),Is.EqualTo(new[]{-3f,36f}));
            Assert.That(other.DownloadToArray(),Is.EqualTo(new[]{-3f,36f}));
            using var borrowed=first.Execute(input);Assert.That(borrowed.backendType,Is.EqualTo(BackendType.GPUCompute));
        }
        [Test]
        public void ProjectionCountMismatchIsRejectedBeforeCreatingWorkers()
        {
            using var projection=new GpuBlockedMatmul(2,2,new[]{1f,0f,0f,1f});
            var error=Assert.Throws<System.Reflection.TargetInvocationException>(()=>Create(Models(),new[]{projection}));
            Assert.That(error.InnerException,Is.InstanceOf<ArgumentException>());
        }
    }
}
