using System;
using System.Collections;
using System.Reflection;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEngine;

namespace EmbeddingGemma.Tests
{
    public sealed class GpuProjectionPipelineTests
    {
        sealed class ObservedOperation : IGpuTensorOperation
        {
            readonly GpuBlockedMatmul identity=new GpuBlockedMatmul(2,2,new[]{1f,0f,0f,1f});
            public Action BeforeExecute;
            public Tensor<float> LastOutput;
            public BackendType Backend=>BackendType.GPUCompute;
            public Tensor<float> Execute(Tensor<float> input)
            {
                BeforeExecute?.Invoke();return LastOutput=identity.Execute(input);
            }
            public void Dispose()=>identity.Dispose();
        }

        static GpuProjectionPipeline CachePipeline(IGpuTensorOperation operation,bool release)
        {
            var constructor=typeof(GpuProjectionPipeline).GetConstructor(new[]{typeof(Model[]),typeof(string[][]),typeof(string[]),typeof(IGpuTensorOperation[]),typeof(bool)});
            Assert.That(constructor,Is.Not.Null,"Opt-in unused GPU cache release is missing.");
            var first=new FunctionalGraph();var x=first.AddInput<float>(new TensorShape(1,2),"input");var before=Rename(first.Compile(x,x*x+x),"input","work");
            var last=new FunctionalGraph();var residual=last.AddInput<float>(new TensorShape(1,2),"input");var projected=last.AddInput<float>(new TensorShape(1,2),"projected");
            var after=Rename(last.Compile(projected*projected+residual),"final");
            return (GpuProjectionPipeline)constructor.Invoke(new object[]{new[]{before,after},new[]{new[]{"work"}},new[]{"projected"},new[]{operation},release});
        }
        static int CachedGpuBuffers(GpuProjectionPipeline pipeline,int stage)
        {
            const BindingFlags fields=BindingFlags.Instance|BindingFlags.NonPublic;
            var workers=(Worker[])typeof(GpuProjectionPipeline).GetField("workers",fields).GetValue(pipeline);
            var storage=typeof(Worker).GetField("m_Storage",fields).GetValue(workers[stage]);
            var tensorPool=storage.GetType().GetField("m_TensorPool",fields).GetValue(storage);
            var computePool=tensorPool.GetType().GetField("m_computeMemoryPool",fields).GetValue(tensorPool);
            return ((IDictionary)computePool.GetType().GetField("freeBuffers",fields).GetValue(computePool)).Count;
        }
        [TestCase(false)] [TestCase(true)]
        public void OptInCacheReleaseKeepsLiveResidualOwnedOutputsAndBorrowedOperation(bool release)
        {
            Assert.That(SystemInfo.supportsComputeShaders,Is.True,"Actual GPU required; no skip.");
            using var operation=new ObservedOperation();using var pipeline=CachePipeline(operation,release);
            operation.BeforeExecute=()=>{if(release)Assert.That(CachedGpuBuffers(pipeline,0),Is.Zero);else Assert.That(CachedGpuBuffers(pipeline,0),Is.GreaterThan(0));};
            using var input=new Tensor<float>(new TensorShape(1,2),new[]{-3f,4f});ComputeTensorData.Pin(input);
            using var output=pipeline.Execute(input);using var repeated=pipeline.Execute(input);
            if(release)Assert.That(CachedGpuBuffers(pipeline,1),Is.Zero);else Assert.That(CachedGpuBuffers(pipeline,1),Is.GreaterThan(0));
            pipeline.Dispose();pipeline.Dispose();operation.BeforeExecute=null;
            Assert.That(output.DownloadToArray(),Is.EqualTo(new[]{33f,404f}));Assert.That(repeated.DownloadToArray(),Is.EqualTo(new[]{33f,404f}));
            Assert.That(input.DownloadToArray(),Is.EqualTo(new[]{-3f,4f}));using var borrowed=operation.Execute(input);Assert.That(borrowed.DownloadToArray(),Is.EqualTo(new[]{-3f,4f}));
        }
        [Test]
        public void CacheReleaseRemainsReusableAfterBorrowedOperationThrows()
        {
            Assert.That(SystemInfo.supportsComputeShaders,Is.True,"Actual GPU required; no skip.");
            using var operation=new ObservedOperation();using var pipeline=CachePipeline(operation,true);
            using var input=new Tensor<float>(new TensorShape(1,2),new[]{-3f,4f});ComputeTensorData.Pin(input);
            operation.BeforeExecute=()=>throw new InvalidOperationException("Controlled borrowed operation failure.");Assert.Throws<InvalidOperationException>(()=>pipeline.Execute(input));
            operation.BeforeExecute=null;using var output=pipeline.Execute(input);Assert.That(output.DownloadToArray(),Is.EqualTo(new[]{33f,404f}));
            Assert.That(CachedGpuBuffers(pipeline,0),Is.Zero);Assert.That(CachedGpuBuffers(pipeline,1),Is.Zero);
        }

        [Test]
        public void DeadIntermediatesAreReleasedBeforeLaterGpuOperationsAndBorrowedInputsSurvive()
        {
            Assert.That(SystemInfo.supportsComputeShaders,Is.True,"Actual GPU required; no skip.");
            Model Identity(string input,string output)
            {
                var graph=new FunctionalGraph();var value=graph.AddInput<float>(new TensorShape(1,2),input);
                return Rename(graph.Compile(value),output);
            }
            using var first=new ObservedOperation();using var second=new ObservedOperation();using var third=new ObservedOperation();
            using var input=new Tensor<float>(new TensorShape(1,2),new[]{-3f,4f});ComputeTensorData.Pin(input);
            third.BeforeExecute=()=>
            {
                Assert.That(first.LastOutput.dataOnBackend,Is.Null,"Dead stage output must be released before a later operation.");
                Assert.That(input.dataOnBackend,Is.Not.Null,"Caller input is borrowed.");
            };
            using var pipeline=new GpuProjectionPipeline(
                new[]{Identity("input","x"),Identity("a","b"),Identity("c","d"),Identity("e","final")},
                new[]{"x","b","d"},new[]{"a","c","e"},new IGpuTensorOperation[]{first,second,third});
            using var result=pipeline.Execute(input);using var repeated=pipeline.Execute(input);
            pipeline.Dispose();
            Assert.That(result.DownloadToArray(),Is.EqualTo(new[]{-3f,4f}));
            Assert.That(repeated.DownloadToArray(),Is.EqualTo(new[]{-3f,4f}));
            Assert.That(third.LastOutput.dataOnBackend,Is.Null,"Final stage consumes and releases its operation input.");
            using var reused=first.Execute(input);Assert.That(reused.DownloadToArray(),Is.EqualTo(new[]{-3f,4f}));
        }

        [Test]
        public void BinaryGpuStageCopiesInputNamesAndBorrowsOperation()
        {
            Assert.That(SystemInfo.supportsComputeShaders,Is.True,"Actual GPU required; no skip.");
            var constructor=Required().GetConstructor(new[]{typeof(Model[]),typeof(string[][]),typeof(string[]),typeof(IGpuTensorOperation[])});
            Assert.That(constructor,Is.Not.Null,"Binary GPU stage binding is missing.");
            var graph=new FunctionalGraph();var x=graph.AddInput<float>(new TensorShape(2,3),"input");var prefix=Rename(graph.Compile(x,Functional.Transpose(x,0,1)),"left","right");
            var finalGraph=new FunctionalGraph();var product=finalGraph.AddInput<float>(new TensorShape(2,2),"product");var tail=Rename(finalGraph.Compile(product),"final");
            using var operation=new GpuBatchedMatmul();var names=new[]{new[]{"left","right"}};
            using var pipeline=(IDisposable)constructor.Invoke(new object[]{new[]{prefix,tail},names,new[]{"product"},new IGpuTensorOperation[]{operation}});
            names[0][0]="changed";
            using var input=new Tensor<float>(new TensorShape(2,3),new[]{1f,2f,3f,4f,5f,6f});ComputeTensorData.Pin(input);
            using var output=Execute(pipeline,input);pipeline.Dispose();Assert.That(output.DownloadToArray(),Is.EqualTo(new[]{14f,32f,32f,77f}));
            using var right=new Tensor<float>(new TensorShape(3,2),new[]{1f,4f,2f,5f,3f,6f});ComputeTensorData.Pin(right);
            using var reused=operation.Execute(input,right);Assert.That(reused.backendType,Is.EqualTo(BackendType.GPUCompute));
        }

        [Test]
        public void MixedGpuStagesPreserveResidualOutputsAndBorrowedSiluAfterDispose()
        {
            Assert.That(SystemInfo.supportsComputeShaders,Is.True,"Actual GPU required; no skip.");
            var contract=typeof(GpuSilu).Assembly.GetType("EmbeddingGemma.IGpuTensorOperation");
            Assert.That(contract,Is.Not.Null,"Mixed GPU stage contract is missing.");
            using var projection=new GpuBlockedMatmul(2,2,new[]{1f,0f,0f,1f});using var silu=new GpuSilu();
            var operations=Array.CreateInstance(contract,2);operations.SetValue(projection,0);operations.SetValue(silu,1);
            var pipeline=(IDisposable)Activator.CreateInstance(Required(),Models(),new[]{"input","b"},new[]{"a","c"},operations);
            using var input=new Tensor<float>(new TensorShape(1,2),new[]{-2f,0f});ComputeTensorData.Pin(input);
            using var output=Execute(pipeline,input);pipeline.Dispose();
            Assert.That(output.backendType,Is.EqualTo(BackendType.GPUCompute));Assert.That(output.DownloadToArray(),Is.EqualTo(new[]{-2f,0f}));
            using var borrowed=silu.Execute(input);Assert.That(borrowed.backendType,Is.EqualTo(BackendType.GPUCompute));
        }

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
