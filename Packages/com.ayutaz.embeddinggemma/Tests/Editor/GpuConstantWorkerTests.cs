using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.ExceptionServices;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEngine;

namespace EmbeddingGemma.Tests
{
    public sealed class GpuConstantWorkerTests
    {
        static Type Api()
        {var type=typeof(MultimodalBackbone).Assembly.GetType("EmbeddingGemma.GpuConstantWorker");Assert.That(type,Is.Not.Null,"Bounded constant upload is missing.");return type;}
        static IDisposable Create(Model model,BackendType backend=BackendType.GPUCompute,int bytes=8)
        {try{return (IDisposable)Activator.CreateInstance(Api(),model,backend,bytes);}catch(TargetInvocationException error){ExceptionDispatchInfo.Capture(error.InnerException).Throw();throw;}}
        static Worker Worker(IDisposable holder)=>(Worker)holder.GetType().GetProperty("Worker").GetValue(holder);
        static int Value(IDisposable holder,string name)=>(int)holder.GetType().GetProperty(name).GetValue(holder);
        static Model FloatModel()
        {
            var graph=new FunctionalGraph();var input=graph.AddInput<float>(new TensorShape(1,7),"input");
            return graph.Compile(input+Functional.Constant(new TensorShape(1,7),new[] {-1f,-0.25f,0f,1f,3.25f,5f,6f}));
        }
        [TestCase(0)] [TestCase(-4)] [TestCase(3)] [TestCase(6)] [TestCase(16777220)]
        public void InvalidChunkBudgetIsRejected(int bytes)
        {Assert.Throws<ArgumentOutOfRangeException>(()=>Create(FloatModel(),bytes:bytes));}
        [Test] public void NullModelIsRejected()
        {Assert.Throws<ArgumentNullException>(()=>Create(null));}
        [Test] public void UnsupportedBackendIsRejected()
        {Assert.Throws<ArgumentOutOfRangeException>(()=>Create(FloatModel(),(BackendType)99));}
        [TestCase(BackendType.CPU)] [TestCase(BackendType.GPUCompute)]
        public void UnchangedBorrowedModelProducesExactValuesAcrossRepeatedSchedules(BackendType backend)
        {
            if(backend==BackendType.GPUCompute)Assert.That(SystemInfo.supportsComputeShaders,Is.True,"Actual GPU required; never skip.");
            var model=FloatModel();var inputs=model.inputs.ToArray();var constants=model.constants.ToArray();var layers=model.layers.ToArray();
            using var holder=Create(model,backend);var worker=Worker(holder);
            using var input=new Tensor<float>(new TensorShape(1,7),new float[7]);
            for(var repeat=0;repeat<3;repeat++)
            {
                worker.Schedule(input);var output=(Tensor<float>)worker.PeekOutput();
                Assert.That(output.backendType,Is.EqualTo(backend));
                Assert.That(output.DownloadToArray(),Is.EqualTo(new[] {-1f,-0.25f,0f,1f,3.25f,5f,6f}));
            }
            Assert.That(model.inputs,Is.EqualTo(inputs));Assert.That(model.constants,Is.EqualTo(constants));Assert.That(model.layers,Is.EqualTo(layers));
            Assert.That(Value(holder,"UploadedConstantCount"),Is.EqualTo(backend==BackendType.CPU?0:1));
            Assert.That(Value(holder,"UploadCallCount"),Is.EqualTo(backend==BackendType.CPU?0:4));
            Assert.That(Value(holder,"MaximumUploadBytes"),Is.EqualTo(backend==BackendType.CPU?0:8));
            if(backend==BackendType.GPUCompute)
            {
                var cpu=(HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallback",BindingFlags.NonPublic|BindingFlags.Instance).GetValue(worker);
                Assert.That(cpu.Intersect(model.layers.SelectMany(layer=>layer.outputs)),Is.Empty);
                var flush=(HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallbackShouldFlushGPU",BindingFlags.NonPublic|BindingFlags.Instance).GetValue(worker);
                Assert.That(flush,Is.Empty);
            }
        }
        [Test] public void BudgetBoundaryDoesNotUploadSmallConstants()
        {
            using var holder=Create(FloatModel(),bytes:28);
            Assert.That(Value(holder,"UploadedConstantCount"),Is.Zero);Assert.That(Value(holder,"UploadCallCount"),Is.Zero);
        }
        [Test] public void IntegerConstantsKeepAllBitsIncludingSign()
        {
            var graph=new FunctionalGraph();var input=graph.AddInput<int>(new TensorShape(1,7),"input");
            var values=new[] {int.MinValue,-1234567,-1,0,1,1234567,int.MaxValue};
            var model=graph.Compile(input+Functional.Constant(new TensorShape(1,7),values));
            using var holder=Create(model);using var tensor=new Tensor<int>(new TensorShape(1,7),new int[7]);
            Worker(holder).Schedule(tensor);Assert.That(((Tensor<int>)Worker(holder).PeekOutput()).DownloadToArray(),Is.EqualTo(values));
            Assert.That(Value(holder,"MaximumUploadBytes"),Is.LessThanOrEqualTo(8));
        }
        [Test] public void OddFloat16ConstantUsesPackedWordsAndKeepsTheProjection()
        {
            var graph=new FunctionalGraph();var input=graph.AddInput<float>(new TensorShape(1,3),"input");
            var values=Enumerable.Range(1,15).Select(value=>value*0.25f).ToArray();
            var model=graph.Compile(Functional.MatMul(input,Functional.Constant(new TensorShape(3,5),values)));
            ModelQuantizer.QuantizeWeights(QuantizationType.Float16,ref model);
            // Production consumes saved models; serialization normalizes quantizer input indexes.
            using(var serialized=new MemoryStream()) {ModelWriter.Save(serialized,model);serialized.Position=0;model=ModelLoader.Load(serialized);}
            Assert.That(model.constants.Any(value=>value.dataType==DataType.Short&&value.shape.length==15),Is.True,"Real packed half constant required.");
            using var holder=Create(model);using var tensor=new Tensor<float>(new TensorShape(1,3),new[] {1f,2f,3f});
            Worker(holder).Schedule(tensor);
            var expected=Enumerable.Range(0,5).Select(column=>values[column]+2*values[5+column]+3*values[10+column]).ToArray();
            Assert.That(((Tensor<float>)Worker(holder).PeekOutput()).DownloadToArray(),Is.EqualTo(expected));
            Assert.That(Value(holder,"UploadedConstantCount"),Is.GreaterThan(0));Assert.That(Value(holder,"MaximumUploadBytes"),Is.LessThanOrEqualTo(8));
        }
        [Test] public void DisposingOwnerReleasesWorkerAndKeepsBorrowedModelUsable()
        {
            var model=FloatModel();var holder=Create(model);holder.Dispose();holder.Dispose();
            Assert.Throws<TargetInvocationException>(()=>Worker(holder));
            using var baseline=new Worker(model,BackendType.CPU);using var input=new Tensor<float>(new TensorShape(1,7),new float[7]);
            baseline.Schedule(input);Assert.That(((Tensor<float>)baseline.PeekOutput()).DownloadToArray()[6],Is.EqualTo(6));
        }
        [Test] public void NoncontiguousInputIndexesAreRejectedBeforeWorkerCreation()
        {
            var model=FloatModel();var input=model.inputs[0];input.index=17;model.inputs[0]=input;
            Assert.Throws<ArgumentException>(()=>{using var holder=Create(model);});
        }
        [TestCase(BackendType.CPU)] [TestCase(BackendType.GPUCompute)]
        public void BackboneOwnsBoundedUploaderAndRetainsItsOriginalFeatureContract(BackendType backend)
        {
            var graph=new FunctionalGraph();var ids=graph.AddInput<int>(new TensorShape(1,8),"input_ids");
            var mask=graph.AddInput<int>(new TensorShape(1,8),"attention_mask");
            var image=graph.AddInput<float>(new TensorShape(1,512),"image_features");
            var audio=graph.AddInput<float>(new TensorShape(0,512),"audio_features");
            var source=Functional.ReduceSum((ids+mask).Float(),new[] {0,1},true)+
                Functional.ReduceSum(image,new[] {0,1},true)/512f+Functional.ReduceSum(audio,new[] {0,1},true);
            const int count=2097153;
            var weights=Enumerable.Range(0,count).Select(index=>0.25f+((uint)index*1664525u+1013904223u)%65536/65536f).ToArray();
            var seed=Functional.ReduceSum(source*Functional.Constant(new TensorShape(1,count),weights),1,true)/count;
            var first=new float[768];first[0]=1;var second=new float[768];second[1]=1;
            var vector=seed*Functional.Constant(new TensorShape(1,768),first)+Functional.Constant(new TensorShape(1,768),second);
            var model=graph.Compile(vector/Functional.Sqrt(Functional.ReduceSumSquare(vector,1,true)));
            Assert.That(model.constants.Any(value=>value.lengthBytes>8*1024*1024),Is.True,"A real default-budget crossing is required.");
            var profile=new MultimodalBackboneProfile(8,10,11,1,0,32,0);
            using var backbone=new MultimodalBackbone(model,profile,backend);
            var owner=typeof(MultimodalBackbone).GetField("workerOwner",BindingFlags.NonPublic|BindingFlags.Instance);
            Assert.That(owner,Is.Not.Null,"Production backbone must use bounded uploads.");
            var holder=(IDisposable)owner.GetValue(backbone);
            Assert.That(Value(holder,"UploadedConstantCount"),Is.EqualTo(backend==BackendType.CPU?0:1));
            Assert.That(Value(holder,"MaximumUploadBytes"),Is.LessThanOrEqualTo(8*1024*1024));
            using var features=new Tensor<float>(new TensorShape(1,512),Enumerable.Repeat(0.5f,512).ToArray());
            if(backend==BackendType.GPUCompute)ComputeTensorData.Pin(features);
            using var empty=new Tensor<float>(new TensorShape(0,512));
            var result=backbone.Embed(new[] {10,0,0,0,0,0,0,0},new[] {1,0,0,0,0,0,0,0},features,empty);
            Assert.That(result.Length,Is.EqualTo(768));Assert.That(result.Sum(value=>(double)value*value),Is.EqualTo(1).Within(1e-6));
        }
    }
}
