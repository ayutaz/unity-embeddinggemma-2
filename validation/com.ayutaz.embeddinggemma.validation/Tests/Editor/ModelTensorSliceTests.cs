using System;
using System.IO;
using System.Linq;
using System.Reflection;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEngine;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class ModelTensorSliceTests
    {
        static byte[] Bytes(Model model){using var stream=new MemoryStream();ModelWriter.Save(stream,model);return stream.ToArray();}
        [TestCase(BackendType.CPU)] [TestCase(BackendType.GPUCompute)]
        public void SlicedHighTensorIndexIsRemappedToDenseInputListPosition(BackendType backend)
        {
            if(backend==BackendType.GPUCompute)Assert.That(SystemInfo.supportsComputeShaders,Is.True,"Actual GPU required; never skip.");
            var graph=new FunctionalGraph();var x=graph.AddInput<float>(new TensorShape(1,3),"input");
            var hidden=Functional.Relu(x*2f+3f);var source=graph.Compile(hidden*hidden);var before=Bytes(source);
            var index=source.layers.Single(layer=>layer.GetType().Name=="Relu").outputs[0];Assert.That(index,Is.GreaterThan(1));
            var boundary=new Model.Input{name="external",index=index,dataType=DataType.Float,shape=new DynamicTensorShape(1,3)};
            var slice=ModelTensorSlice.Extract(source,source.outputs[0].index,"selected",new[]{boundary});
            Assert.That(slice.inputs.Single().index,Is.Zero,"Worker named input lookup must use a dense list position.");
            using var value=new Tensor<float>(new TensorShape(1,3),new[]{2f,3f,4f});if(backend==BackendType.GPUCompute)ComputeTensorData.Pin(value);
            using var worker=new Worker(slice,backend);worker.SetInput("external",value);worker.Schedule();
            Assert.That(((Tensor<float>)worker.PeekOutput()).DownloadToArray(),Is.EqualTo(new[]{4f,9f,16f}));
            Assert.That(Bytes(source),Is.EqualTo(before));
        }
        [TestCase(BackendType.CPU)] [TestCase(BackendType.GPUCompute)]
        public void ExternalBoundaryDoesNotRecomputeTheRemovedProjectionAndRetainsResidualInput(BackendType backend)
        {
            if(backend==BackendType.GPUCompute)Assert.That(SystemInfo.supportsComputeShaders,Is.True,"Actual GPU required; never skip.");
            var graph=new FunctionalGraph();var x=graph.AddInput<float>(new TensorShape(1,3),"input");
            var source=graph.Compile(x*2f+x);var before=Bytes(source);
            var add=source.layers.Single(layer=>layer.GetType().Name=="Add");var boundaryIndex=add.inputs.Single(i=>i!=source.inputs[0].index);
            var boundary=new Model.Input{name="external",index=boundaryIndex,dataType=DataType.Float,shape=new DynamicTensorShape(1,3)};
            var slice=ModelTensorSlice.Extract(source,source.outputs[0].index,"selected",new[]{boundary});
            Assert.That(slice.layers.Count,Is.EqualTo(1));Assert.That(slice.layers[0].GetType().Name,Is.EqualTo("Add"));
            using var input=new Tensor<float>(new TensorShape(1,3),new[]{1f,2f,3f});
            using var external=new Tensor<float>(new TensorShape(1,3),new[]{10f,20f,30f});
            if(backend==BackendType.GPUCompute){ComputeTensorData.Pin(input);ComputeTensorData.Pin(external);}
            using var worker=new Worker(slice,backend);worker.SetInput("input",input);worker.SetInput("external",external);worker.Schedule();
            Assert.That(((Tensor<float>)worker.PeekOutput()).DownloadToArray(),Is.EqualTo(new[]{11f,22f,33f}));
            Assert.That(Bytes(source),Is.EqualTo(before));
        }
        [TestCase(BackendType.CPU)] [TestCase(BackendType.GPUCompute)]
        public void SliceKeepsResidualRoutingAndBorrowedCoefficientsWhilePruningUnrelatedWork(BackendType backend)
        {
            if(backend==BackendType.GPUCompute)Assert.That(SystemInfo.supportsComputeShaders,Is.True,"Actual GPU required; never skip.");
            var graph=new FunctionalGraph();var input=graph.AddInput<float>(new TensorShape(1,3),"input");
            var hidden=input*Functional.Constant(new[]{2f,3f,4f});var residual=hidden+input;
            var source=graph.Compile(residual,Functional.Exp(input));var before=Bytes(source);
            var slice=ModelTensorSlice.Extract(source,source.outputs[0].index,"selected");
            Assert.That(slice.layers.Any(layer=>layer.GetType().Name=="Exp"),Is.False);
            var bytesField=typeof(Constant).GetField("array",BindingFlags.NonPublic|BindingFlags.Instance);
            Assert.That(slice.constants.All(value=>source.constants.Any(original=>
                ReferenceEquals(((ArraySegment<byte>)bytesField.GetValue(value)).Array,((ArraySegment<byte>)bytesField.GetValue(original)).Array))),Is.True);
            using var value=new Tensor<float>(new TensorShape(1,3),new[]{1f,2f,3f});
            if(backend==BackendType.GPUCompute)ComputeTensorData.Pin(value);
            using var worker=new Worker(slice,backend);worker.Schedule(value);
            Assert.That(((Tensor<float>)worker.PeekOutput()).DownloadToArray(),Is.EqualTo(new[]{3f,8f,15f}));
            Assert.That(Bytes(source),Is.EqualTo(before));
            Assert.Throws<ArgumentException>(()=>ModelTensorSlice.Extract(source,int.MaxValue,"missing"));
            Assert.That(Bytes(source),Is.EqualTo(before));
        }
    }
}
