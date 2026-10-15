using System;
using System.IO;
using System.Linq;
using System.Reflection;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEngine;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class SavedProjectionPartitionTests
    {
        [TestCase(3)] [TestCase(4)]
        public void DynamicBatchedMatmulKeepsBothGpuOperandsAndOriginalResidual(int rank)
        {
            Assert.That(SystemInfo.supportsComputeShaders,Is.True,"Actual GPU required; no skip.");
            var shape=rank==3?new TensorShape(1,2,2):new TensorShape(1,1,2,2);
            var graph=new FunctionalGraph();var x=graph.AddInput<float>(shape,"input");
            var source=graph.Compile(Functional.MatMul(x+1f,x*2f)+x);var before=Bytes(source);
            var product=source.layers.Single(layer=>layer.GetType().Name=="MatMul");
            var plan=SavedProjectionPartition.Create(source,product.outputs);
            Assert.That(plan.Inputs.Single().Length,Is.EqualTo(2),"Both dynamic operands must cross the GPU boundary.");
            Assert.That(plan.Inputs[0][0],Is.Not.EqualTo(plan.Inputs[0][1]));Assert.That(plan.Weights.Single(),Is.Null);
            Assert.That(plan.Widths.Single(),Is.EqualTo(2));Assert.That(plan.Columns.Single(),Is.EqualTo(2));Assert.That(plan.Rows.Single(),Is.EqualTo(2));
            using var operation=new GpuBatchedMatmul();using var input=new Tensor<float>(shape,new[]{1f,2f,3f,4f});ComputeTensorData.Pin(input);
            using var pipeline=new GpuProjectionPipeline(plan.Models,plan.Inputs,plan.Outputs,new IGpuTensorOperation[]{operation},true);
            using var output=pipeline.Execute(input);Assert.That(output.shape,Is.EqualTo(shape));Assert.That(output.DownloadToArray(),Is.EqualTo(new[]{23f,34f,41f,60f}));
            Assert.That(input.DownloadToArray(),Is.EqualTo(new[]{1f,2f,3f,4f}));Assert.That(Bytes(source),Is.EqualTo(before));
        }

        static byte[] Bytes(Model model){using var stream=new MemoryStream();ModelWriter.Save(stream,model);return stream.ToArray();}
        static Model Fixture()
        {
            var graph=new FunctionalGraph();var input=graph.AddInput<float>(new TensorShape(1,2),"input");
            var a=Functional.MatMul(input,Functional.Constant(new TensorShape(2,2),new[]{1f,2f,3f,4f}));
            var b=Functional.MatMul(Functional.Relu(a),Functional.Constant(new TensorShape(2,2),new[]{2f,0f,0f,2f}));
            return graph.Compile(input+a+b*b);
        }
        static int[] Selected(Model model)=>model.layers.Where(layer=>layer.GetType().Name.StartsWith("MatMul")).Select(layer=>layer.outputs[0]).ToArray();

        [Test]
        public void SavedGeluMarkerRetainsResidualAndOriginalModelBytes()
        {
            Assert.That(SystemInfo.supportsComputeShaders,Is.True,"Actual GPU required; no skip.");
            var graph=new FunctionalGraph();var x=graph.AddInput<float>(new TensorShape(1,2),"input");var source=graph.Compile(Functional.Relu(x)+x);
            var original=source.layers.Single(layer=>layer.GetType().Name=="Relu");
            var type=typeof(Worker).Assembly.GetTypes().Single(value=>value.Name=="GeluFast"&&typeof(Layer).IsAssignableFrom(value));
            var gelu=(Layer)Activator.CreateInstance(type);gelu.inputs=(int[])original.inputs.Clone();gelu.outputs=(int[])original.outputs.Clone();source.layers[source.layers.IndexOf(original)]=gelu;
            var before=Bytes(source);var method=typeof(SavedProjectionPartition).GetMethod("MarkGelu");Assert.That(method,Is.Not.Null,"Saved GELU GPU markers are missing.");
            var marked=((Model model,int[] outputs))method.Invoke(null,new object[]{source});Assert.That(marked.outputs,Is.EqualTo(gelu.outputs));
            var plan=SavedProjectionPartition.Create(marked.model,marked.outputs);
            using var operation=new GpuSequentialScale(2f,new[]{1f,1f});using var input=new Tensor<float>(new TensorShape(1,2),new[]{1f,2f});ComputeTensorData.Pin(input);
            using var pipeline=new GpuProjectionPipeline(plan.Models,plan.Inputs,plan.Outputs,new IGpuTensorOperation[]{operation});using var output=pipeline.Execute(input);
            Assert.That(output.DownloadToArray(),Is.EqualTo(new[]{3f,6f}));Assert.That(Bytes(source),Is.EqualTo(before));
        }
        [Test]
        public void SavedGeluMarkerRejectsModelWithoutGeluAndKeepsSource()
        {
            var source=Fixture();var before=Bytes(source);var method=typeof(SavedProjectionPartition).GetMethod("MarkGelu");Assert.That(method,Is.Not.Null,"Saved GELU GPU markers are missing.");
            var error=Assert.Throws<TargetInvocationException>(()=>method.Invoke(null,new object[]{source}));Assert.That(error.InnerException,Is.TypeOf<ArgumentException>());
            Assert.That(Bytes(source),Is.EqualTo(before));
        }

        [TestCase(2)] [TestCase(4)]
        public void ExplicitUnaryGpuMarkerKeepsTheOriginalResidualAndDoesNotRecomputeItsPlaceholder(int rank)
        {
            Assert.That(SystemInfo.supportsComputeShaders,Is.True,"Actual GPU required; never skip.");
            var shape=rank==4?new TensorShape(1,1,1,2):new TensorShape(1,2);
            var graph=new FunctionalGraph();var x=graph.AddInput<float>(shape,"input");
            var source=graph.Compile(Functional.Relu(x)+x);var original=source.layers.Single(layer=>layer.GetType().Name=="Relu");
            var type=typeof(Worker).Assembly.GetTypes().Single(value=>value.Name=="Identity"&&typeof(Layer).IsAssignableFrom(value));
            var marker=(Layer)Activator.CreateInstance(type);marker.inputs=(int[])original.inputs.Clone();marker.outputs=(int[])original.outputs.Clone();
            source.layers[source.layers.IndexOf(original)]=marker;var before=Bytes(source);
            var plan=SavedProjectionPartition.Create(source,new[]{marker.outputs[0]});
            Assert.That(plan.Weights.Single(),Is.Null,"Explicit markers require the caller's GPU operation, not matrix coefficients.");
            using var operation=new GpuSequentialScale(2f,new[]{1f,1f});
            using var input=new Tensor<float>(shape,new[]{1f,2f});ComputeTensorData.Pin(input);
            using var pipeline=new GpuProjectionPipeline(plan.Models,plan.Inputs,plan.Outputs,new IGpuTensorOperation[]{operation});
            using var output=pipeline.Execute(input);Assert.That(output.shape,Is.EqualTo(shape));Assert.That(output.DownloadToArray(),Is.EqualTo(new[]{3f,6f}));
            Assert.That(Bytes(source),Is.EqualTo(before));Assert.That(input.DownloadToArray(),Is.EqualTo(new[]{1f,2f}));
        }

        [TestCase(BackendType.CPU)] [TestCase(BackendType.GPUCompute)]
        public void MultipleOutputsKeepNamesValuesAndImmutableSource(BackendType backend)
        {
            if(backend==BackendType.GPUCompute)Assert.That(SystemInfo.supportsComputeShaders,Is.True,"Actual GPU required; never skip.");
            var graph=new FunctionalGraph();var x=graph.AddInput<float>(new TensorShape(1,2),"input");
            var source=graph.Compile(x*2f,x*3f,Functional.Exp(x));var before=Bytes(source);
            var slice=ModelTensorSlice.Extract(source,new[]{new Model.Output{name="twice",index=source.outputs[0].index},new Model.Output{name="thrice",index=source.outputs[1].index}},Array.Empty<Model.Input>());
            Assert.That(slice.layers.Any(layer=>layer.GetType().Name=="Exp"),Is.False);
            using var input=new Tensor<float>(new TensorShape(1,2),new[]{2f,3f});if(backend==BackendType.GPUCompute)ComputeTensorData.Pin(input);
            using var worker=new Worker(slice,backend);worker.Schedule(input);
            Assert.That(((Tensor<float>)worker.PeekOutput("twice")).DownloadToArray(),Is.EqualTo(new[]{4f,6f}));
            Assert.That(((Tensor<float>)worker.PeekOutput("thrice")).DownloadToArray(),Is.EqualTo(new[]{6f,9f}));
            Assert.That(Bytes(source),Is.EqualTo(before));
        }

        [Test]
        public void OrderedGpuPartitionsRetainFutureResidualAndBorrowedInput()
        {
            Assert.That(SystemInfo.supportsComputeShaders,Is.True,"Actual GPU required; never skip.");
            var source=Fixture();var before=Bytes(source);var plan=SavedProjectionPartition.Create(source,Selected(source));
            Assert.That(plan.Models.Length,Is.EqualTo(3));Assert.That(plan.Widths,Is.EqualTo(new[]{2,2}));Assert.That(plan.Rows,Is.EqualTo(new[]{1,1}));
            var operations=plan.Weights.Select((weight,i)=>new GpuBlockedMatmul(plan.Widths[i],plan.Columns[i],weight,384)).ToArray();
            try
            {
                using var input=new Tensor<float>(new TensorShape(1,2),new[]{-2f,3f});ComputeTensorData.Pin(input);
                using var pipeline=new GpuProjectionPipeline(plan.Models,plan.Inputs,plan.Outputs,operations.Cast<IGpuTensorOperation>().ToArray());
                using var output=pipeline.Execute(input);pipeline.Dispose();
                Assert.That(output.DownloadToArray(),Is.EqualTo(new[]{201f,267f}));Assert.That(input.DownloadToArray(),Is.EqualTo(new[]{-2f,3f}));
                Assert.That(Bytes(source),Is.EqualTo(before));
            }
            finally{foreach(var operation in operations)operation.Dispose();}
        }

        [TestCase("missing")] [TestCase("duplicate")] [TestCase("reversed")]
        public void InvalidProjectionSelectionFailsWithoutChangingSource(string problem)
        {
            var source=Fixture();var indices=Selected(source);var before=Bytes(source);
            if(problem=="missing")indices[0]=int.MaxValue;
            if(problem=="duplicate")indices[1]=indices[0];
            if(problem=="reversed")Array.Reverse(indices);
            Assert.Throws<ArgumentException>(()=>SavedProjectionPartition.Create(source,indices));
            Assert.That(Bytes(source),Is.EqualTo(before));
        }
    }
}
