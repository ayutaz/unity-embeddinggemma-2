using System;
using System.IO;
using System.Linq;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEngine;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class ModelTensorSliceTests
    {
        static byte[] Bytes(Model model){using var stream=new MemoryStream();ModelWriter.Save(stream,model);return stream.ToArray();}
        [TestCase(BackendType.CPU)] [TestCase(BackendType.GPUCompute)]
        public void SliceKeepsResidualRoutingAndBorrowedCoefficientsWhilePruningUnrelatedWork(BackendType backend)
        {
            if(backend==BackendType.GPUCompute)Assert.That(SystemInfo.supportsComputeShaders,Is.True,"Actual GPU required; never skip.");
            var graph=new FunctionalGraph();var input=graph.AddInput<float>(new TensorShape(1,3),"input");
            var hidden=input*Functional.Constant(new[]{2f,3f,4f});var residual=hidden+input;
            var source=graph.Compile(residual,Functional.Exp(input));var before=Bytes(source);
            var slice=ModelTensorSlice.Extract(source,source.outputs[0].index,"selected");
            Assert.That(slice.layers.Any(layer=>layer.GetType().Name=="Exp"),Is.False);
            Assert.That(slice.constants.All(value=>source.constants.Contains(value)),Is.True);
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
