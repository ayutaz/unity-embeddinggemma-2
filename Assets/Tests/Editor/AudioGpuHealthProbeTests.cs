using System;
using System.Linq;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEngine;

namespace EmbeddingGemma.Tests
{
    public sealed class AudioGpuHealthProbeTests
    {
        [TestCase(BackendType.CPU)] [TestCase(BackendType.GPUCompute)]
        public void ZeroInputProducesMelFloorLogAndUnitOutput(BackendType backend)
        {
            if(backend==BackendType.GPUCompute)Assert.That(SystemInfo.supportsComputeShaders,Is.True);
            var graph=new FunctionalGraph();var input=graph.AddInput<float>(new TensorShape(1,4),"input");
            var model=graph.Compile(Functional.Log(Functional.Max(input,Functional.Constant(0.001f))),input+1f);
            using var worker=new Worker(model,backend);using var tensor=new Tensor<float>(new TensorShape(1,4),new float[4]);
            if(backend==BackendType.GPUCompute)ComputeTensorData.Pin(tensor);
            for(var pass=0;pass<3;pass++)
            {
                worker.Schedule(tensor);
                var log=(Tensor<float>)worker.PeekOutput(model.outputs[0].name);var unit=(Tensor<float>)worker.PeekOutput(model.outputs[1].name);
                Assert.That(log.backendType,Is.EqualTo(backend));Assert.That(unit.backendType,Is.EqualTo(backend));
                Assert.That(log.DownloadToArray(),Is.All.EqualTo((float)Math.Log(0.001)).Within(1e-5));
                Assert.That(unit.DownloadToArray(),Is.All.EqualTo(1f));
            }
        }
    }
}
