using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEngine;

namespace EmbeddingGemma.Tests
{
    public sealed class GpuPipelineConstantUploadTests
    {
        [TestCase(false)] [TestCase(true)]
        public void LargeGpuConstantIsResidentBeforeSchedulingAndKeepsRepeatedResults(bool releaseCache)
        {
            Assert.That(SystemInfo.supportsComputeShaders, Is.True, "Actual GPU required; no skip.");
            const int count = 2097153; // Crosses the production 8 MiB transfer budget.
            var weights = Enumerable.Range(0, count).Select(i => .25f + (i % 16) / 16f).ToArray();
            var graph = new FunctionalGraph();
            var x = graph.AddInput<float>(new TensorShape(1, count), "input");
            var model = graph.Compile(Functional.ReduceSum(x * Functional.Constant(new TensorShape(1, count), weights), 1, true));
            var large = model.constants.Where(c => c.lengthBytes > GpuConstantWorker.DefaultChunkBytes).ToArray();
            Assert.That(large.Length, Is.EqualTo(1), "Exercise a genuine default-budget crossing.");
            var layers = model.layers.ToArray(); var constants = model.constants.ToArray();
            using var pipeline = new GpuProjectionPipeline(new[] {model}, Array.Empty<string[]>(),
                Array.Empty<string>(), Array.Empty<IGpuTensorOperation>(), releaseCache);
            const BindingFlags fields = BindingFlags.Instance | BindingFlags.NonPublic;
            var workers = (Worker[])typeof(GpuProjectionPipeline).GetField("workers", fields).GetValue(pipeline);
            var storage = typeof(Worker).GetField("m_Storage", fields).GetValue(workers[0]);
            var get = storage.GetType().GetMethod("GetTensor", new[] {typeof(int)});
            foreach (var constant in large)
                Assert.That(((Tensor)get.Invoke(storage, new object[] {constant.index})).backendType,
                    Is.EqualTo(BackendType.GPUCompute), "Large constants must be transferred before lazy scheduling can exceed D3D12 upload capacity.");
            var values = new float[count]; values[0] = 1;
            using var input = new Tensor<float>(new TensorShape(1, count), values); ComputeTensorData.Pin(input);
            for (var repeat = 0; repeat < 2; repeat++)
            {
                using var result = pipeline.Execute(input);
                Assert.That(result.backendType, Is.EqualTo(BackendType.GPUCompute));
                Assert.That(result.DownloadToArray(), Is.EqualTo(new[] {.25f}));
            }
            Assert.That(model.layers, Is.EqualTo(layers)); Assert.That(model.constants, Is.EqualTo(constants));
            pipeline.Dispose(); pipeline.Dispose();
            Assert.That(input.backendType, Is.EqualTo(BackendType.GPUCompute));
        }
    }
}
