using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEngine;

namespace EmbeddingGemma.Tests
{
    // Small graphs prove Runtime input/output and ownership contracts, not FFT/mel accuracy.
    public sealed class AudioPreprocessorTests
    {
        static AudioPcmInput Input(float value = 0.25f) =>
            new AudioPcmInput(Enumerable.Repeat(value, 321).ToArray());

        static Model SmallModel(string problem = null)
        {
            var graph = new FunctionalGraph();
            var waveform = graph.AddInput<float>(new TensorShape(1, 179200), "waveform");
            var mask = graph.AddInput<int>(new TensorShape(1, 179200), "sample_mask");
            var seed = Functional.ReduceSum(waveform, new[] { 0, 1 }, true) / 179200f;
            var frames = problem == "feature_shape" ? 1118 : 1119;
            var features = seed * Functional.Constant(new TensorShape(1, frames, 128),
                Enumerable.Repeat(1f, frames * 128).ToArray());
            var indices = Enumerable.Range(0, problem == "mask_shape" ? 1118 : 1119)
                .Select(index => 160 + index * 160).ToArray();
            var frameMask = Functional.IndexSelect(mask, 1, Functional.Constant(new TensorShape(indices.Length), indices));
            var model = graph.Compile(problem == "feature_type" ? features.Int() : features,
                problem == "mask_type" ? frameMask.Float() : frameMask);
            return model;
        }

        [Test]
        public void NullModelIsRejectedBeforeWorkerCreation() =>
            Assert.Throws<ArgumentNullException>(() => new AudioPreprocessor(null, BackendType.CPU));

        [TestCase(BackendType.GPUPixel)]
        [TestCase((BackendType)12345)]
        public void UnsupportedBackendIsRejectedExplicitly(BackendType backend) =>
            Assert.Throws<ArgumentOutOfRangeException>(() => new AudioPreprocessor(SmallModel(), backend));

        [TestCase("input_name")]
        [TestCase("mask_name")]
        [TestCase("input_type")]
        [TestCase("mask_type")]
        [TestCase("input_shape")]
        [TestCase("mask_shape")]
        [TestCase("input_count")]
        [TestCase("output_count")]
        [TestCase("duplicate_output_name")]
        public void MismatchedFixedGraphContractIsRejected(string problem)
        {
            var model = SmallModel();
            if (problem == "input_count") model.inputs.RemoveAt(1);
            else if (problem == "output_count") model.outputs.RemoveAt(1);
            else if (problem == "duplicate_output_name")
            {
                var output = model.outputs[1]; output.name = model.outputs[0].name; model.outputs[1] = output;
            }
            else
            {
                var index = problem.StartsWith("mask", StringComparison.Ordinal) ? 1 : 0;
                var input = model.inputs[index];
                if (problem.EndsWith("name", StringComparison.Ordinal)) input.name = "unsupported";
                if (problem.EndsWith("type", StringComparison.Ordinal))
                    input.dataType = index == 0 ? DataType.Int : DataType.Float;
                if (problem.EndsWith("shape", StringComparison.Ordinal)) input.shape = new DynamicTensorShape(1, 179199);
                model.inputs[index] = input;
            }
            Assert.Throws<ArgumentException>(() => new AudioPreprocessor(model, BackendType.CPU));
        }

        [Test]
        public void NullInputIsRejectedWithoutConsumingTheWorker()
        {
            using var preprocessor = new AudioPreprocessor(SmallModel(), BackendType.CPU);
            Assert.Throws<ArgumentNullException>(() => preprocessor.Process(null));
            using var result = preprocessor.Process(Input());
            Assert.That(result, Is.Not.Null);
            Assert.That(result.ValidFrameCount, Is.EqualTo(2));
        }

        [TestCase("feature_shape")]
        [TestCase("mask_shape")]
        [TestCase("feature_type")]
        [TestCase("mask_type")]
        public void InvalidScheduledOutputsAreRejectedRepeatedly(string problem)
        {
            using var preprocessor = new AudioPreprocessor(SmallModel(problem), BackendType.CPU);
            Assert.Throws<InvalidOperationException>(() => preprocessor.Process(Input()));
            Assert.Throws<InvalidOperationException>(() => preprocessor.Process(Input()));
        }

        [TestCase(BackendType.CPU)]
        [TestCase(BackendType.GPUCompute)]
        public void OwnedResultsSurviveWorkerReuseAndDisposal(BackendType backend)
        {
            if (backend == BackendType.GPUCompute)
                Assert.That(SystemInfo.supportsComputeShaders, Is.True, "Actual GPU required; never skip.");
            var preprocessor = new AudioPreprocessor(SmallModel(), backend);
            AudioPreprocessingResult first = null, second = null;
            try
            {
                first = preprocessor.Process(Input());
                Assert.That(first, Is.Not.Null);
                second = preprocessor.Process(Input(0.5f));
                Assert.That(second, Is.Not.Null);
                Assert.That(first.Features, Is.Not.SameAs(second.Features));
                Assert.That(first.FrameMask, Is.Not.SameAs(second.FrameMask));
                Assert.That(first.Features.shape, Is.EqualTo(new TensorShape(1, 1119, 128)));
                Assert.That(first.FrameMask.shape, Is.EqualTo(new TensorShape(1, 1119)));
                Assert.That(first.Features.backendType, Is.EqualTo(backend));
                Assert.That(first.FrameMask.backendType, Is.EqualTo(backend));
                Assert.That(preprocessor.Backend, Is.EqualTo(backend));
                Assert.That(first.ValidFrameCount, Is.EqualTo(2));
                preprocessor.Dispose(); preprocessor.Dispose();
                // Diagnostic readback after Runtime processing verifies independent output ownership.
                Assert.That(first.Features.DownloadToArray(), Is.All.EqualTo(321 * 0.25f / 179200).Within(1e-7));
                Assert.That(second.Features.DownloadToArray(), Is.All.EqualTo(321 * 0.5f / 179200).Within(1e-7));
                Assert.That(first.FrameMask.DownloadToArray(), Is.EqualTo(new[] { 1, 1 }.Concat(new int[1117])));
                var features = first.Features; var mask = first.FrameMask;
                first.Dispose(); first.Dispose();
                Assert.That(features.dataOnBackend, Is.Null); Assert.That(mask.dataOnBackend, Is.Null);
                Assert.Throws<ObjectDisposedException>(() => { var ignored = first.Features; });
                Assert.Throws<ObjectDisposedException>(() => { var ignored = first.FrameMask; });
            }
            finally { first?.Dispose(); second?.Dispose(); preprocessor.Dispose(); }
        }

        [Test]
        public void DisposedRuntimeRejectsBothValidAndNullCalls()
        {
            var preprocessor = new AudioPreprocessor(SmallModel(), BackendType.CPU);
            preprocessor.Dispose(); preprocessor.Dispose();
            Assert.Throws<ObjectDisposedException>(() => preprocessor.Process(Input()));
            Assert.Throws<ObjectDisposedException>(() => preprocessor.Process(null));
        }

        [Test]
        [Category("GPUCompute")]
        public void GpuFeaturesAndIntegerMaskChainWithoutIntermediateCpuReadback()
        {
            Assert.That(SystemInfo.supportsComputeShaders, Is.True, "Actual GPU required; never skip.");
            var model = SmallModel();
            using var preprocessor = new AudioPreprocessor(model, BackendType.GPUCompute);
            using var result = preprocessor.Process(Input());
            Assert.That(result, Is.Not.Null);
            Assert.That(result.Features.backendType, Is.EqualTo(BackendType.GPUCompute));
            Assert.That(result.FrameMask.backendType, Is.EqualTo(BackendType.GPUCompute));
            var worker = (Worker)typeof(AudioPreprocessor).GetField("worker", BindingFlags.NonPublic | BindingFlags.Instance)?.GetValue(preprocessor);
            Assert.That(worker, Is.Not.Null);
            AssertNoCpuFallback(worker, model);
            var graph = new FunctionalGraph();
            var features = graph.AddInput<float>(new TensorShape(1, 1119, 128), "features");
            var mask = graph.AddInput<int>(new TensorShape(1, 1119), "frame_mask");
            var scalar = Functional.ReduceSum(features, new[] { 0, 1, 2 }, true)
                + Functional.ReduceSum(mask.Float(), new[] { 0, 1 }, true);
            var consumer = graph.Compile(scalar);
            using var next = new Worker(consumer, BackendType.GPUCompute);
            next.SetInput("features", result.Features); next.SetInput("frame_mask", result.FrameMask); next.Schedule();
            AssertNoCpuFallback(next, consumer);
            var output = next.PeekOutput() as Tensor<float>;
            Assert.That(output.backendType, Is.EqualTo(BackendType.GPUCompute));
            // Only the final scalar is read; no Features / FrameMask download before chaining.
            Assert.That(output.DownloadToArray()[0], Is.EqualTo(1119 * 128 * (321 * 0.25 / 179200) + 2).Within(0.001));
            Assert.That(result.Features.backendType, Is.EqualTo(BackendType.GPUCompute));
            Assert.That(result.FrameMask.backendType, Is.EqualTo(BackendType.GPUCompute));
        }

        static void AssertNoCpuFallback(Worker worker, Model model)
        {
            var flags = BindingFlags.NonPublic | BindingFlags.Instance;
            var cpu = (HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallback", flags).GetValue(worker);
            Assert.That(cpu.Intersect(model.layers.SelectMany(layer => layer.outputs)
                .Concat(model.inputs.Select(input => input.index))), Is.Empty);
            var flush = (HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallbackShouldFlushGPU", flags).GetValue(worker);
            Assert.That(flush, Is.Empty);
        }
    }
}
