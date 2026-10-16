using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEngine;

namespace EmbeddingGemma.Tests
{
    // Synthetic graphs test Runtime contracts and device ownership, not real audio accuracy.
    public sealed class AudioFeatureEncoderTests
    {
        static AudioEncoderProfile Profile(int frames) => new AudioEncoderProfile(frames);
        static AudioFeatureEncoder Encoder(Model model, BackendType backend, AudioEncoderProfile profile) =>
            new AudioFeatureEncoder(model, backend, profile);
        static AudioEncodingResult Encode(AudioFeatureEncoder encoder, AudioPreprocessingResult input) => encoder.Encode(input);
        static Tensor<float> Tokens(AudioEncodingResult result) => result.SoftTokens;

        static Model EncoderGraph(int count = 1, string problem = null)
        {
            var graph = new FunctionalGraph();
            var input = graph.AddInput<float>(new TensorShape(1, 1119, 128), "input");
            // Every synthetic feature is identical. Max preserves that value exactly;
            // a large sum introduces backend-dependent rounding into an ownership test.
            var seed = Functional.ReduceMax(input, new[] { 0, 1, 2 }, false);
            var output = seed * Functional.Constant(new TensorShape(problem == "shape" ? count + 1 : count, 512),
                Enumerable.Repeat(1f, (problem == "shape" ? count + 1 : count) * 512).ToArray());
            return graph.Compile(problem == "type" ? output.Int() : output);
        }
        static Model PreprocessorGraph()
        {
            var graph = new FunctionalGraph();
            var pcm = graph.AddInput<float>(new TensorShape(1, 179200), "waveform");
            var mask = graph.AddInput<int>(new TensorShape(1, 179200), "sample_mask");
            var seed = Functional.ReduceSum(pcm, new[] { 0, 1 }, true) / 179200f;
            var features = seed * Functional.Constant(new TensorShape(1, 1119, 128), Enumerable.Repeat(1f, 1119 * 128).ToArray());
            var indices = Enumerable.Range(0, 1119).Select(index => 160 + 160 * index).ToArray();
            return graph.Compile(features, Functional.IndexSelect(mask, 1, Functional.Constant(new TensorShape(1119), indices)));
        }
        static AudioPcmInput Pcm(float value = 0.25f, int count = 321) => new AudioPcmInput(Enumerable.Repeat(value, count).ToArray());

        [TestCase(1, 1)] [TestCase(4, 1)] [TestCase(5, 2)] [TestCase(839, 210)] [TestCase(1119, 280)]
        public void ProfileMatchesTwoStrideTwoSubsamplingStages(int frames, int count)
        {
            var profile = Profile(frames);
            Assert.That(profile.ValidFrameCount, Is.EqualTo(frames));
            Assert.That(profile.FeatureCount, Is.EqualTo(count));
        }
        [TestCase(-1)] [TestCase(0)] [TestCase(1120)]
        public void UnsupportedPrefixIsRejected(int frames) => Assert.Throws<ArgumentOutOfRangeException>(() => Profile(frames));

        [Test]
        public void NullModelAndProfileAreRejected()
        {
            var profile = Profile(2);
            Assert.Throws<ArgumentNullException>(() => Encoder(null, BackendType.CPU, profile));
            Assert.Throws<ArgumentNullException>(() => Encoder(EncoderGraph(), BackendType.CPU, null));
        }
        [TestCase(BackendType.GPUPixel)] [TestCase((BackendType)12345)]
        public void UnsupportedBackendIsRejected(BackendType backend) =>
            Assert.Throws<ArgumentOutOfRangeException>(() => Encoder(EncoderGraph(), backend, Profile(2)));

        [TestCase("name")] [TestCase("type")] [TestCase("shape")] [TestCase("input_count")] [TestCase("output_count")]
        public void WrongFixedModelContractIsRejectedBeforeSchedule(string problem)
        {
            var profile = Profile(2); var model = EncoderGraph();
            if (problem == "input_count") model.inputs.Clear();
            else if (problem == "output_count") model.outputs.Clear();
            else
            {
                var input = model.inputs[0];
                if (problem == "name") input.name = "other";
                if (problem == "type") input.dataType = DataType.Int;
                if (problem == "shape") input.shape = new DynamicTensorShape(1, 1118, 128);
                model.inputs[0] = input;
            }
            Assert.Throws<ArgumentException>(() => Encoder(model, BackendType.CPU, profile));
        }
        [Test]
        public void PrefixMismatchWithTheSameSoftTokenCountIsRejectedAndWorkerRemainsUsable()
        {
            using var encoder = Encoder(EncoderGraph(), BackendType.CPU, Profile(2));
            using var preprocessor = new AudioPreprocessor(PreprocessorGraph(), BackendType.CPU);
            using var wrong = preprocessor.Process(Pcm(count: 481)); // 3 frames still produce one soft token.
            Assert.Throws<ArgumentException>(() => Encode(encoder, wrong));
            Assert.Throws<ArgumentNullException>(() => Encode(encoder, null));
            using var valid = preprocessor.Process(Pcm());
            using var result = Encode(encoder, valid);
            Assert.That(Tokens(result).shape, Is.EqualTo(new TensorShape(1, 512)));
        }
        [TestCase("shape")] [TestCase("type")]
        public void InvalidScheduledOutputIsRejectedRepeatedly(string problem)
        {
            using var encoder = Encoder(EncoderGraph(problem: problem), BackendType.CPU, Profile(2));
            using var preprocessor = new AudioPreprocessor(PreprocessorGraph(), BackendType.CPU);
            using var input = preprocessor.Process(Pcm());
            Assert.Throws<InvalidOperationException>(() => Encode(encoder, input));
            Assert.Throws<InvalidOperationException>(() => Encode(encoder, input));
        }
        [TestCase(BackendType.CPU)] [TestCase(BackendType.GPUCompute)]
        public void ResultsOwnDeviceCopiesAndInputsRemainBorrowed(BackendType backend)
        {
            if (backend == BackendType.GPUCompute) Assert.That(SystemInfo.supportsComputeShaders, Is.True, "Actual GPU required; never skip.");
            using var preprocessor = new AudioPreprocessor(PreprocessorGraph(), backend);
            using var input = preprocessor.Process(Pcm()); using var changed = preprocessor.Process(Pcm(0.5f));
            var encoderModel = EncoderGraph();
            using var encoder = Encoder(encoderModel, backend, Profile(2));
            using var first = Encode(encoder, input); using var second = Encode(encoder, changed);
            Assert.That(Tokens(first), Is.Not.SameAs(Tokens(second)));
            Assert.That(Tokens(first).backendType, Is.EqualTo(backend));
            if (backend == BackendType.GPUCompute)
            {
                var worker = (Worker)encoder.GetType().GetField("worker", BindingFlags.Instance | BindingFlags.NonPublic).GetValue(encoder);
                var model = encoderModel;
                const BindingFlags flags = BindingFlags.Instance | BindingFlags.NonPublic;
                var cpu = (HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallback", flags).GetValue(worker);
                var flush = (HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallbackShouldFlushGPU", flags).GetValue(worker);
                Assert.That(cpu.Intersect(model.layers.SelectMany(layer => layer.outputs).Concat(model.inputs.Select(value => value.index))), Is.Empty);
                Assert.That(flush, Is.Empty);
            }
            encoder.Dispose(); encoder.Dispose();
            Assert.That(Tokens(first).DownloadToArray(), Is.All.EqualTo(321 * 0.25f / 179200).Within(1e-7));
            Assert.That(Tokens(second).DownloadToArray(), Is.All.EqualTo(321 * 0.5f / 179200).Within(1e-7));
            Assert.That(input.Features.dataOnBackend, Is.Not.Null);
            var owned = Tokens(first); first.Dispose(); first.Dispose();
            Assert.That(owned.dataOnBackend, Is.Null);
            Assert.Throws<ObjectDisposedException>(() => Tokens(first));
            Assert.Throws<ObjectDisposedException>(() => Encode(encoder, input));
        }
        [Test]
        public void DisposedPreprocessingResultIsRejected()
        {
            using var encoder = Encoder(EncoderGraph(), BackendType.CPU, Profile(2));
            using var preprocessor = new AudioPreprocessor(PreprocessorGraph(), BackendType.CPU);
            var input = preprocessor.Process(Pcm()); input.Dispose();
            Assert.Throws<ObjectDisposedException>(() => Encode(encoder, input));
        }
        [Test]
        public void CpuInputsAreRejectedByGpuEncoderWithoutImplicitUpload()
        {
            Assert.That(SystemInfo.supportsComputeShaders, Is.True, "Actual GPU required; never skip.");
            using var encoder = Encoder(EncoderGraph(), BackendType.GPUCompute, Profile(2));
            using var preprocessor = new AudioPreprocessor(PreprocessorGraph(), BackendType.CPU);
            using var input = preprocessor.Process(Pcm());
            Assert.Throws<ArgumentException>(() => Encode(encoder, input));
        }
    }
}
