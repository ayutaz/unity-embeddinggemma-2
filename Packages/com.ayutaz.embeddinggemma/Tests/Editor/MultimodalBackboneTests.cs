using System;
using System.Linq;
using System.Collections.Generic;
using System.Reflection;
using NUnit.Framework;
using Unity.InferenceEngine;

namespace EmbeddingGemma.Tests
{
    // Small Sentis graphs verify the API / ownership contract, not real encoders.
    public sealed class MultimodalBackboneTests
    {
        static MultimodalBackboneProfile Profile() => new MultimodalBackboneProfile(6, 12, 13, 2, 1, 8, 0);
        static int[] Ids() => new[] { 1, 12, 13, 2, 12, 0 };
        static int[] Mask() => new[] { 1, 1, 1, 1, 1, 0 };
        static Tensor<float> Image(float value = 0.1f) => new Tensor<float>(new TensorShape(2, 512), Enumerable.Repeat(value, 1024).ToArray());
        static Tensor<float> Audio() => new Tensor<float>(new TensorShape(1, 512), Enumerable.Repeat(0.2f, 512).ToArray());

        static Model SmallModel(int length = 6, int width = 512, bool normalized = true,
            bool fullShape = true, bool nonfinite = false)
        {
            var graph = new FunctionalGraph();
            var ids = graph.AddInput<int>(new TensorShape(1, length), "input_ids");
            var mask = graph.AddInput<int>(new TensorShape(1, length), "attention_mask");
            var image = graph.AddInput<float>(new TensorShape(2, width), "image_features");
            var audio = graph.AddInput<float>(new TensorShape(1, width), "audio_features");
            var seed = Functional.ReduceSum((ids + mask).Float(), new[] { 0, 1 }, true)
                + Functional.ReduceSum(image, new[] { 0, 1 }, true) / 1000f
                + Functional.ReduceSum(audio, new[] { 0, 1 }, true) / 1000f;
            var size = fullShape ? 768 : 3;
            var first = new float[size]; first[0] = 1;
            var second = new float[size]; second[1] = 1;
            var vector = seed * Functional.Constant(new TensorShape(1, size), first)
                + Functional.Constant(new TensorShape(1, size), second);
            if (nonfinite) vector /= 0f;
            else if (normalized) vector /= Functional.Sqrt(Functional.ReduceSumSquare(vector, 1, true));
            return graph.Compile(vector);
        }

        [Test]
        public void AcceptsOrderedOovPlaceholdersWithMaskAndPreservesProfile()
        {
            var profile = Profile();
            Assert.DoesNotThrow(() => profile.ValidateTokens(Ids(), Mask()));
            Assert.That(profile.ImageFeatureCount, Is.EqualTo(2));
            Assert.That(profile.AudioFeatureCount, Is.EqualTo(1));
            Assert.That(profile.SequenceLength, Is.EqualTo(6));
        }

        [TestCase("length")]
        [TestCase("negative_count")]
        [TestCase("budget")]
        [TestCase("same_ids")]
        [TestCase("pad_id")]
        [TestCase("vocabulary")]
        public void InvalidProfilesAreRejected(string problem)
        {
            Assert.Throws<ArgumentException>(() => new MultimodalBackboneProfile(
                problem == "length" ? 1 : 6,
                problem == "pad_id" ? 0 : 12,
                problem == "same_ids" ? 12 : 13,
                problem == "negative_count" ? -1 : problem == "budget" ? 6 : 2,
                1, problem == "vocabulary" ? 0 : 8, 0));
        }

        [TestCase("shape")]
        [TestCase("nonbinary")]
        [TestCase("empty_mask")]
        [TestCase("masked_feature")]
        [TestCase("missing_image")]
        [TestCase("extra_audio")]
        [TestCase("invalid_id")]
        public void InvalidTokenLayoutsAreRejectedBeforeInference(string problem)
        {
            var ids = Ids(); var mask = Mask();
            if (problem == "shape") ids = ids.Take(5).ToArray();
            if (problem == "nonbinary") mask[0] = 2;
            if (problem == "empty_mask") Array.Clear(mask, 0, mask.Length);
            if (problem == "masked_feature") mask[1] = 0;
            if (problem == "missing_image") ids[1] = 1;
            if (problem == "extra_audio") ids[0] = 13;
            if (problem == "invalid_id") ids[3] = 99;
            Assert.Throws<ArgumentException>(() => Profile().ValidateTokens(ids, mask));
        }

        [Test]
        public void NullLayoutsAreRejected()
        {
            Assert.Throws<ArgumentNullException>(() => Profile().ValidateTokens(null, Mask()));
            Assert.Throws<ArgumentNullException>(() => Profile().ValidateTokens(Ids(), null));
        }

        [Test]
        public void ConstructorRejectsNullAndUnsupportedOrMismatchedModels()
        {
            Assert.Throws<ArgumentNullException>(() => new MultimodalBackbone(null, Profile(), BackendType.CPU));
            Assert.Throws<ArgumentNullException>(() => new MultimodalBackbone(SmallModel(), null, BackendType.CPU));
            Assert.Throws<ArgumentOutOfRangeException>(() => new MultimodalBackbone(SmallModel(), Profile(), BackendType.GPUPixel));
            Assert.Throws<ArgumentException>(() => new MultimodalBackbone(new Model(), Profile(), BackendType.CPU));
            Assert.Throws<ArgumentException>(() => new MultimodalBackbone(SmallModel(length: 7), Profile(), BackendType.CPU));
            Assert.Throws<ArgumentException>(() => new MultimodalBackbone(SmallModel(width: 256), Profile(), BackendType.CPU));
        }

        [Test]
        public void ReusesWorkerAndCopiesOnlyFinalVectorsWhileBorrowingFeatures()
        {
            using var image = Image(); using var audio = Audio();
            var backbone = new MultimodalBackbone(SmallModel(), Profile(), BackendType.CPU);
            var first = backbone.Embed(Ids(), Mask(), image, audio);
            var snapshot = first.ToArray();
            var seed = 45 + 1024 * 0.1 / 1000 + 512 * 0.2 / 1000;
            Assert.That(first, Has.Length.EqualTo(768));
            Assert.That(first[1], Is.EqualTo(1 / Math.Sqrt(seed * seed + 1)).Within(1e-6));
            using var changed = Image(0.8f);
            var second = backbone.Embed(Ids(), Mask(), changed, audio);
            Assert.That(first, Is.EqualTo(snapshot));
            Assert.That(second, Is.Not.EqualTo(first));
            Assert.That(second.Sum(value => (double)value * value), Is.EqualTo(1).Within(0.002));
            backbone.Dispose(); backbone.Dispose();
            Assert.That(image.DownloadToArray()[0], Is.EqualTo(0.1f));
            Assert.That(audio.DownloadToArray()[0], Is.EqualTo(0.2f));
            Assert.Throws<ObjectDisposedException>(() => backbone.Embed(Ids(), Mask(), image, audio));
        }

        [TestCase(true, false, false)]
        [TestCase(false, true, false)]
        [TestCase(false, true, true)]
        public void InvalidFinalVectorIsRejected(bool normalized, bool fullShape, bool nonfinite)
        {
            using var image = Image(); using var audio = Audio();
            using var backbone = new MultimodalBackbone(SmallModel(normalized: normalized,
                fullShape: fullShape, nonfinite: nonfinite), Profile(), BackendType.CPU);
            Assert.Throws<InvalidOperationException>(() => backbone.Embed(Ids(), Mask(), image, audio));
        }

        [Test]
        public void NullAndWrongFeatureShapesAreRejected()
        {
            using var image = Image(); using var audio = Audio();
            using var wrong = new Tensor<float>(new TensorShape(1, 512), new float[512]);
            using var backbone = new MultimodalBackbone(SmallModel(), Profile(), BackendType.CPU);
            Assert.Throws<ArgumentNullException>(() => backbone.Embed(Ids(), Mask(), null, audio));
            Assert.Throws<ArgumentNullException>(() => backbone.Embed(Ids(), Mask(), image, null));
            Assert.Throws<ArgumentException>(() => backbone.Embed(Ids(), Mask(), wrong, audio));
        }

        [Test]
        [Category("GPUCompute")]
        public void GpuChainRejectsCpuResidentNonemptyFeatures()
        {
            using var image = Image(); using var audio = Audio();
            using var backbone = new MultimodalBackbone(SmallModel(), Profile(), BackendType.GPUCompute);
            Assert.Throws<ArgumentException>(() => backbone.Embed(Ids(), Mask(), image, audio));
        }

        [Test]
        [Category("GPUCompute")]
        public void ActualGpuAcceptanceBorrowsDeviceFeaturesWithoutCpuFallback()
        {
            var model = SmallModel();
            using var image = Image(); using var audio = Audio();
            // This upload is the input boundary, before shared-backbone inference.
            ComputeTensorData.Pin(image); ComputeTensorData.Pin(audio);
            Assert.That(image.backendType, Is.EqualTo(BackendType.GPUCompute));
            Assert.That(audio.backendType, Is.EqualTo(BackendType.GPUCompute));
            var backbone = new MultimodalBackbone(model, Profile(), BackendType.GPUCompute);
            try
            {
                var result = backbone.Embed(Ids(), Mask(), image, audio);
                var seed = 45 + 1024 * 0.1 / 1000 + 512 * 0.2 / 1000;
                Assert.That(result[1], Is.EqualTo(1 / Math.Sqrt(seed * seed + 1)).Within(1e-6));
                Assert.That(backbone.Backend, Is.EqualTo(BackendType.GPUCompute));
                Assert.That(image.backendType, Is.EqualTo(BackendType.GPUCompute));
                Assert.That(audio.backendType, Is.EqualTo(BackendType.GPUCompute));
                var worker = (Worker)typeof(MultimodalBackbone).GetField("worker", BindingFlags.NonPublic | BindingFlags.Instance).GetValue(backbone);
                var resident = (HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallback", BindingFlags.NonPublic | BindingFlags.Instance).GetValue(worker);
                Assert.That(resident.Intersect(model.layers.SelectMany(layer => layer.outputs)), Is.Empty);
                var flush = (HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallbackShouldFlushGPU", BindingFlags.NonPublic | BindingFlags.Instance).GetValue(worker);
                Assert.That(flush, Is.Empty);
            }
            finally { backbone.Dispose(); }
            // Diagnostic readback after disposal proves caller ownership only.
            Assert.That(image.DownloadToArray()[0], Is.EqualTo(0.1f));
            Assert.That(audio.DownloadToArray()[0], Is.EqualTo(0.2f));
        }

        [Test]
        public void DisposedFeaturesAreRejectedBeforeSchedule()
        {
            using var image = Image(); using var audio = Audio();
            using var backbone = new MultimodalBackbone(SmallModel(), Profile(), BackendType.CPU);
            image.Dispose();
            Assert.Throws<ArgumentException>(() => backbone.Embed(Ids(), Mask(), image, audio));
        }
    }
}
