using System;
using System.Linq;
using NUnit.Framework;
using Unity.InferenceEngine;

namespace EmbeddingGemma.Tests
{
    // Small models verify API contracts only. Real-model CPU/GPU acceptance is separate.
    public sealed class TextEmbedderTests
    {
        const string TokenizerJson = @"{
          ""added_tokens"": [], ""normalizer"": null,
          ""pre_tokenizer"": {""type"": ""WhitespaceSplit""},
          ""post_processor"": null, ""decoder"": null,
          ""truncation"": {""direction"": ""Right"", ""max_length"": 128, ""strategy"": ""LongestFirst"", ""stride"": 0},
          ""padding"": {""strategy"": {""Fixed"": 128}, ""direction"": ""Right"", ""pad_to_multiple_of"": null, ""pad_id"": 0, ""pad_type_id"": 0, ""pad_token"": ""[UNK]""},
          ""model"": {""type"": ""WordLevel"", ""vocab"": {""[UNK]"": 0, ""hello"": 1, ""world"": 2}, ""unk_token"": ""[UNK]""}
        }";

        static Model SmallModel(bool normalized = true, bool fullShape = true, bool nonfinite = false)
        {
            var graph = new FunctionalGraph();
            var ids = graph.AddInput<int>(new TensorShape(1, 128), "input_ids");
            var mask = graph.AddInput<int>(new TensorShape(1, 128), "attention_mask");
            var values = (ids + mask).Float();
            var vector = fullShape ? Functional.Concat(Enumerable.Repeat(values, 6).ToArray(), 1) : values;
            if (nonfinite) vector /= 0.0f;
            else if (normalized) vector /= Functional.Sqrt(Functional.ReduceSumSquare(vector, 1, true));
            return graph.Compile(vector);
        }

        [Test]
        public void ReusesWorkerAndReturnsIndependentNormalizedVectors()
        {
            using var embedder = new TextEmbedder(SmallModel(), TokenizerJson, BackendType.CPU);
            Assert.That(embedder.Backend, Is.EqualTo(BackendType.CPU));
            var first = embedder.EmbedRaw("hello");
            var snapshot = first.ToArray();
            var second = embedder.EmbedRaw("hello world");
            Assert.That(first, Has.Length.EqualTo(768));
            Assert.That(first, Is.EqualTo(snapshot));
            Assert.That(first[0], Is.EqualTo(1 / Math.Sqrt(6)).Within(1e-5));
            Assert.That(second, Is.Not.EqualTo(first));
            Assert.That(second.Sum(v => (double)v * v), Is.EqualTo(1).Within(1e-3));
        }

        [Test]
        public void DisposeIsIdempotentAndRejectsFurtherInference()
        {
            var embedder = new TextEmbedder(SmallModel(), TokenizerJson, BackendType.CPU);
            embedder.Dispose();
            Assert.DoesNotThrow(() => embedder.Dispose());
            Assert.Throws<ObjectDisposedException>(() => embedder.EmbedRaw("hello"));
        }

        [Test]
        public void NullModelAndTokenizerAreRejected()
        {
            Assert.Throws<ArgumentNullException>(() => new TextEmbedder(null, TokenizerJson, BackendType.CPU));
            Assert.Throws<ArgumentNullException>(() => new TextEmbedder(SmallModel(), null, BackendType.CPU));
        }

        [Test]
        public void UnsupportedBackendAndWrongModelContractAreRejected()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => new TextEmbedder(SmallModel(), TokenizerJson, BackendType.GPUPixel));
            Assert.Throws<ArgumentException>(() => new TextEmbedder(new Model(), TokenizerJson, BackendType.CPU));
        }

        [TestCase(true, false, false)]
        [TestCase(false, true, false)]
        [TestCase(false, true, true)]
        public void WrongShapeNonUnitOrNonFiniteOutputIsRejected(bool normalized, bool fullShape, bool nonfinite)
        {
            using var embedder = new TextEmbedder(SmallModel(normalized, fullShape, nonfinite), TokenizerJson, BackendType.CPU);
            Assert.Throws<InvalidOperationException>(() => embedder.EmbedRaw("hello"));
        }

        [Test]
        public void TokenizerWithoutFixedPaddingIsRejectedBeforeInference()
        {
            var noPadding = TokenizerJson.Replace("\"Fixed\": 128", "\"Fixed\": 3");
            using var embedder = new TextEmbedder(SmallModel(), noPadding, BackendType.CPU);
            Assert.Throws<InvalidOperationException>(() => embedder.EmbedRaw("hello"));
        }
    }
}
