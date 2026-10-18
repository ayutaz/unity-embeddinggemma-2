using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using EmbeddingGemma;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using Unity.InferenceEngine;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class PlayerMixedSearchProtocolTests
    {
        static readonly string[] TextIds = { "doc-cat-ja", "doc-cat-en", "doc-space-ja", "doc-space-en", "doc-recipe-a", "doc-recipe-b" };
        static readonly string[] ImageIds = { "image-square-0", "image-square-1", "image-landscape-0", "image-landscape-1", "image-portrait-0", "image-portrait-1" };
        static readonly string[] AudioIds = { "audio-silence", "audio-impulse", "audio-sine", "audio-nyquist", "audio-noise", "audio-padded_noise" };
        static readonly string[] QueryIds = { "query-cat-ja", "query-space-en", "query-recipe-ja", "query-cat-en" };
        static float[] Basis(int index) { var values = new float[768]; values[index] = 1; return values; }
        sealed class Text : ITextEmbedder
        {
            public BackendType Backend { get; set; } = BackendType.GPUCompute;
            public readonly Dictionary<string, float[]> Documents = new Dictionary<string, float[]>();
            public readonly Dictionary<string, float[]> Queries = new Dictionary<string, float[]>();
            public int DocumentCalls, QueryCalls, Disposals;
            public float[] EmbedDocument(string text, string title = null) { DocumentCalls++; return Documents[text].ToArray(); }
            public float[] EmbedQuery(string text) { QueryCalls++; return Queries[text].ToArray(); }
            public void Dispose() { Disposals++; }
        }
        static (JArray docs, JArray queries, JObject media, Text text) Fixture()
        {
            var docs = new JArray(); var queries = new JArray(); var media = new JObject(); var text = new Text();
            var groups = new[] { ("text", TextIds), ("image", ImageIds), ("audio", AudioIds) };
            foreach (var (kind, ids) in groups)
                for (var i = 0; i < ids.Length; i++)
                {
                    var id = ids[i]; var expected = Basis(kind == "text" && i == 5 ? 4 : i);
                    docs.Add(new JObject { ["id"] = id, ["kind"] = kind, ["text"] = "fixed " + id, ["title"] = id, ["expected"] = new JArray(expected) });
                    if (kind == "text") text.Documents["fixed " + id] = expected; else media[id] = new JArray(expected);
                }
            for (var i = 0; i < QueryIds.Length; i++)
            {
                var id = QueryIds[i]; var expected = Basis(i);
                queries.Add(new JObject { ["id"] = id, ["text"] = "fixed " + id, ["expected"] = new JArray(expected) });
                text.Queries["fixed " + id] = expected;
            }
            return (docs, queries, media, text);
        }
        static JObject Run(JArray docs, JArray queries, JObject media, Text text, string precision = "fp32")
        {
            var type = AppDomain.CurrentDomain.GetAssemblies().Select(a => a.GetType("EmbeddingGemma.Validation.PlayerMixedSearchProtocol")).FirstOrDefault(t => t != null);
            Assert.That(type, Is.Not.Null, "The Player must validate one complete mixed text/image/audio index against the original vectors.");
            return (JObject)type.GetMethod("Run").Invoke(null, new object[] { docs, queries, media, precision, text });
        }
        static void Reject(Action run)
        { Assert.That(Assert.Throws<TargetInvocationException>(() => run()).InnerException, Is.InstanceOf<ArgumentException>().Or.InstanceOf<InvalidOperationException>()); }

        [TestCase("fp32")] [TestCase("float16")]
        public void CompleteMixedIndexChecksEveryDocumentAndQueryWithoutCertifyingInjectedGpu(string precision)
        {
            var f = Fixture(); var result = Run(f.docs, f.queries, f.media, f.text, precision);
            Assert.That((bool)result["success"], Is.True); Assert.That((bool)result["gpu_verified"], Is.False);
            Assert.That((int)result["document_count"], Is.EqualTo(18)); Assert.That((int)result["rank_and_score_pairs"], Is.EqualTo(72));
            Assert.That(f.text.DocumentCalls, Is.EqualTo(6)); Assert.That(f.text.QueryCalls, Is.EqualTo(4)); Assert.That(f.text.Disposals, Is.Zero);
            var ties = (JArray)result["queries"][0]["hits"];
            Assert.That(ties.Select(h => (string)h["id"]).ToArray(), Is.EqualTo(ties.OrderByDescending(h => (double)h["score"]).ThenBy(h => (string)h["id"], StringComparer.Ordinal).Select(h => (string)h["id"]).ToArray()));
        }
        [TestCase("text")] [TestCase("image")] [TestCase("audio")]
        public void MissingModalityCannotBecomeASmallerPassingSuite(string kind)
        {
            var f = Fixture(); foreach (var row in f.docs.Where(d => (string)d["kind"] == kind).ToArray()) row.Remove();
            Reject(() => Run(f.docs, f.queries, f.media, f.text)); Assert.That(f.text.DocumentCalls + f.text.QueryCalls, Is.Zero);
        }
        [TestCase("duplicate_document")] [TestCase("wrong_kind")] [TestCase("missing_media")]
        [TestCase("extra_media")] [TestCase("missing_query")] [TestCase("duplicate_query")]
        [TestCase("blank_query")] [TestCase("unknown_precision")] [TestCase("cpu")]
        public void MalformedOrCpuInputIsRejectedBeforeTextExecution(string problem)
        {
            var f = Fixture(); var precision = "fp32";
            if (problem == "duplicate_document") f.docs[1]["id"] = f.docs[0]["id"].DeepClone();
            if (problem == "wrong_kind") f.docs[0]["kind"] = "image";
            if (problem == "missing_media") f.media.Remove(ImageIds[0]);
            if (problem == "extra_media") f.media["unexpected"] = new JArray(Basis(0));
            if (problem == "missing_query") f.queries[0].Remove();
            if (problem == "duplicate_query") f.queries[1]["id"] = f.queries[0]["id"].DeepClone();
            if (problem == "blank_query") f.queries[0]["text"] = " ";
            if (problem == "unknown_precision") precision = "auto";
            if (problem == "cpu") f.text.Backend = BackendType.CPU;
            Reject(() => Run(f.docs, f.queries, f.media, f.text, precision)); Assert.That(f.text.DocumentCalls + f.text.QueryCalls, Is.Zero);
        }
        [TestCase("zero")] [TestCase("opposite")] [TestCase("nonfinite")] [TestCase("dimension")] [TestCase("norm")]
        public void CorruptMediaVectorsCannotEnterTheMixedIndex(string corruption)
        {
            var f = Fixture(); var v = Basis(0);
            if (corruption == "zero") v[0] = 0;
            if (corruption == "opposite") v[0] = -1;
            if (corruption == "nonfinite") v[0] = float.NaN;
            if (corruption == "dimension") v = new float[767];
            if (corruption == "norm") v[0] = 2;
            f.media[ImageIds[0]] = new JArray(v); Reject(() => Run(f.docs, f.queries, f.media, f.text));
        }
        [Test] public void ValidCosineDoesNotReplaceOriginalScoreAndRankGates()
        {
            var f = Fixture(); var v = Basis(0); v[1] = .01f; var norm = Math.Sqrt(v.Sum(x => (double)x * x));
            f.media[ImageIds[0]] = new JArray(v.Select(x => (float)(x / norm)));
            Reject(() => Run(f.docs, f.queries, f.media, f.text));
        }
        [Test] public void WrongTextDocumentOrQueryCannotBeHiddenByCorrectMedia()
        {
            var f = Fixture(); f.text.Documents["fixed " + TextIds[0]] = Basis(1);
            Reject(() => Run(f.docs, f.queries, f.media, f.text));
            f = Fixture(); f.text.Queries["fixed " + QueryIds[0]] = Basis(1);
            Reject(() => Run(f.docs, f.queries, f.media, f.text));
        }
    }
}
