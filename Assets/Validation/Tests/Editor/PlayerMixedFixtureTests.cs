using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using EmbeddingGemma;
using Newtonsoft.Json.Linq;
using NUnit.Framework;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class PlayerMixedFixtureTests
    {
        static float[] Basis(int i) { var values = new float[768]; values[i] = 1; return values; }
        static Type Required()
        {
            var type = AppDomain.CurrentDomain.GetAssemblies().Select(a => a.GetType("EmbeddingGemma.Validation.PlayerMixedSearchFixture")).FirstOrDefault(t => t != null);
            Assert.That(type, Is.Not.Null, "Mixed Player fixtures must retain fixed text and all media roles."); return type;
        }
        static JObject Reference()
        {
            var docs = new JArray(); var queries = new JArray();
            var ids = new[] { "doc-cat-ja", "doc-cat-en", "doc-space-ja", "doc-space-en", "doc-recipe-a", "doc-recipe-b" };
            for (var i = 0; i < ids.Length; i++) docs.Add(new JObject { ["id"] = ids[i], ["text"] = "fixed " + ids[i], ["title"] = "title", ["role"] = "document", ["embedding"] = new JArray(Basis(i)) });
            var qids = new[] { "query-cat-ja", "query-space-en", "query-recipe-ja", "query-cat-en" };
            for (var i = 0; i < qids.Length; i++) queries.Add(new JObject { ["id"] = qids[i], ["text"] = "fixed " + qids[i], ["role"] = "query", ["embedding"] = new JArray(Basis(i)) });
            return new JObject { ["documents"] = docs, ["queries"] = queries };
        }
        static JObject Parse(JObject reference) => (JObject)Required().GetMethod("ParseTextReference").Invoke(null, new object[] { reference });
        static void Reject(Action action) => Assert.That(Assert.Throws<TargetInvocationException>(() => action()).InnerException,
            Is.InstanceOf<ArgumentException>().Or.InstanceOf<InvalidOperationException>().Or.InstanceOf<InvalidDataException>());
        static (List<PlayerMediaCase> cases, JObject report) Module(string module)
        {
            string[] ids;
            if (module == "audio-1119") ids = new[] { "silence", "impulse", "sine", "nyquist", "noise" };
            else if (module == "audio-839") ids = new[] { "padded_noise" };
            else { var shape = module.Substring(6); ids = new[] { shape + "-0", shape + "-1" }; }
            var cases = new List<PlayerMediaCase>(); var rows = new JArray();
            foreach (var role in new[] { TextRole.Document, TextRole.Query, TextRole.Raw })
                for (var i = 0; i < ids.Length; i++)
                {
                    var expected = Basis((role == TextRole.Document ? 0 : role == TextRole.Query ? 20 : 40) + i);
                    cases.Add(new PlayerMediaCase { Id = ids[i], Role = role, Expected = expected });
                    rows.Add(new JObject { ["id"] = ids[i], ["role"] = role.ToString().ToLowerInvariant(), ["embedding"] = new JArray(expected) });
                }
            return (cases, new JObject { ["success"] = true, ["session_released"] = true, ["gpu_verified"] = false,
                ["validation_mode"] = "injected_contract", ["roles"] = new JArray("document", "query", "raw"), ["cases"] = rows });
        }
        static void Append(JArray docs, JObject vectors, string module, List<PlayerMediaCase> cases, JObject report)
            => Required().GetMethod("AppendMedia").Invoke(null, new object[] { docs, vectors, module, cases, report, "fp32" });
        [Test] public void ParsingSyntheticReferenceDoesNotClaimOriginalByteVerification()
        {
            var result = Parse(Reference());
            Assert.That(((JArray)result["documents"]).Count, Is.EqualTo(6)); Assert.That(((JArray)result["queries"]).Count, Is.EqualTo(4));
            Assert.That((bool)result["original_bytes_verified"], Is.False);
            Assert.That(result["documents"].All(r => (string)r["kind"] == "text" && ((JArray)r["expected"]).Count == 768), Is.True);
        }
        [TestCase("missing_document")] [TestCase("duplicate_document")] [TestCase("missing_query")]
        [TestCase("wrong_document_role")] [TestCase("wrong_query_role")] [TestCase("blank_text")] [TestCase("bad_vector")]
        public void PartialOrChangedTextReferenceIsRejected(string problem)
        {
            var r = Reference();
            if (problem == "missing_document") r["documents"][0].Remove();
            if (problem == "duplicate_document") r["documents"][1]["id"] = r["documents"][0]["id"].DeepClone();
            if (problem == "missing_query") r["queries"][0].Remove();
            if (problem == "wrong_document_role") r["documents"][0]["role"] = "query";
            if (problem == "wrong_query_role") r["queries"][0]["role"] = "document";
            if (problem == "blank_text") r["documents"][0]["text"] = " ";
            if (problem == "bad_vector") r["queries"][0]["embedding"] = new JArray(new float[768]);
            Reject(() => Parse(r));
        }
        [Test] public void WrongOriginalFileBytesFailBeforeAnyModelLoad()
        {
            var type = Required(); var root = Path.Combine(Path.GetTempPath(), "embeddinggemma-mixed-fixture-" + Guid.NewGuid().ToString("N"));
            Assert.That(Directory.Exists(root), Is.False); Directory.CreateDirectory(root); var path = Path.Combine(root, "search-reference.json");
            try { File.WriteAllText(path, "{}"); Reject(() => type.GetMethod("LoadText").Invoke(null, new object[] { root })); }
            finally { File.Delete(path); Directory.Delete(root); }
        }
        [Test] public void AllThirtySixMediaRoleCapturesProduceExactlyTwelveMixedMediaDocuments()
        {
            var text = Parse(Reference()); var docs = (JArray)text["documents"]; var vectors = new JObject(); var count = 0;
            foreach (var module in new[] { "image-square", "image-landscape", "image-portrait", "audio-1119", "audio-839" })
            { var data = Module(module); count += data.cases.Count; Append(docs, vectors, module, data.cases, data.report); }
            Assert.That(count, Is.EqualTo(36)); Assert.That(docs.Count, Is.EqualTo(18)); Assert.That(vectors.Count, Is.EqualTo(12));
            Assert.That(docs.Count(r => (string)r["kind"] == "text"), Is.EqualTo(6));
            Assert.That(docs.Count(r => (string)r["kind"] == "image"), Is.EqualTo(6)); Assert.That(docs.Count(r => (string)r["kind"] == "audio"), Is.EqualTo(6));
        }
        [TestCase("missing_expected_role")] [TestCase("missing_capture")] [TestCase("corrupt_query_capture")]
        [TestCase("duplicate_capture")] [TestCase("failed_module")] [TestCase("unreleased_module")]
        public void MediaFailuresCannotBeHiddenByKeepingOnlyCorrectDocumentVectors(string problem)
        {
            var docs = (JArray)Parse(Reference())["documents"]; var vectors = new JObject(); var data = Module("image-square");
            if (problem == "missing_expected_role") data.cases.RemoveAt(2);
            if (problem == "missing_capture") data.report["cases"][2].Remove();
            if (problem == "corrupt_query_capture") data.report["cases"][2]["embedding"] = new JArray(Basis(0));
            if (problem == "duplicate_capture") data.report["cases"][2] = data.report["cases"][0].DeepClone();
            if (problem == "failed_module") data.report["success"] = false;
            if (problem == "unreleased_module") data.report["session_released"] = false;
            Reject(() => Append(docs, vectors, "image-square", data.cases, data.report));
            Assert.That(docs.Count, Is.EqualTo(6)); Assert.That(vectors.Count, Is.Zero);
        }
    }
}
