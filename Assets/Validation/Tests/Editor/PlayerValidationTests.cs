using System;
using System.Linq;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using Unity.InferenceEngine;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class PlayerValidationTests
    {
        public static float[] Vector() { var result = new float[768]; result[0] = 1; return result; }
        public static PlayerReferenceSet Reference()
        {
            PlayerRow Row(string id, string role) => new PlayerRow { id = id, role = role, text = "猫", formatted_text = TextPrompts.Format("猫", Role(role)),
                input_ids = new int[128], attention_mask = new int[128], embedding = Vector() };
            var reference = new PlayerReferenceSet
            {
                Cases = Enumerable.Range(0, 15).Select(i => Row("case-" + i, "raw")).ToArray(),
                Documents = Enumerable.Range(0, 6).Select(i => Row("doc-" + i, "document")).ToArray(),
                Queries = Enumerable.Range(0, 4).Select(i => Row("query-" + i, "query")).ToArray()
            };
            foreach (var row in reference.Queries) row.ranking = reference.Documents.Select(doc => new PlayerRanking { document_id = doc.id, score = 1 }).ToArray();
            return reference;
        }
        static TextRole Role(string role) => role == "query" ? TextRole.Query : role == "document" ? TextRole.Document : TextRole.Raw;
        public static JObject ReferenceJson(PlayerReferenceSet reference, bool search)
        {
            var result = new JObject { ["schema_version"] = 1, ["metadata"] = new JObject {
                ["model_id"] = "google/embeddinggemma-2", ["model_revision"] = PlayerValidation.ModelRevision,
                ["source_commit"] = new string(search ? 'b' : 'a', 40), ["sequence_length"] = 128, ["batch_size"] = 1,
                ["dtype"] = "float32", ["embedding_dimension"] = 768, ["pooling"] = "masked_mean_including_prompt", ["normalization"] = "l2",
                ["versions"] = new JObject { ["torch"] = "pinned", ["transformers"] = "pinned", ["sentence-transformers"] = "pinned", ["tokenizers"] = "pinned" }
            } };
            if (search) { result["documents"] = JArray.FromObject(reference.Documents); result["queries"] = JArray.FromObject(reference.Queries); }
            else result["cases"] = JArray.FromObject(reference.Cases);
            return result;
        }
        sealed class Provider : IPlayerEmbedder
        {
            public BackendType Backend { get; set; } = BackendType.CPU;
            public int Disposals, Calls;
            public bool ThrowOnInference, ThrowOnDispose;
            public float[] Value = Vector();
            float[] Embed() { Calls++; if (ThrowOnInference) throw new InvalidOperationException("inference failed"); return (float[])Value.Clone(); }
            public float[] EmbedRaw(string text) => Embed();
            public float[] EmbedQuery(string text) => Embed();
            public float[] EmbedDocument(string text, string title = null) => Embed();
            public void Dispose() { Disposals++; if (ThrowOnDispose) throw new InvalidOperationException("release failed"); }
        }

        [Test] public void ParsesBothCompletePinnedReferences()
        {
            var reference = Reference();
            var actual = PlayerValidation.ParseReferences(ReferenceJson(reference, false), ReferenceJson(reference, true), new string('a', 40), new string('b', 40));
            Assert.That(actual.Cases, Has.Length.EqualTo(15)); Assert.That(actual.Documents, Has.Length.EqualTo(6)); Assert.That(actual.Queries, Has.Length.EqualTo(4));
        }
        [TestCase("revision")] [TestCase("dependencies")] [TestCase("source")] [TestCase("prompt")] [TestCase("count")] [TestCase("ranking")] [TestCase("token_type")]
        public void InvalidReferenceContractsCannotBeExecuted(string violation)
        {
            var reference = Reference(); var m1 = ReferenceJson(reference, false); var search = ReferenceJson(reference, true);
            if (violation == "revision") search["metadata"]["model_revision"] = new string('c', 40);
            if (violation == "dependencies") search["metadata"]["versions"]["torch"] = "other";
            if (violation == "source") m1["metadata"]["source_commit"] = new string('b', 40);
            if (violation == "prompt") m1["cases"][0]["formatted_text"] = "other";
            if (violation == "count") ((JArray)m1["cases"]).RemoveAt(0);
            if (violation == "ranking") search["queries"][0]["ranking"][0]["document_id"] = "unknown";
            if (violation == "token_type") m1["cases"][0]["input_ids"][0] = "0";
            Assert.Throws<ArgumentException>(() => PlayerValidation.ParseReferences(m1, search, new string('a', 40), new string('b', 40)));
        }
        [Test] public void ComparesEveryTokenAndMaskWithoutApproximation()
        {
            var row = Reference().Cases[0]; PlayerValidation.CheckTokens(row, new int[128], new int[128]);
            var wrong = new int[128]; wrong[127] = 1;
            Assert.Throws<ArgumentException>(() => PlayerValidation.CheckTokens(row, wrong, new int[128]));
            Assert.Throws<ArgumentException>(() => PlayerValidation.CheckTokens(row, new int[128], wrong));
        }
        [TestCase("length")] [TestCase("nan")] [TestCase("norm")] [TestCase("precision")]
        public void InvalidVectorsCannotPassCosine(string violation)
        {
            var value = Vector(); if (violation == "length") value = new float[767];
            if (violation == "nan") value[7] = float.NaN;
            if (violation == "norm") value[0] = 2;
            Assert.Throws<ArgumentException>(() => PlayerValidation.CheckEmbedding(value, Vector(), violation == "precision" ? "float8" : "fp32"));
        }
        [Test] public void PrecisionThresholdsAreAppliedToFiniteUnitVectors()
        {
            var value = Vector(); value[0] = 0.995f; value[1] = (float)Math.Sqrt(1 - value[0] * value[0]);
            Assert.That(PlayerValidation.CheckEmbedding(value, Vector(), "float16"), Is.GreaterThanOrEqualTo(0.99));
            Assert.Throws<ArgumentException>(() => PlayerValidation.CheckEmbedding(value, Vector(), "fp32"));
        }
        [Test] public void CompleteConditionRecordsEveryCaseRankingTimingAndRelease()
        {
            var provider = new Provider(); var result = PlayerValidation.RunCondition(Reference(), "fp32", BackendType.CPU, () => provider);
            Assert.That(result.success, Is.True, result.error); Assert.That(result.caseCount, Is.EqualTo(15)); Assert.That(result.documentCount, Is.EqualTo(6)); Assert.That(result.queryCount, Is.EqualTo(4));
            Assert.That(result.requestedBackend, Is.EqualTo("CPU")); Assert.That(result.actualBackend, Is.EqualTo("CPU"));
            Assert.That(result.queries.All(query => query.ranking.Length == 6), Is.True); Assert.That(result.warmInferenceMilliseconds, Has.Length.EqualTo(3));
            Assert.That(result.workerReleased, Is.True); Assert.That(provider.Disposals, Is.EqualTo(1)); Assert.That(provider.Calls, Is.EqualTo(29));
        }
        [TestCase("backend")] [TestCase("ranking")] [TestCase("inference")] [TestCase("release")]
        public void EveryExecutionFailureIsReportedAndReleasesItsProvider(string failure)
        {
            var provider = new Provider { ThrowOnInference = failure == "inference", ThrowOnDispose = failure == "release" };
            var reference = Reference(); if (failure == "ranking") Array.Reverse(reference.Queries[0].ranking);
            var result = PlayerValidation.RunCondition(reference, "fp32", failure == "backend" ? BackendType.GPUCompute : BackendType.CPU, () => provider);
            Assert.That(result.success, Is.False); Assert.That(result.error, Is.Not.Empty); Assert.That(provider.Disposals, Is.EqualTo(1));
            Assert.That(result.workerReleased, Is.EqualTo(failure != "release"));
        }
        [Test] public void UnsupportedGpuIsAFailedConditionWithNoClaimOfActualBackend()
        {
            var result = PlayerValidation.RunCondition(Reference(), "fp32", BackendType.GPUCompute, () => throw new NotSupportedException("GPU unavailable"));
            Assert.That(result.success, Is.False); Assert.That(result.actualBackend, Is.Null); Assert.That(result.error, Does.Contain("GPU unavailable"));
        }
        [Test] public void AuditsAllTwentyFiveTokenRowsUsingTheirExactFormattedPrompts()
        {
            var calls = 0;
            var count = PlayerValidation.CheckAllTokens(Reference(), text => { calls++; Assert.That(text, Does.Contain("猫")); return (new int[128], new int[128]); });
            Assert.That(count, Is.EqualTo(25)); Assert.That(calls, Is.EqualTo(25));
        }
        [Test] public void ATokenMismatchInTheLastSearchQueryCannotBeIgnored()
        {
            var reference = Reference(); reference.Queries[3].attention_mask[127] = 1;
            Assert.Throws<ArgumentException>(() => PlayerValidation.CheckAllTokens(reference, _ => (new int[128], new int[128])));
        }
        PlayerCondition[] FourConditions() => new[] { "fp32", "float16" }.SelectMany(precision => new[] { BackendType.CPU, BackendType.GPUCompute }
            .Select(backend => PlayerValidation.RunCondition(Reference(), precision, backend, () => new Provider { Backend = backend }))).ToArray();
        [Test] public void OnlyFourCompleteDistinctConditionsCanPassTheCompletionAudit()
            => Assert.That(PlayerValidation.AllConditionsPassed(FourConditions()), Is.True);
        [TestCase("missing")] [TestCase("duplicate")] [TestCase("backend")] [TestCase("release")] [TestCase("cases")] [TestCase("error")]
        public void PartialOrInconsistentConditionsCannotPassTheCompletionAudit(string failure)
        {
            var conditions = FourConditions();
            if (failure == "missing") conditions = conditions.Take(3).ToArray();
            if (failure == "duplicate") conditions[3] = conditions[0];
            if (failure == "backend") conditions[1].actualBackend = "CPU";
            if (failure == "release") conditions[1].workerReleased = false;
            if (failure == "cases") conditions[1].caseCount = 14;
            if (failure == "error") conditions[1].error = "hidden error";
            Assert.That(PlayerValidation.AllConditionsPassed(conditions), Is.False);
        }
    }
}
