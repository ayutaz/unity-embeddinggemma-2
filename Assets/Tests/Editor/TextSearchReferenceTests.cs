using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEngine;

namespace EmbeddingGemma.Tests
{
    public sealed class TextSearchReferenceTests
    {
        [Serializable] sealed class Reference { public Metadata metadata; public Row[] documents, queries; }
        [Serializable] sealed class Metadata { public string model_revision, source_commit; public int sequence_length, embedding_dimension; }
        [Serializable] sealed class Row { public string id, text, title; public float[] embedding; public Ranking[] ranking; }
        [Serializable] sealed class Ranking { public string document_id; public double score; }
        [Serializable] sealed class Prepared { public bool success; public string sourceCommit, searchSourceCommit; }
        [Serializable] sealed class QueryResult { public string id; public Ranking[] ranking; }
        [Serializable] sealed class Condition
        {
            public bool success; public string precision, backend, error;
            public double minimumCosine = 1;
            public int documentCount, queryCount;
            public List<QueryResult> queries = new();
        }
        [Serializable] sealed class Report
        {
            public bool success;
            public string startedUtc, completedUtc, unity, os, gpu, graphicsApi, modelRevision, referenceSource, modelSource;
            public List<Condition> conditions = new();
        }
        sealed class RecordingEmbedder : ITextEmbedder
        {
            readonly TextEmbedder inner;
            public readonly List<float[]> Documents = new();
            public float[] LastQuery;
            public BackendType Backend => inner.Backend;
            public RecordingEmbedder(TextEmbedder inner) { this.inner = inner; }
            public float[] EmbedDocument(string text, string title = null) { var result = inner.EmbedDocument(text, title); Documents.Add(result); return result; }
            public float[] EmbedQuery(string text) { LastQuery = inner.EmbedQuery(text); return LastQuery; }
            public void Dispose() => inner.Dispose();
        }
        Report report;
        string output;

        [OneTimeSetUp]
        public void Start()
        {
            output = Path.GetFullPath(Path.Combine(Application.dataPath, "../artifacts/m2-search/results.json"));
            Directory.CreateDirectory(Path.GetDirectoryName(output));
            report = new Report { startedUtc = DateTime.UtcNow.ToString("O"), unity = Application.unityVersion,
                os = SystemInfo.operatingSystem, gpu = SystemInfo.graphicsDeviceName, graphicsApi = SystemInfo.graphicsDeviceType.ToString() };
            Save();
        }
        void Save() => File.WriteAllText(output, JsonUtility.ToJson(report, true));

        [TestCase("fp32", BackendType.CPU)]
        [TestCase("fp32", BackendType.GPUCompute)]
        [TestCase("float16", BackendType.CPU)]
        [TestCase("float16", BackendType.GPUCompute)]
        [Timeout(1200000)]
        public void RealModelMatchesEmbeddingsAndEveryFullRanking(string precision, BackendType backend)
        {
            var condition = new Condition { precision = precision, backend = backend.ToString() };
            report.conditions.Add(condition);
            Save();
            try
            {
                var referencePath = Path.GetFullPath(Path.Combine(Application.dataPath, "../artifacts/m1/search-reference.json"));
                Assert.That(File.Exists(referencePath), Is.True, "Stage the successful CI artifact with --search first.");
                var reference = JsonUtility.FromJson<Reference>(File.ReadAllText(referencePath));
                Assert.That(reference.metadata.model_revision, Is.EqualTo("914f7f89142e33e77833254d9c9b90c3cef7303b"));
                Assert.That(reference.metadata.source_commit, Does.Match("^[a-f0-9]{40}$"));
                Assert.That(reference.metadata.sequence_length, Is.EqualTo(128));
                Assert.That(reference.metadata.embedding_dimension, Is.EqualTo(768));
                Assert.That(reference.documents, Has.Length.EqualTo(6));
                Assert.That(reference.queries, Has.Length.EqualTo(4));
                report.modelRevision = reference.metadata.model_revision; report.referenceSource = reference.metadata.source_commit;
                if (backend == BackendType.GPUCompute) Assert.That(SystemInfo.supportsComputeShaders, Is.True, "GPU must execute; no skips or CPU fallback.");
                var directory = Path.Combine(Application.streamingAssetsPath, "EmbeddingGemmaTextSearch");
                var prepared = JsonUtility.FromJson<Prepared>(File.ReadAllText(Path.Combine(directory, "preparation.json")));
                Assert.That(prepared.success, Is.True);
                Assert.That(prepared.searchSourceCommit, Is.EqualTo(reference.metadata.source_commit));
                report.modelSource = prepared.sourceCommit;
                var modelPath = Path.Combine(directory, "model-" + precision + ".sentis");
                using var provider = new RecordingEmbedder(new TextEmbedder(TextModelFile.Load(modelPath), File.ReadAllText(Path.Combine(directory, "tokenizer.json")), backend));
                Assert.That(provider.Backend, Is.EqualTo(backend));
                var index = new TextSearchIndex(reference.documents.Select(row => new SearchDocument(row.id, row.text, row.title)), provider);
                var threshold = precision == "fp32" ? 0.999 : 0.99;
                for (var i = 0; i < reference.documents.Length; i++)
                {
                    var cosine = TextSearchIndex.Cosine(provider.Documents[i], reference.documents[i].embedding);
                    Assert.That(cosine, Is.GreaterThanOrEqualTo(threshold), reference.documents[i].id);
                    condition.minimumCosine = Math.Min(condition.minimumCosine, cosine);
                }
                foreach (var query in reference.queries)
                {
                    var hits = index.Search(query.text);
                    var cosine = TextSearchIndex.Cosine(provider.LastQuery, query.embedding);
                    Assert.That(cosine, Is.GreaterThanOrEqualTo(threshold), query.id);
                    condition.minimumCosine = Math.Min(condition.minimumCosine, cosine);
                    Assert.That(hits.Select(hit => hit.Document.Id), Is.EqualTo(query.ranking.Select(row => row.document_id)), query.id + " full ranking");
                    for (var i = 0; i < hits.Length; i++) Assert.That(hits[i].Score, Is.EqualTo(query.ranking[i].score).Within(precision == "fp32" ? 0.002 : 0.02), query.id + " score");
                    condition.queries.Add(new QueryResult { id = query.id, ranking = hits.Select(hit => new Ranking { document_id = hit.Document.Id, score = hit.Score }).ToArray() });
                }
                Assert.That(provider.Documents, Has.Count.EqualTo(6), "Documents are embedded once, not on every search.");
                condition.documentCount = 6; condition.queryCount = 4; condition.success = true;
            }
            catch (Exception exception) { condition.error = exception.Message; throw; }
            finally { Save(); }
        }

        [OneTimeTearDown]
        public void Finish()
        {
            report.success = report.conditions.Count == 4 && report.conditions.All(condition => condition.success);
            report.completedUtc = DateTime.UtcNow.ToString("O"); Save();
        }
    }
}
