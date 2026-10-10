using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text.RegularExpressions;
using Newtonsoft.Json.Linq;
using Unity.InferenceEngine;

namespace EmbeddingGemma.Validation
{
    [Serializable] public sealed class PlayerRow
    {
        public string id, role, text, title, formatted_text;
        public int[] input_ids, attention_mask;
        public float[] embedding;
        public PlayerRanking[] ranking;
    }
    [Serializable] public sealed class PlayerRanking { public string document_id; public double score; }
    public sealed class PlayerReferenceSet { public PlayerRow[] Cases, Documents, Queries; }
    [Serializable] public sealed class PlayerCaseResult { public string id; public double cosine; }
    [Serializable] public sealed class PlayerQueryResult { public string id; public double cosine; public PlayerRanking[] ranking; }
    [Serializable] public sealed class PlayerCondition
    {
        public bool success, workerReleased;
        public string precision, requestedBackend, actualBackend, error;
        public double minimumCosine = 1, firstInferenceMilliseconds, releaseMilliseconds;
        public int caseCount, documentCount, queryCount;
        public double[] warmInferenceMilliseconds;
        public PlayerCaseResult[] cases;
        public PlayerQueryResult[] queries;
    }
    public interface IPlayerEmbedder : ITextEmbedder { float[] EmbedRaw(string text); }

    public static class PlayerValidation
    {
        public const string ModelRevision = "914f7f89142e33e77833254d9c9b90c3cef7303b";
        public static PlayerReferenceSet ParseReferences(JObject m1, JObject search, string modelSource, string searchSource)
        {
            var first = CheckMetadata(m1, modelSource);
            var second = CheckMetadata(search, searchSource);
            Require(JToken.DeepEquals(first, second), "M1 and search model settings or dependencies differ.");
            var reference = new PlayerReferenceSet
            {
                Cases = ReadRows(m1, "cases", 15),
                Documents = ReadRows(search, "documents", 6),
                Queries = ReadRows(search, "queries", 4)
            };
            Require(reference.Documents.All(row => row.role == "document" && !string.IsNullOrWhiteSpace(row.text)), "Document roles/text differ.");
            Require(reference.Queries.All(row => row.role == "query" && !string.IsNullOrWhiteSpace(row.text)), "Query roles/text differ.");
            var documentIds = reference.Documents.Select(row => row.id).OrderBy(id => id, StringComparer.Ordinal).ToArray();
            foreach (var query in reference.Queries)
            {
                Require(query.ranking != null && query.ranking.Length == 6 && query.ranking.All(rank => rank != null &&
                    !double.IsNaN(rank.score) && !double.IsInfinity(rank.score)) &&
                    query.ranking.Select(rank => rank.document_id).OrderBy(id => id, StringComparer.Ordinal).SequenceEqual(documentIds),
                    "Every query requires the complete unique document ranking.");
            }
            return reference;
        }

        static JObject CheckMetadata(JObject data, string source)
        {
            Require(data != null && data["schema_version"]?.Type == JTokenType.Integer && (int)data["schema_version"] == 1, "Reference schema must be 1.");
            var metadata = data["metadata"] as JObject;
            Require(metadata != null && source != null && Regex.IsMatch(source, "\\A[a-f0-9]{40}\\z") &&
                (string)metadata["source_commit"] == source, "Reference source commit differs.");
            var expected = new JObject { ["model_id"] = "google/embeddinggemma-2", ["model_revision"] = ModelRevision,
                ["sequence_length"] = 128, ["batch_size"] = 1, ["dtype"] = "float32", ["embedding_dimension"] = 768,
                ["pooling"] = "masked_mean_including_prompt", ["normalization"] = "l2" };
            foreach (var field in expected.Properties()) Require(JToken.DeepEquals(metadata[field.Name], field.Value), "Pinned reference setting differs: " + field.Name);
            var versions = metadata["versions"] as JObject;
            Require(versions != null && new[] { "torch", "transformers", "sentence-transformers", "tokenizers" }
                .All(name => versions[name]?.Type == JTokenType.String && !string.IsNullOrWhiteSpace((string)versions[name])), "Reference dependency versions are missing.");
            var comparable = (JObject)metadata.DeepClone(); comparable.Remove("source_commit");
            return comparable;
        }

        static PlayerRow[] ReadRows(JObject data, string group, int count)
        {
            var rows = data[group] as JArray;
            Require(rows != null && rows.Count == count, "Reference requires " + count + " " + group + ".");
            foreach (var token in rows)
            {
                Require(token is JObject, "Reference row must be an object.");
                var ids = token["input_ids"] as JArray; var mask = token["attention_mask"] as JArray; var vector = token["embedding"] as JArray;
                Require(ids != null && ids.Count == 128 && ids.All(value => value.Type == JTokenType.Integer && (long)value >= 0 && (long)value <= int.MaxValue), "Expected 128 integer token IDs.");
                Require(mask != null && mask.Count == 128 && mask.All(value => value.Type == JTokenType.Integer && ((long)value == 0 || (long)value == 1)), "Expected 128 binary integer mask entries.");
                Require(vector != null && vector.Count == 768 && vector.All(value => value.Type == JTokenType.Float || value.Type == JTokenType.Integer), "Expected 768 numeric vector entries.");
                Require(token["id"]?.Type == JTokenType.String && Regex.IsMatch((string)token["id"], "\\A[a-z0-9][a-z0-9_-]*\\z") &&
                    token["text"]?.Type == JTokenType.String && token["formatted_text"]?.Type == JTokenType.String &&
                    token["role"]?.Type == JTokenType.String, "Reference ID, text and prompt must be strings.");
            }
            PlayerRow[] result;
            try { result = rows.ToObject<PlayerRow[]>(); }
            catch (Newtonsoft.Json.JsonException exception) { throw new ArgumentException("Invalid reference row.", exception); }
            Require(result.Select(row => row.id).Distinct(StringComparer.Ordinal).Count() == count, "Reference IDs must be unique.");
            foreach (var row in result)
            {
                var role = ParseRole(row.role);
                Require(row.title == null || role == TextRole.Document, "Only documents can have titles.");
                Require(row.formatted_text == TextPrompts.Format(row.text, role, row.title), "Formatted prompt differs: " + row.id);
                CheckUnitVector(row.embedding);
            }
            return result;
        }

        static TextRole ParseRole(string role)
        {
            if (role == "query") return TextRole.Query;
            if (role == "document") return TextRole.Document;
            if (role == "raw") return TextRole.Raw;
            throw new ArgumentException("Unknown reference role: " + role);
        }

        public static void CheckTokens(PlayerRow row, int[] ids, int[] mask)
            => Require(row != null && ids != null && mask != null && ids.Length == 128 && mask.Length == 128 &&
                row.input_ids != null && row.attention_mask != null && ids.SequenceEqual(row.input_ids) && mask.SequenceEqual(row.attention_mask), "Token IDs or attention mask differ: " + row?.id);

        public static double CheckEmbedding(float[] actual, float[] expected, string precision)
        {
            Require(precision == "fp32" || precision == "float16", "Unknown weight precision.");
            CheckUnitVector(actual); CheckUnitVector(expected);
            var cosine = TextSearchIndex.Cosine(actual, expected);
            Require(cosine >= (precision == "fp32" ? 0.999 : 0.99), "Embedding cosine below the precision threshold: " + cosine);
            return cosine;
        }

        static void CheckUnitVector(float[] vector)
            => Require(vector != null && vector.Length == 768 && vector.All(value => !float.IsNaN(value) && !float.IsInfinity(value)) &&
                Math.Abs(vector.Sum(value => (double)value * value) - 1) <= 0.002, "Expected a finite unit-length 768-dimensional embedding.");

        static void Require(bool condition, string message) { if (!condition) throw new ArgumentException(message); }

        public static int CheckAllTokens(PlayerReferenceSet reference, Func<string, (int[] ids, int[] mask)> encode)
        {
            Require(reference?.Cases?.Length == 15 && reference.Documents?.Length == 6 && reference.Queries?.Length == 4, "All fixed token rows are required.");
            var rows = reference.Cases.Concat(reference.Documents).Concat(reference.Queries).ToArray();
            foreach (var row in rows)
            {
                var actual = encode(row.formatted_text);
                CheckTokens(row, actual.ids, actual.mask);
            }
            return rows.Length;
        }

        public static bool AllConditionsPassed(PlayerCondition[] conditions)
        {
            if (conditions == null || conditions.Length != 4 || conditions.Any(result => result == null)) return false;
            var pairs = conditions.Select(result => result.precision + ":" + result.requestedBackend).OrderBy(pair => pair, StringComparer.Ordinal);
            if (!pairs.SequenceEqual(new[] { "float16:CPU", "float16:GPUCompute", "fp32:CPU", "fp32:GPUCompute" })) return false;
            return conditions.All(result => result.success && result.workerReleased && string.IsNullOrEmpty(result.error) &&
                result.actualBackend == result.requestedBackend && result.caseCount == 15 && result.documentCount == 6 && result.queryCount == 4 &&
                result.cases?.Length == 15 && result.queries?.Length == 4 && result.queries.All(query => query?.ranking?.Length == 6) &&
                result.warmInferenceMilliseconds?.Length >= 1 && result.warmInferenceMilliseconds.All(value => value >= 0 && !double.IsNaN(value) && !double.IsInfinity(value)) &&
                result.minimumCosine >= (result.precision == "fp32" ? 0.999 : 0.99) && result.minimumCosine <= 1);
        }

        sealed class RecordingProvider : ITextEmbedder
        {
            readonly IPlayerEmbedder inner;
            public readonly List<float[]> Documents = new();
            public float[] LastQuery;
            public BackendType Backend => inner.Backend;
            public RecordingProvider(IPlayerEmbedder inner) { this.inner = inner; }
            public float[] EmbedDocument(string text, string title = null) { var value = inner.EmbedDocument(text, title); Documents.Add(value); return value; }
            public float[] EmbedQuery(string text) { LastQuery = inner.EmbedQuery(text); return LastQuery; }
            public void Dispose() => inner.Dispose();
        }

        public static PlayerCondition RunCondition(PlayerReferenceSet reference, string precision, BackendType requested, Func<IPlayerEmbedder> create, int warmRepetitions = 3)
        {
            var result = new PlayerCondition { precision = precision, requestedBackend = requested.ToString() };
            IPlayerEmbedder provider = null;
            var completed = false;
            var cases = new List<PlayerCaseResult>(); var queries = new List<PlayerQueryResult>();
            try
            {
                Require(reference?.Cases?.Length == 15 && reference.Documents?.Length == 6 && reference.Queries?.Length == 4, "All fixed cases are required.");
                Require(precision == "fp32" || precision == "float16", "Unknown weight precision.");
                Require(requested == BackendType.CPU || requested == BackendType.GPUCompute, "CPU or GPUCompute is required.");
                Require(warmRepetitions >= 1 && warmRepetitions <= 100, "Use 1 to 100 warm inference repetitions.");
                provider = create(); Require(provider != null, "Provider was not created.");
                result.actualBackend = provider.Backend.ToString();
                Require(provider.Backend == requested, "Actual backend differs from requested backend.");
                foreach (var row in reference.Cases)
                {
                    var timer = Stopwatch.StartNew();
                    var role = ParseRole(row.role);
                    var vector = role == TextRole.Query ? provider.EmbedQuery(row.text) : role == TextRole.Document ? provider.EmbedDocument(row.text, row.title) : provider.EmbedRaw(row.text);
                    timer.Stop(); if (cases.Count == 0) result.firstInferenceMilliseconds = timer.Elapsed.TotalMilliseconds;
                    var cosine = CheckEmbedding(vector, row.embedding, precision);
                    result.minimumCosine = Math.Min(result.minimumCosine, cosine);
                    cases.Add(new PlayerCaseResult { id = row.id, cosine = cosine }); result.caseCount = cases.Count;
                }
                var recording = new RecordingProvider(provider);
                var index = new TextSearchIndex(reference.Documents.Select(row => new SearchDocument(row.id, row.text, row.title)), recording);
                for (var i = 0; i < reference.Documents.Length; i++)
                    result.minimumCosine = Math.Min(result.minimumCosine, CheckEmbedding(recording.Documents[i], reference.Documents[i].embedding, precision));
                result.documentCount = index.DocumentCount;
                foreach (var query in reference.Queries)
                {
                    var hits = index.Search(query.text);
                    var cosine = CheckEmbedding(recording.LastQuery, query.embedding, precision);
                    result.minimumCosine = Math.Min(result.minimumCosine, cosine);
                    Require(hits.Select(hit => hit.Document.Id).SequenceEqual(query.ranking.Select(rank => rank.document_id)), "Full ranking differs: " + query.id);
                    for (var i = 0; i < hits.Length; i++) Require(Math.Abs(hits[i].Score - query.ranking[i].score) <= (precision == "fp32" ? 0.002 : 0.02), "Ranking score differs: " + query.id);
                    queries.Add(new PlayerQueryResult { id = query.id, cosine = cosine,
                        ranking = hits.Select(hit => new PlayerRanking { document_id = hit.Document.Id, score = hit.Score }).ToArray() }); result.queryCount = queries.Count;
                }
                var warmQuery = reference.Queries[0];
                _ = CheckEmbedding(provider.EmbedQuery(warmQuery.text), warmQuery.embedding, precision);
                result.warmInferenceMilliseconds = new double[warmRepetitions];
                for (var i = 0; i < warmRepetitions; i++)
                {
                    var timer = Stopwatch.StartNew(); var vector = provider.EmbedQuery(warmQuery.text); timer.Stop();
                    result.warmInferenceMilliseconds[i] = timer.Elapsed.TotalMilliseconds;
                    result.minimumCosine = Math.Min(result.minimumCosine, CheckEmbedding(vector, warmQuery.embedding, precision));
                }
                completed = true;
            }
            catch (Exception exception) { result.error = exception.GetType().Name + ": " + exception.Message; }
            finally
            {
                result.cases = cases.ToArray(); result.queries = queries.ToArray();
                if (provider != null)
                {
                    var timer = Stopwatch.StartNew();
                    try { provider.Dispose(); result.workerReleased = true; }
                    catch (Exception exception) { result.error = (result.error ?? "") + " Release: " + exception.Message; }
                    finally { timer.Stop(); result.releaseMilliseconds = timer.Elapsed.TotalMilliseconds; }
                }
            }
            result.success = completed && result.workerReleased;
            return result;
        }
    }
}
