using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using EmbeddingGemma;
using Newtonsoft.Json.Linq;
using Unity.InferenceEngine;

namespace EmbeddingGemma.Validation
{
    /// <summary>Complete mixed-index numerical check. GPU execution needs separate live Worker evidence.</summary>
    public static class PlayerMixedSearchProtocol
    {
        static readonly string[] TextIds = { "doc-cat-ja", "doc-cat-en", "doc-space-ja", "doc-space-en", "doc-recipe-a", "doc-recipe-b" };
        static readonly string[] ImageIds = { "image-square-0", "image-square-1", "image-landscape-0", "image-landscape-1", "image-portrait-0", "image-portrait-1" };
        static readonly string[] AudioIds = { "audio-silence", "audio-impulse", "audio-sine", "audio-nyquist", "audio-noise", "audio-padded_noise" };
        static readonly string[] QueryIds = { "query-cat-ja", "query-space-en", "query-recipe-ja", "query-cat-en" };
        static bool Text(JToken token) => token?.Type == JTokenType.String && !string.IsNullOrWhiteSpace((string)token);
        static float[] Vector(JToken token)
        {
            if (!(token is JArray values) || values.Count != 768 || values.Any(v => v.Type != JTokenType.Float && v.Type != JTokenType.Integer))
                throw new ArgumentException("A complete numeric 768-dimensional vector is required.");
            return values.Values<float>().ToArray();
        }
        static void Cohort(JArray rows, string[] ids)
        {
            if (rows == null || rows.Count != ids.Length || rows.Any(r => !(r is JObject) || !Text(r["id"]) || !Text(r["text"])) ||
                !rows.Select(r => (string)r["id"]).OrderBy(v => v, StringComparer.Ordinal).SequenceEqual(ids.OrderBy(v => v, StringComparer.Ordinal)))
                throw new ArgumentException("Every unique fixed input is required; partial mixed cohorts are rejected.");
        }
        static double Cosine(float[] left, float[] right)
        {
            double dot = 0, nl = 0, nr = 0;
            for (var i = 0; i < 768; i++) { dot += (double)left[i] * right[i]; nl += (double)left[i] * left[i]; nr += (double)right[i] * right[i]; }
            return dot / Math.Sqrt(nl * nr);
        }
        sealed class QueryCapture : ITextEmbedder
        {
            readonly ITextEmbedder source;
            public QueryCapture(ITextEmbedder source) { this.source = source; }
            public float[] Last { get; private set; }
            public BackendType Backend => source.Backend;
            public float[] EmbedQuery(string text) { var values = source.EmbedQuery(text); Last = values?.ToArray(); return values; }
            public float[] EmbedDocument(string text, string title = null) => throw new NotSupportedException("Only query capture is used by this mixed index.");
            public void Dispose() { } // The caller retains ownership of its real text Worker.
        }

        public static JObject Run(JArray documents, JArray queries, JObject mediaVectors, string precision, ITextEmbedder text)
        {
            if (precision != "fp32" && precision != "float16" || text == null || text.Backend != BackendType.GPUCompute)
                throw new ArgumentException("Explicit precision and a GPUCompute text provider are required.");
            Cohort(documents, TextIds.Concat(ImageIds).Concat(AudioIds).ToArray()); Cohort(queries, QueryIds);
            if (mediaVectors == null || !mediaVectors.Properties().Select(p => p.Name).OrderBy(v => v, StringComparer.Ordinal)
                .SequenceEqual(ImageIds.Concat(AudioIds).OrderBy(v => v, StringComparer.Ordinal)))
                throw new ArgumentException("Exactly twelve original image/audio document vectors are required.");
            var expected = new Dictionary<string, float[]>(StringComparer.Ordinal);
            var actual = new Dictionary<string, float[]>(StringComparer.Ordinal);
            foreach (var row in documents)
            {
                var id = (string)row["id"]; var kind = TextIds.Contains(id) ? "text" : ImageIds.Contains(id) ? "image" : "audio";
                if ((string)row["kind"] != kind) throw new ArgumentException("Each fixed document must retain its modality.");
                var reference = Vector(row["expected"]); PlayerValidation.CheckEmbedding(reference, reference, precision); expected.Add(id, reference);
                if (kind != "text")
                {
                    var vector = Vector(mediaVectors[id]); PlayerValidation.CheckEmbedding(vector, reference, precision); actual.Add(id, vector);
                }
            }
            foreach (var query in queries)
            { var vector = Vector(query["expected"]); PlayerValidation.CheckEmbedding(vector, vector, precision); }
            var result = new JObject { ["schema_version"] = 1, ["success"] = false, ["gpu_verified"] = false,
                ["scope"] = "complete_mixed_index_numeric_check_requires_separate_live_gpu_and_process_evidence",
                ["document_count"] = 18, ["query_count"] = 4, ["rank_and_score_pairs"] = 72, ["precision"] = precision,
                ["original_score_absolute_error_gate"] = .001, ["documents"] = new JArray(), ["queries"] = new JArray(),
                ["timing_scope"] = "Main-thread wall time, including tokenization, GPU scheduling/fences, final vector download and CPU index ranking; not GPU-only time" };
            var inputDocuments = new List<SearchDocument>(); var vectors = new List<float[]>();
            foreach (var row in documents)
            {
                var id = (string)row["id"]; var timer = Stopwatch.StartNew();
                if ((string)row["kind"] == "text") actual.Add(id, text.EmbedDocument((string)row["text"], (string)row["title"]));
                timer.Stop(); var vector = actual[id]; var cosine = PlayerValidation.CheckEmbedding(vector, expected[id], precision);
                inputDocuments.Add(new SearchDocument(id, (string)row["text"], (string)row["title"])); vectors.Add(vector);
                ((JArray)result["documents"]).Add(new JObject { ["id"] = id, ["kind"] = row["kind"].DeepClone(),
                    ["embedding"] = new JArray(vector), ["cosine"] = cosine,
                    ["text_document_milliseconds"] = (string)row["kind"] == "text" ? timer.Elapsed.TotalMilliseconds : (double?)null });
            }
            var captured = new QueryCapture(text); var index = new MultimodalSearchIndex(inputDocuments, vectors, captured);
            double maximumError = 0;
            foreach (var query in queries)
            {
                var timer = Stopwatch.StartNew(); var hits = index.Search((string)query["text"], 18); timer.Stop();
                var q = Vector(query["expected"]); var cosine = PlayerValidation.CheckEmbedding(captured.Last, q, precision);
                var ranks = expected.Select(pair => (id: pair.Key, score: Cosine(q, pair.Value)))
                    .OrderByDescending(v => v.score).ThenBy(v => v.id, StringComparer.Ordinal).ToArray();
                if (hits.Length != 18 || !hits.Select(h => h.Document.Id).SequenceEqual(ranks.Select(r => r.id)))
                    throw new InvalidOperationException("Mixed-index ranks differ from the complete original reference.");
                var captures = new JArray();
                for (var i = 0; i < ranks.Length; i++)
                {
                    var error = Math.Abs(hits[i].Score - ranks[i].score);
                    if (double.IsNaN(error) || double.IsInfinity(error) || error > .001)
                        throw new InvalidOperationException("Mixed-index score differs from the unchanged original tolerance.");
                    maximumError = Math.Max(maximumError, error);
                    captures.Add(new JObject { ["id"] = hits[i].Document.Id, ["score"] = hits[i].Score, ["reference_score"] = ranks[i].score });
                }
                ((JArray)result["queries"]).Add(new JObject { ["id"] = query["id"].DeepClone(), ["embedding"] = new JArray(captured.Last),
                    ["cosine"] = cosine, ["milliseconds"] = timer.Elapsed.TotalMilliseconds, ["hits"] = captures });
            }
            result["maximum_reference_score_error"] = maximumError; result["success"] = true; return result;
        }
    }
}
