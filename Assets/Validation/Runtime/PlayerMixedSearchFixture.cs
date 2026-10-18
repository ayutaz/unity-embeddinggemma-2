using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using EmbeddingGemma;
using Newtonsoft.Json.Linq;

namespace EmbeddingGemma.Validation
{
    /// <summary>Original fixed text bytes and complete all-role media captures feed the mixed-index gate.</summary>
    public static class PlayerMixedSearchFixture
    {
        public const string TextReferenceSha256 = "f316d5199c920431c7f721d43b2cedbe4a6384fb0e0664511b75bc2e98fe80fe";
        static readonly string[] TextIds = { "doc-cat-ja", "doc-cat-en", "doc-space-ja", "doc-space-en", "doc-recipe-a", "doc-recipe-b" };
        static readonly string[] QueryIds = { "query-cat-ja", "query-space-en", "query-recipe-ja", "query-cat-en" };
        static readonly TextRole[] Roles = { TextRole.Document, TextRole.Query, TextRole.Raw };
        static bool Text(JToken token) => token?.Type == JTokenType.String && !string.IsNullOrWhiteSpace((string)token);
        static float[] Vector(JToken token)
        {
            if (!(token is JArray a) || a.Count != 768 || a.Any(v => v.Type != JTokenType.Integer && v.Type != JTokenType.Float))
                throw new ArgumentException("A complete original numeric vector is required.");
            return a.Values<float>().ToArray();
        }
        static JArray TextRows(JToken input, string[] ids, string role)
        {
            if (!(input is JArray rows) || rows.Count != ids.Length || rows.Any(r => !(r is JObject) || !Text(r["id"]) || !Text(r["text"]) || (string)r["role"] != role)
                || !rows.Select(r => (string)r["id"]).OrderBy(v => v, StringComparer.Ordinal).SequenceEqual(ids.OrderBy(v => v, StringComparer.Ordinal)))
                throw new ArgumentException("All fixed original text IDs, inputs and roles are required.");
            var result = new JArray();
            foreach (var row in rows)
            {
                var vector = Vector(row["embedding"]); PlayerValidation.CheckEmbedding(vector, vector, "fp32");
                result.Add(new JObject { ["id"] = row["id"].DeepClone(), ["kind"] = "text", ["text"] = row["text"].DeepClone(),
                    ["title"] = row["title"]?.DeepClone(), ["expected"] = new JArray(vector) });
            }
            return result;
        }
        public static JObject ParseTextReference(JObject reference)
        {
            if (reference == null) throw new ArgumentException("A complete original text reference is required.");
            return new JObject { ["documents"] = TextRows(reference["documents"], TextIds, "document"),
                ["queries"] = TextRows(reference["queries"], QueryIds, "query"), ["original_bytes_verified"] = false };
        }
        public static JObject LoadText(string directory)
        {
            const string name = "search-reference.json"; var file = Path.Combine(directory, name); var bytes = new FileInfo(file).Length;
            if (bytes > 4 * 1024 * 1024) throw new InvalidDataException("Original text reference exceeds its bounded allowance.");
            var verified = PlayerMediaFixture.VerifyFile(directory, name, bytes, TextReferenceSha256);
            var result = ParseTextReference(JObject.Parse(File.ReadAllText(verified)));
            result["original_bytes_verified"] = true; result["reference_sha256"] = TextReferenceSha256; return result;
        }
        static string[] MediaIds(string module)
        {
            if (module == "audio-1119") return new[] { "silence", "impulse", "sine", "nyquist", "noise" };
            if (module == "audio-839") return new[] { "padded_noise" };
            if (module == "image-square" || module == "image-landscape" || module == "image-portrait")
            { var shape = module.Substring(6); return new[] { shape + "-0", shape + "-1" }; }
            throw new ArgumentException("All mixed media inputs must come from one of the five fixed modules.");
        }
        static string Key(string id, string role) => id + "|" + role;
        public static void AppendMedia(JArray documents, JObject mediaVectors, string module, IReadOnlyList<PlayerMediaCase> cases, JObject report, string precision)
        {
            var ids = MediaIds(module); var required = ids.SelectMany(id => Roles.Select(role => Key(id, role.ToString().ToLowerInvariant()))).OrderBy(v => v, StringComparer.Ordinal).ToArray();
            if (documents == null || mediaVectors == null || cases == null || cases.Count != required.Length || cases.Any(r => r == null || !Roles.Contains(r.Role))
                || !cases.Select(r => Key(r.Id, r.Role.ToString().ToLowerInvariant())).OrderBy(v => v, StringComparer.Ordinal).SequenceEqual(required)
                || report?["success"]?.Type != JTokenType.Boolean || (bool)report["success"] != true
                || report?["session_released"]?.Type != JTokenType.Boolean || (bool)report["session_released"] != true
                || !(report["cases"] is JArray captures) || captures.Count != required.Length || captures.Any(r => !(r is JObject) || !Text(r["id"]) || !Text(r["role"]))
                || !captures.Select(r => Key((string)r["id"], (string)r["role"])).OrderBy(v => v, StringComparer.Ordinal).SequenceEqual(required)
                || !(report["roles"] is JArray roles) || roles.Count != 3
                || !roles.Values<string>().OrderBy(v => v, StringComparer.Ordinal).SequenceEqual(new[] { "document", "query", "raw" }))
                throw new ArgumentException("Complete unique role captures and a successfully released module are required.");
            var pending = new List<(string id, string kind, string original, float[] expected, float[] actual)>();
            foreach (var row in cases)
            {
                PlayerValidation.CheckEmbedding(row.Expected, row.Expected, precision);
                var capture = captures.Single(r => (string)r["id"] == row.Id && (string)r["role"] == row.Role.ToString().ToLowerInvariant());
                var vector = Vector(capture["embedding"]); PlayerValidation.CheckEmbedding(vector, row.Expected, precision);
                if (row.Role != TextRole.Document) continue;
                var kind = module.StartsWith("image-", StringComparison.Ordinal) ? "image" : "audio"; var id = kind + "-" + row.Id;
                if (documents.Any(d => (string)d["id"] == id) || mediaVectors.Property(id) != null)
                    throw new ArgumentException("A media document cannot replace an existing mixed-index entry.");
                pending.Add((id, kind, row.Id, row.Expected, vector));
            }
            // Validation above is complete before changing either output collection.
            foreach (var row in pending)
            {
                documents.Add(new JObject { ["id"] = row.id, ["kind"] = row.kind, ["text"] = "fixed " + row.kind + " " + row.original,
                    ["title"] = row.original, ["expected"] = new JArray(row.expected) });
                mediaVectors.Add(row.id, new JArray(row.actual));
            }
        }
    }
}
