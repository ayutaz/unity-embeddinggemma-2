using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using Newtonsoft.Json.Linq;

namespace EmbeddingGemma.Validation
{
    public sealed class PlayerBundle
    {
        public string Directory, TokenizerJson;
        public JObject Receipt;
        public PlayerReferenceSet Reference;
    }
    public static class PlayerBundleLoader
    {
        static readonly string[] FileNames = { "model-fp32.sentis", "model-float16.sentis", "tokenizer.json", "reference.json", "search-reference.json" };
        public static PlayerBundle Load(string directory)
        {
            // Android jar URLs need an explicit extraction step before this filesystem audit.
            if (directory != null && directory.Contains("://")) throw new NotSupportedException("Extract the StreamingAssets bundle to a filesystem directory before auditing it.");
            directory = Path.GetFullPath(directory);
            var receipt = JObject.Parse(File.ReadAllText(Path.Combine(directory, "bundle.json")));
            Require(receipt["success"]?.Type == JTokenType.Boolean && (bool)receipt["success"] &&
                (string)receipt["model_revision"] == PlayerValidation.ModelRevision &&
                (string)receipt["unity_preparation_version"] == "6000.3.16f1", "Successful pinned Unity bundle receipt is required.");
            foreach (var name in new[] { "model_source", "search_source" })
                Require(receipt[name]?.Type == JTokenType.String && Regex.IsMatch((string)receipt[name], "\\A[a-f0-9]{40}\\z"), "Invalid source commit: " + name);
            Require(receipt["model_origin_sha256"]?.Type == JTokenType.String && Regex.IsMatch((string)receipt["model_origin_sha256"], "\\A[a-f0-9]{64}\\z"), "Original model SHA-256 is missing.");
            var files = receipt["files"] as JArray;
            Require(files != null && files.Count == 5 && files.All(file => file is JObject && file["name"]?.Type == JTokenType.String) &&
                files.Select(file => (string)file["name"]).SequenceEqual(FileNames), "Bundle must declare exactly the five fixed file names in order.");
            var text = new Dictionary<string, string>();
            foreach (var entry in files)
            {
                var name = (string)entry["name"];
                Require(entry["bytes"]?.Type == JTokenType.Integer && (long)entry["bytes"] > 0 &&
                    entry["sha256"]?.Type == JTokenType.String && Regex.IsMatch((string)entry["sha256"], "\\A[a-f0-9]{64}\\z"), "Invalid bundle size or hash: " + name);
                using var stream = File.OpenRead(Path.Combine(directory, name));
                if (stream.Length != (long)entry["bytes"]) throw new InvalidDataException("Bundle file length differs: " + name);
                using var hash = SHA256.Create();
                var digest = BitConverter.ToString(hash.ComputeHash(stream)).Replace("-", "").ToLowerInvariant();
                if (digest != (string)entry["sha256"]) throw new InvalidDataException("Bundle file SHA-256 differs: " + name);
                if (name.EndsWith(".json", StringComparison.Ordinal))
                {
                    // Read the audited bytes while the same read-only handle is still open.
                    stream.Position = 0;
                    using var reader = new StreamReader(stream, Encoding.UTF8, true, 4096, leaveOpen: true);
                    text[name] = reader.ReadToEnd();
                }
            }
            return new PlayerBundle { Directory = directory, TokenizerJson = text["tokenizer.json"], Receipt = receipt,
                Reference = PlayerValidation.ParseReferences(JObject.Parse(text["reference.json"]), JObject.Parse(text["search-reference.json"]),
                    (string)receipt["model_source"], (string)receipt["search_source"]) };
        }
        static void Require(bool condition, string message) { if (!condition) throw new ArgumentException(message); }
    }
}
