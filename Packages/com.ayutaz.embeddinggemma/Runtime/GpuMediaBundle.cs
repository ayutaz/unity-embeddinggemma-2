using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace EmbeddingGemma
{
    /// <summary>Fixed-revision GPU pipeline recipes; verifies prepared bytes before a model or coefficient is loaded.</summary>
    public sealed partial class GpuMediaBundle
    {
        public const string PinnedRevision = "914f7f89142e33e77833254d9c9b90c3cef7303b";
        readonly string directory;
        readonly JObject descriptor;
        static readonly HashSet<string> OperationKinds = new HashSet<string>(StringComparer.Ordinal) {
            "blocked_matmul", "fixed_left_matmul", "batched_matmul", "gelu", "softmax", "silu", "tanh", "scalar_divide", "glu", "sequential_scale" };

        GpuMediaBundle(string directory, JObject descriptor) { this.directory = directory; this.descriptor = descriptor; }

        public static GpuMediaBundle Open(string directory)
        {
            if (string.IsNullOrWhiteSpace(directory)) throw new ArgumentException("A prepared filesystem directory is required.", nameof(directory));
            directory = Path.GetFullPath(directory);
            try
            {
                var data = JObject.Parse(File.ReadAllText(Path.Combine(directory, "bundle.json")));
                Require((int?)data["schema_version"] == 1 && (string)data["backend"] == "GPUCompute"
                    && (string)data["sentis_version"] == "2.6.1" && (string)data["model_revision"] == PinnedRevision
                    && Regex.IsMatch((string)data["source_commit"] ?? "", "\\A[a-f0-9]{40}\\z"), "Pinned GPU model provenance is required.");
                var bundle = new GpuMediaBundle(directory, data); bundle.Validate(); return bundle;
            }
            catch (Exception error) when (error is JsonException || error is ArgumentException || error is OverflowException || error is InvalidCastException)
            { throw new InvalidDataException("Invalid GPU media bundle metadata.", error); }
        }

        static void Require(bool condition, string message) { if (!condition) throw new InvalidDataException(message); }
        static bool Text(JToken value) => value?.Type == JTokenType.String && !string.IsNullOrWhiteSpace((string)value);
        static bool Positive(JToken value) => value?.Type == JTokenType.Integer && (long)value > 0 && (long)value <= int.MaxValue;
        string Resolve(string name)
        {
            Require(!string.IsNullOrEmpty(name) && !Path.IsPathRooted(name) && name.IndexOfAny(new[] { '\\', ':' }) < 0
                && name.Split('/').All(part => part.Length > 0 && part != "." && part != ".."), "Bundle paths must be relative canonical file names.");
            var path = Path.GetFullPath(Path.Combine(directory, name));
            Require(path.StartsWith(directory.TrimEnd(Path.DirectorySeparatorChar) + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase), "Bundle file escapes its directory.");
            return path;
        }
        void FileReference(JToken token, bool model = false)
        {
            Require(Text(token), "A declared file reference is required."); var name = (string)token; Resolve(name);
            Require(descriptor["files"]?[name] is JObject, "A recipe references an undeclared file: " + name);
            Require(!model || name.EndsWith(".sentis", StringComparison.Ordinal), "Prepared models must be .sentis files.");
        }
        void Validate()
        {
            Require(descriptor["files"] is JObject files && files.HasValues, "Prepared files are required.");
            foreach (var file in ((JObject)descriptor["files"]).Properties())
            {
                Resolve(file.Name); var value = file.Value;
                Require(value is JObject && value["bytes"]?.Type == JTokenType.Integer && (long)value["bytes"] >= 0
                    && Regex.IsMatch((string)value["sha256"] ?? "", "\\A[a-f0-9]{64}\\z"), "Prepared file size and SHA256 are required.");
            }
            Require(descriptor["modules"] is JObject modules && modules.HasValues, "At least one media module is required.");
            foreach (var entry in ((JObject)descriptor["modules"]).Properties())
            {
                Require(Regex.IsMatch(entry.Name, "\\A[a-z0-9][a-z0-9_-]*\\z") && entry.Value is JObject, "Invalid module identifier.");
                var module = (JObject)entry.Value; var kind = (string)module["kind"];
                Require(kind == "text" || kind == "image" || kind == "audio", "Unknown media module kind.");
                Require(module["backend"] == null || (string)module["backend"] == "GPUCompute", "Every media module must use GPUCompute.");
                FileReference(module["tokenizer"]);
                if (kind == "text") { FileReference(module["model"], true); continue; }
                FileReference(module["tokenizer_config"]); FileReference(module["backbone"], true);
                if (kind == "image")
                {
                    Require(Positive(module["resized_width"]) && Positive(module["resized_height"]), "Image geometry is required.");
                    _ = new ImageEncoderProfile((int)module["resized_width"], (int)module["resized_height"]);
                }
                else
                {
                    Require(Positive(module["valid_audio_frames"]), "A fixed valid audio prefix is required.");
                    _ = new AudioEncoderProfile((int)module["valid_audio_frames"]); FileReference(module["preprocessor"], true);
                }
                Require(module["models"] is JArray models && models.Count > 0 && module["operations"] is JArray operations
                    && models.Count == operations.Count + 1, "Every operation must connect two declared model segments.");
                foreach (var model in (JArray)module["models"]) FileReference(model, true);
                foreach (var operation in (JArray)module["operations"]) ValidateOperation(operation);
            }
        }
        void ValidateOperation(JToken operation)
        {
            var kind = (string)operation?["kind"];
            Require(operation is JObject && OperationKinds.Contains(kind ?? ""), "Unknown GPU operation.");
            Require(operation["inputs"] is JArray inputs && inputs.Count == (kind == "batched_matmul" ? 2 : 1)
                && inputs.All(Text) && Text(operation["output"]), "Explicit GPU operation bindings are required.");
            if (kind == "blocked_matmul" || kind == "fixed_left_matmul" || kind == "sequential_scale") FileReference(operation["weight_file"]);
            if (kind == "blocked_matmul")
            {
                Require(Positive(operation["width"]) && Positive(operation["columns"]) && Positive(operation["block_size"]), "Static matrix dimensions and block size are required.");
                Require((long)descriptor["files"][(string)operation["weight_file"]]["bytes"] == (long)operation["width"] * (long)operation["columns"] * 4, "Static matrix coefficient size differs.");
            }
            if (kind == "fixed_left_matmul") Require(operation["weight_shape"] is JArray shape && shape.Count > 0 && shape.All(Positive), "Fixed-left coefficient geometry is required.");
            if (kind == "gelu") Require(operation["shape"] is JArray shape && shape.Count > 0 && shape.All(Positive)
                && operation["release_cache"]?.Type == JTokenType.Boolean && operation["fuse_inner"]?.Type == JTokenType.Boolean
                && operation["sleef_tanh"]?.Type == JTokenType.Boolean, "Qualified GELU shape and arithmetic options are required.");
            if (kind == "softmax") Require((int?)operation["vector_width"] == 8 || (int?)operation["vector_width"] == 16, "Qualified softmax vector width is required.");
            if (kind == "scalar_divide" || kind == "sequential_scale")
            {
                var value = operation[kind == "scalar_divide" ? "divisor" : "first_scale"];
                Require(value != null && (value.Type == JTokenType.Float || value.Type == JTokenType.Integer)
                    && !float.IsNaN((float)value) && !float.IsInfinity((float)value) && (kind != "scalar_divide" || (float)value != 0), "Finite qualified operation scaling is required.");
            }
        }

        public JObject Module(string name)
        {
            if (name == null || !(descriptor["modules"]?[name] is JObject value)) throw new KeyNotFoundException("Unknown media module: " + name);
            return (JObject)value.DeepClone();
        }
        public string VerifyFile(string name)
        {
            var path = Resolve(name); var metadata = descriptor["files"]?[name];
            Require(metadata is JObject, "Only declared prepared files may be loaded.");
            using var stream = File.OpenRead(path);
            Require(stream.Length == (long)metadata["bytes"], "Prepared file length differs: " + name);
            Require(BundleStreamHash.Compute(stream, out _) == (string)metadata["sha256"], "Prepared file SHA256 differs: " + name);
            return path;
        }
    }
}
