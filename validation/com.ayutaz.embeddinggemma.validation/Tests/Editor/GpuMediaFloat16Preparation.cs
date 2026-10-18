using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text.RegularExpressions;
using Newtonsoft.Json.Linq;
using Unity.InferenceEngine;

namespace EmbeddingGemma.Validation.Tests
{
    /// <summary>CPU preparation of supported model weights; external FP32 coefficients remain bit-identical.</summary>
    public static class GpuMediaFloat16Preparation
    {
        public const string OriginalDescriptorSha256 = "8363f9a63a95b307fbbb835f3551a41f6fa5a1d21aca3017670e6f678647e955";
        static string Hash(string path)
        { using var stream = File.OpenRead(path); using var hash = SHA256.Create(); return BitConverter.ToString(hash.ComputeHash(stream)).Replace("-", "").ToLowerInvariant(); }
        static string Root(string path) => Path.GetFullPath(path).TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar) + Path.DirectorySeparatorChar;
        static string Inputs(Model model) => string.Join("|", model.inputs.Select(input => input.name + ":" + input.dataType + ":" + input.shape));

        public static JObject Prepare(string sourceDirectory, string outputDirectory, string producer, string expectedDescriptorSha256)
        {
            var source = Root(sourceDirectory); var output = Root(outputDirectory);
            if (source.StartsWith(output, StringComparison.OrdinalIgnoreCase) || output.StartsWith(source, StringComparison.OrdinalIgnoreCase))
                throw new ArgumentException("Source and output preparation directories must not overlap.");
            var descriptorPath = Path.Combine(source, "bundle.json");
            if (Hash(descriptorPath) != expectedDescriptorSha256) throw new InvalidDataException("Source descriptor differs from the pinned input.");
            var bundle = GpuMediaBundle.Open(source);
            var descriptor = JObject.Parse(File.ReadAllText(descriptorPath));
            var files = ((JObject)descriptor["files"]).Properties().OrderBy(file => file.Name, StringComparer.Ordinal).ToArray();
            // Validate every object before creating or mutating the output, including files shared by modules.
            var verified = files.ToDictionary(file => file.Name, file => bundle.VerifyFile(file.Name), StringComparer.Ordinal);
            var writer = new GpuMediaBundleWriter(output, producer);
            var mapped = new Dictionary<string, string>(StringComparer.Ordinal);
            var models = new JArray();
            var report = new JObject { ["success"] = false, ["precision"] = "float16-weights", ["inference_executed"] = false,
                ["source_descriptor_sha256"] = expectedDescriptorSha256, ["source_producer"] = descriptor["source_commit"],
                ["producer_source"] = producer, ["model_revision"] = descriptor["model_revision"], ["models"] = models,
                ["coefficient_precision"] = "float32-unchanged", ["scope"] = "Sentis Float16 weight quantization for eligible model constants; not all-half computation or a numerical/GPU pass." };
            void Save() => File.WriteAllText(Path.Combine(output, "float16-preparation.json"), report.ToString());
            Save();
            foreach (var file in files)
            {
                report["phase"] = file.Name; Save();
                if (file.Name.EndsWith(".sentis", StringComparison.Ordinal))
                {
                    var model = ModelLoader.Load(verified[file.Name]);
                    var inputs = Inputs(model); var outputs = model.outputs.Select(value => value.name).ToArray();
                    var beforeShort = model.constants.Count(value => value.dataType == DataType.Short);
                    ModelQuantizer.QuantizeWeights(QuantizationType.Float16, ref model);
                    if (inputs != Inputs(model) || !outputs.SequenceEqual(model.outputs.Select(value => value.name)))
                        throw new InvalidDataException("Weight conversion changed external model bindings.");
                    var converted = writer.StoreModel(model); mapped.Add(file.Name, converted);
                    models.Add(new JObject { ["source"] = file.Name, ["source_sha256"] = file.Value["sha256"],
                        ["output"] = converted, ["output_sha256"] = Hash(Path.Combine(output, converted)),
                        ["short_constants_before"] = beforeShort, ["short_constants_after"] = model.constants.Count(value => value.dataType == DataType.Short) });
                }
                else
                {
                    var copied = writer.StoreFile(verified[file.Name]); mapped.Add(file.Name, copied);
                    if (Hash(Path.Combine(output, copied)) != (string)file.Value["sha256"])
                        throw new InvalidDataException("A non-model source object changed during preparation.");
                }
                Save();
            }
            foreach (var entry in ((JObject)descriptor["modules"]).Properties().OrderBy(module => module.Name, StringComparer.Ordinal))
            {
                var module = (JObject)entry.Value.DeepClone();
                foreach (var value in module.Descendants().OfType<JValue>())
                    if (value.Type == JTokenType.String && mapped.TryGetValue((string)value, out var replacement)) value.Value = replacement;
                writer.CommitModule(entry.Name, module);
            }
            report["module_count"] = ((JObject)descriptor["modules"]).Count;
            report["model_count"] = models.Count; report["copied_object_count"] = mapped.Count - models.Count;
            report["bundle_descriptor_sha256"] = Hash(Path.Combine(output, "bundle.json"));
            report["phase"] = "prepared"; report["success"] = true; Save(); return report;
        }

        public static void PrepareFromEnvironment()
        {
            Prepare(Environment.GetEnvironmentVariable("EG_FLOAT16_SOURCE"), Environment.GetEnvironmentVariable("EG_FLOAT16_OUTPUT"),
                Environment.GetEnvironmentVariable("EG_ACCEPTANCE_SOURCE"), OriginalDescriptorSha256);
        }

        public static void PreparePinnedFromEnvironment()
        {
            var expected = Environment.GetEnvironmentVariable("EG_FLOAT16_DESCRIPTOR_SHA256");
            var producer = Environment.GetEnvironmentVariable("EG_ACCEPTANCE_SOURCE");
            if (!Regex.IsMatch(expected ?? "", "\\A[0-9a-f]{64}\\z") || !Regex.IsMatch(producer ?? "", "\\A[0-9a-f]{40}\\z"))
                throw new ArgumentException("An explicit lowercase descriptor hash and producer commit are required.");
            Prepare(Environment.GetEnvironmentVariable("EG_FLOAT16_SOURCE"), Environment.GetEnvironmentVariable("EG_FLOAT16_OUTPUT"),
                producer, expected);
        }
    }
}
