using System;
using System.IO;
using System.Reflection;
using System.Security.Cryptography;
using Newtonsoft.Json.Linq;
using NUnit.Framework;

namespace EmbeddingGemma.Tests
{
    public sealed class GpuMediaBundleTests
    {
        string directory;
        JObject manifest;
        [SetUp] public void Setup()
        {
            directory = Path.Combine(Path.GetTempPath(), "embeddinggemma-media-bundle-" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(directory); var files = new JObject();
            foreach (var name in new[] { "part.sentis", "backbone.sentis", "tokenizer.json", "config.json" })
            {
                var bytes = System.Text.Encoding.UTF8.GetBytes("synthetic-file-contract:" + name);
                File.WriteAllBytes(Path.Combine(directory, name), bytes); using var hash = SHA256.Create();
                files[name] = new JObject { ["bytes"] = bytes.Length, ["sha256"] = BitConverter.ToString(hash.ComputeHash(bytes)).Replace("-", "").ToLowerInvariant() };
            }
            manifest = new JObject { ["schema_version"] = 1, ["backend"] = "GPUCompute", ["sentis_version"] = "2.6.1",
                ["model_revision"] = "914f7f89142e33e77833254d9c9b90c3cef7303b", ["source_commit"] = new string('a', 40), ["files"] = files,
                ["modules"] = new JObject { ["image-square"] = new JObject { ["kind"] = "image", ["resized_width"] = 768, ["resized_height"] = 768,
                    ["models"] = new JArray("part.sentis"), ["operations"] = new JArray(), ["backbone"] = "backbone.sentis",
                    ["tokenizer"] = "tokenizer.json", ["tokenizer_config"] = "config.json" } } };
        }
        [TearDown] public void Cleanup() { if (Directory.Exists(directory)) Directory.Delete(directory, true); }
        static Type Required()
        {
            var type = typeof(TextEmbedder).Assembly.GetType("EmbeddingGemma.GpuMediaBundle");
            Assert.That(type, Is.Not.Null, "An audited reusable GPU media bundle is required."); return type;
        }
        object Open()
        {
            File.WriteAllText(Path.Combine(directory, "bundle.json"), manifest.ToString());
            return Required().GetMethod("Open").Invoke(null, new object[] { directory });
        }
        static JObject Module(object bundle) => (JObject)Required().GetMethod("Module").Invoke(bundle, new object[] { "image-square" });
        static string Verify(object bundle, string name) => (string)Required().GetMethod("VerifyFile").Invoke(bundle, new object[] { name });

        [Test] public void FixedRevisionBundleExposesOwnedModuleMetadataAndAuditsFilesBeforeLoading()
        {
            var bundle = Open(); var module = Module(bundle); module["kind"] = "changed";
            Assert.That((string)Module(bundle)["kind"], Is.EqualTo("image"));
            Assert.That(Verify(bundle, "part.sentis"), Is.EqualTo(Path.Combine(directory, "part.sentis")));
        }
        [TestCase("backend")] [TestCase("model_revision")] [TestCase("source_commit")] [TestCase("sentis_version")]
        public void MixedModelOrExecutionProvenanceIsRejected(string field)
        {
            manifest[field] = "unqualified";
            Assert.That(Assert.Throws<TargetInvocationException>(() => Open()).InnerException, Is.InstanceOf<InvalidDataException>());
        }
        [TestCase("../escape.sentis")] [TestCase("/absolute.sentis")] [TestCase("C:/absolute.sentis")]
        public void BundleEntriesMustStayInsideThePreparedDirectory(string name)
        {
            manifest["files"][name] = manifest["files"]["part.sentis"].DeepClone();
            Assert.That(Assert.Throws<TargetInvocationException>(() => Open()).InnerException, Is.InstanceOf<InvalidDataException>());
        }
        [TestCase("missing_model")] [TestCase("missing_backbone")] [TestCase("unknown_kind")]
        [TestCase("operation_count")] [TestCase("unknown_operation")] [TestCase("invalid_geometry")]
        public void IncompletePipelineRecipesAreRejectedBeforeWorkersAreCreated(string problem)
        {
            var module = (JObject)manifest["modules"]["image-square"];
            if (problem == "missing_model") module["models"] = new JArray("absent.sentis");
            if (problem == "missing_backbone") module.Remove("backbone");
            if (problem == "unknown_kind") module["kind"] = "cpu_encoder";
            if (problem == "operation_count") module["operations"] = new JArray(new JObject { ["kind"] = "silu", ["inputs"] = new JArray("input"), ["output"] = "output" });
            if (problem == "unknown_operation")
            {
                module["models"] = new JArray("part.sentis", "part.sentis");
                module["operations"] = new JArray(new JObject { ["kind"] = "cpu_fallback", ["inputs"] = new JArray("input"), ["output"] = "output" });
            }
            if (problem == "invalid_geometry") module["resized_width"] = 769;
            Assert.That(Assert.Throws<TargetInvocationException>(() => Open()).InnerException, Is.InstanceOf<InvalidDataException>());
        }
        [TestCase("length")] [TestCase("same_length")]
        public void CorruptedPreparedFilesCannotBeUsedEvenIfTheirNamesMatch(string problem)
        {
            var bundle = Open(); var path = Path.Combine(directory, "part.sentis"); var bytes = File.ReadAllBytes(path);
            if (problem == "length") File.WriteAllText(path, "changed");
            else { bytes[0] ^= 1; File.WriteAllBytes(path, bytes); }
            Assert.That(Assert.Throws<TargetInvocationException>(() => Verify(bundle, "part.sentis")).InnerException, Is.InstanceOf<InvalidDataException>());
        }
        [Test] public void UndeclaredFilesAreNeverLoaded()
        {
            var bundle = Open();
            Assert.That(Assert.Throws<TargetInvocationException>(() => Verify(bundle, "unlisted.sentis")).InnerException, Is.InstanceOf<InvalidDataException>());
        }
    }
}
