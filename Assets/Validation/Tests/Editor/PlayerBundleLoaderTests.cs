using System;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using Newtonsoft.Json.Linq;
using NUnit.Framework;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class PlayerBundleLoaderTests
    {
        string directory;
        JObject receipt;
        [SetUp] public void Setup()
        {
            directory = Path.Combine(Path.GetTempPath(), "embeddinggemma-player-" + Guid.NewGuid().ToString("N")); Directory.CreateDirectory(directory);
            File.WriteAllText(Path.Combine(directory, "model-fp32.sentis"), "fp32"); File.WriteAllText(Path.Combine(directory, "model-float16.sentis"), "float16");
            File.WriteAllText(Path.Combine(directory, "tokenizer.json"), "{}");
            var reference = PlayerValidationTests.Reference();
            File.WriteAllText(Path.Combine(directory, "reference.json"), PlayerValidationTests.ReferenceJson(reference, false).ToString());
            File.WriteAllText(Path.Combine(directory, "search-reference.json"), PlayerValidationTests.ReferenceJson(reference, true).ToString());
            var files = new JArray();
            foreach (var name in new[] { "model-fp32.sentis", "model-float16.sentis", "tokenizer.json", "reference.json", "search-reference.json" })
            {
                var path = Path.Combine(directory, name); using var hash = SHA256.Create();
                files.Add(new JObject { ["name"] = name, ["bytes"] = new FileInfo(path).Length, ["sha256"] = BitConverter.ToString(hash.ComputeHash(File.ReadAllBytes(path))).Replace("-", "").ToLowerInvariant() });
            }
            receipt = new JObject { ["success"] = true, ["unity_executed"] = false, ["gpu_verified"] = false, ["files"] = files,
                ["model_revision"] = PlayerValidation.ModelRevision, ["model_source"] = new string('a', 40), ["search_source"] = new string('b', 40),
                ["model_origin_sha256"] = new string('c', 64), ["unity_preparation_version"] = "6000.3.16f1" };
            Save();
        }
        void Save() => File.WriteAllText(Path.Combine(directory, "bundle.json"), receipt.ToString());
        [TearDown] public void Cleanup() { Directory.Delete(directory, true); }
        [Test] public void AuditsEveryFileAndLoadsBothPinnedReferences()
        {
            var bundle = PlayerBundleLoader.Load(directory);
            Assert.That(bundle.Directory, Is.EqualTo(Path.GetFullPath(directory))); Assert.That(bundle.Reference.Cases, Has.Length.EqualTo(15));
            Assert.That(bundle.Reference.Queries, Has.Length.EqualTo(4)); Assert.That(bundle.TokenizerJson, Is.EqualTo("{}"));
            Assert.That((bool)bundle.Receipt["gpu_verified"], Is.False, "Hash validation does not prove GPU execution.");
        }
        [TestCase("success")] [TestCase("revision")] [TestCase("source")] [TestCase("unity")]
        public void UntrustedOrIncompatibleReceiptsAreRejected(string violation)
        {
            if (violation == "success") receipt["success"] = false;
            if (violation == "revision") receipt["model_revision"] = "other";
            if (violation == "source") receipt["model_source"] = "local";
            if (violation == "unity") receipt["unity_preparation_version"] = "6000.3.15f1";
            Save(); Assert.Throws<ArgumentException>(() => PlayerBundleLoader.Load(directory));
        }
        [TestCase("path")] [TestCase("duplicate")]
        public void ManifestCannotChooseFilesOutsideTheFixedBundle(string violation)
        {
            receipt["files"][1]["name"] = violation == "path" ? "../model-float16.sentis" : "model-fp32.sentis";
            Save(); Assert.Throws<ArgumentException>(() => PlayerBundleLoader.Load(directory));
        }
        [Test] public void ChangedBytesFailEvenWhenLengthAndMtimeMatch()
        {
            var path = Path.Combine(directory, "model-fp32.sentis"); var timestamp = File.GetLastWriteTimeUtc(path);
            File.WriteAllText(path, "oops"); File.SetLastWriteTimeUtc(path, timestamp);
            Assert.Throws<InvalidDataException>(() => PlayerBundleLoader.Load(directory));
        }
        [Test] public void MissingFileDoesNotReuseAnOldSuccessfulReceipt()
        {
            File.Delete(Path.Combine(directory, "tokenizer.json"));
            Assert.Throws<FileNotFoundException>(() => PlayerBundleLoader.Load(directory));
        }
        [TestCase("jar:file:///app.apk!/assets/EmbeddingGemmaValidation")]
        [TestCase("https://example.invalid/bundle")]
        public void UrlSourcesRequireAnExplicitExtractionStep(string source)
        {
            var exception = Assert.Throws<NotSupportedException>(() => PlayerBundleLoader.Load(source));
            Assert.That(exception.Message, Does.Contain("Extract"));
        }
    }
}
