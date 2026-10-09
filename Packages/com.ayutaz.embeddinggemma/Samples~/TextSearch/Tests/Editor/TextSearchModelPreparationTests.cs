using System;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using NUnit.Framework;
using Unity.InferenceEngine;

namespace EmbeddingGemma.Samples.Tests
{
    public sealed class TextSearchModelPreparationTests
    {
        string temporary;
        const string Tokenizer = @"{""added_tokens"":[],""normalizer"":null,""pre_tokenizer"":null,""post_processor"":null,""decoder"":null,""truncation"":null,""padding"":null,""model"":{""type"":""WordLevel"",""vocab"":{""[UNK]"":0},""unk_token"":""[UNK]""}}";

        [SetUp] public void CreateDirectory() { temporary = Path.Combine(Path.GetTempPath(), "embeddinggemma-search-" + Guid.NewGuid().ToString("N")); Directory.CreateDirectory(temporary); }
        [TearDown] public void Cleanup() { if (Directory.Exists(temporary)) Directory.Delete(temporary, true); }

        static Model Model()
        {
            var graph = new FunctionalGraph();
            var ids = graph.AddInput<int>(new TensorShape(1, 128), "input_ids");
            var mask = graph.AddInput<int>(new TensorShape(1, 128), "attention_mask");
            return graph.Compile(Functional.Concat(Enumerable.Repeat((ids + mask).Float(), 6).ToArray(), 1));
        }

        [Test]
        public void ProducesReloadableFormatsTokenizerAndVerifiedFileManifest()
        {
            var tokenizerPath = Path.Combine(temporary, "input-tokenizer.json");
            File.WriteAllText(tokenizerPath, Tokenizer);
            var target = Path.Combine(temporary, "output");
            var report = TextSearchModelPreparation.Prepare(Model(), tokenizerPath, target, new string('a', 40));
            Assert.That(report.success, Is.True);
            Assert.That(report.sourceCommit, Is.EqualTo(new string('a', 40)));
            Assert.That(report.files.Select(file => file.name), Is.EqualTo(new[] { "model-fp32.sentis", "model-float16.sentis", "tokenizer.json" }));
            foreach (var file in report.files)
            {
                var path = Path.Combine(target, file.name);
                Assert.That(new FileInfo(path).Length, Is.EqualTo(file.bytes));
                using var stream = File.OpenRead(path);
                using var hash = SHA256.Create();
                Assert.That(BitConverter.ToString(hash.ComputeHash(stream)).Replace("-", "").ToLowerInvariant(), Is.EqualTo(file.sha256));
                if (file.name.EndsWith(".sentis")) Assert.That(TextModelFile.Load(path).inputs, Has.Count.EqualTo(2));
            }
            Assert.That(File.ReadAllText(Path.Combine(target, "tokenizer.json")), Is.EqualTo(Tokenizer));
            Assert.That(File.Exists(Path.Combine(target, "preparation.json")), Is.True);
        }

        [Test]
        public void InvalidTokenizerFailsBeforeOverwritingExistingModels()
        {
            var target = Path.Combine(temporary, "output"); Directory.CreateDirectory(target);
            var previous = Path.Combine(target, "model-fp32.sentis"); File.WriteAllText(previous, "previous");
            var reportPath = Path.Combine(target, "preparation.json"); File.WriteAllText(reportPath, "{\"success\":true}");
            Assert.Throws<FileNotFoundException>(() => TextSearchModelPreparation.Prepare(Model(), Path.Combine(temporary, "missing.json"), target));
            var invalid = Path.Combine(temporary, "invalid.json"); File.WriteAllText(invalid, "invalid");
            Assert.Throws<ArgumentException>(() => TextSearchModelPreparation.Prepare(Model(), invalid, target));
            Assert.That(File.ReadAllText(previous), Is.EqualTo("previous"));
            Assert.That(UnityEngine.JsonUtility.FromJson<TextSearchModelPreparation.Report>(File.ReadAllText(reportPath)).success, Is.False,
                "A failed preparation must not leave a previous success receipt.");
        }

        [Test]
        public void VerifiedCacheDoesNotLoadModelOrRewriteFiles()
        {
            var tokenizer = Path.Combine(temporary, "input-tokenizer.json"); File.WriteAllText(tokenizer, Tokenizer);
            var target = Path.Combine(temporary, "output");
            var first = TextSearchModelPreparation.PrepareCached(Model, tokenizer, target, "model-source", "search-source", "model-sha256");
            var receipt = File.ReadAllText(Path.Combine(target, "preparation.json"));
            var before = first.files.Select(file => File.GetLastWriteTimeUtc(Path.Combine(target, file.name))).ToArray();
            var second = TextSearchModelPreparation.PrepareCached(() => throw new Exception("Cache hit must not load the model."),
                tokenizer, target, "model-source", "search-source", "model-sha256");
            Assert.That(second.success, Is.True);
            Assert.That(File.ReadAllText(Path.Combine(target, "preparation.json")), Is.EqualTo(receipt));
            Assert.That(second.files.Select(file => File.GetLastWriteTimeUtc(Path.Combine(target, file.name))), Is.EqualTo(before));
        }

        [TestCase("source")]
        [TestCase("search")]
        [TestCase("model-hash")]
        [TestCase("unity")]
        [TestCase("legacy")]
        [TestCase("failed")]
        [TestCase("duplicate")]
        [TestCase("missing")]
        [TestCase("corrupt")]
        [TestCase("tokenizer")]
        [TestCase("malformed")]
        public void ChangedInputsOrInvalidCacheAreRebuilt(string mutation)
        {
            var tokenizer = Path.Combine(temporary, "input-tokenizer.json"); File.WriteAllText(tokenizer, Tokenizer);
            var target = Path.Combine(temporary, "output");
            var first = TextSearchModelPreparation.PrepareCached(Model, tokenizer, target, "model-source", "search-source", "model-sha256");
            var path = Path.Combine(target, "preparation.json");
            var json = UnityEngine.JsonUtility.FromJson<TextSearchModelPreparation.Report>(File.ReadAllText(path));
            switch (mutation)
            {
                case "source": json.sourceCommit = "other"; break;
                case "search": json.searchSourceCommit = "other"; break;
                case "model-hash": json.modelSha256 = "other"; break;
                case "unity": json.unityVersion = "other"; break;
                case "legacy": json.modelSha256 = null; break;
                case "failed": json.success = false; break;
                case "duplicate": json.files[1] = json.files[0]; break;
                case "missing": File.Delete(Path.Combine(target, "model-float16.sentis")); break;
                case "corrupt":
                    var bytes = File.ReadAllBytes(Path.Combine(target, "model-fp32.sentis")); bytes[bytes.Length - 1] ^= 1;
                    File.WriteAllBytes(Path.Combine(target, "model-fp32.sentis"), bytes); break;
                case "tokenizer": File.WriteAllText(tokenizer, Tokenizer + " "); break;
            }
            File.WriteAllText(path, mutation == "malformed" ? "invalid-json" : UnityEngine.JsonUtility.ToJson(json, true));
            var loads = 0;
            var rebuilt = TextSearchModelPreparation.PrepareCached(() => { loads++; return Model(); }, tokenizer,
                target, "model-source", "search-source", "model-sha256");
            Assert.That(loads, Is.EqualTo(1), mutation);
            Assert.That(rebuilt.success, Is.True);
            Assert.That(File.ReadAllText(Path.Combine(target, "tokenizer.json")), Is.EqualTo(File.ReadAllText(tokenizer)));
            Assert.That(TextModelFile.Load(Path.Combine(target, "model-fp32.sentis")).inputs, Has.Count.EqualTo(2));
        }

        [Test]
        public void RebuildLoadFailureInvalidatesPreviousSuccessAndPreservesModels()
        {
            var tokenizer = Path.Combine(temporary, "input-tokenizer.json"); File.WriteAllText(tokenizer, Tokenizer);
            var target = Path.Combine(temporary, "output");
            TextSearchModelPreparation.PrepareCached(Model, tokenizer, target, "source", "search", "old-hash");
            var before = File.ReadAllBytes(Path.Combine(target, "model-fp32.sentis"));
            Assert.Throws<InvalidOperationException>(() => TextSearchModelPreparation.PrepareCached(
                () => throw new InvalidOperationException("load failed"), tokenizer, target, "source", "search", "new-hash"));
            var report = UnityEngine.JsonUtility.FromJson<TextSearchModelPreparation.Report>(File.ReadAllText(Path.Combine(target, "preparation.json")));
            Assert.That(report.success, Is.False);
            Assert.That(report.error, Is.EqualTo("load failed"));
            Assert.That(File.ReadAllBytes(Path.Combine(target, "model-fp32.sentis")), Is.EqualTo(before));
        }
    }
}
