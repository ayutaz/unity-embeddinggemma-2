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
    }
}
