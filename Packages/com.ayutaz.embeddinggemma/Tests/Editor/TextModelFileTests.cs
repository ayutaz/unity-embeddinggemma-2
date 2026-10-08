using System;
using System.IO;
using System.Text.RegularExpressions;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.TestTools;

namespace EmbeddingGemma.Tests
{
    public sealed class TextModelFileTests
    {
        string directory;
        [SetUp] public void Setup() => directory = Path.Combine(Path.GetTempPath(), "embeddinggemma-" + Guid.NewGuid());
        [TearDown] public void Cleanup() { if (Directory.Exists(directory)) Directory.Delete(directory, true); }

        static Model SmallModel()
        {
            var graph = new FunctionalGraph();
            var input = graph.AddInput<float>(new TensorShape(1, 2));
            return graph.Compile(Functional.MatMul(input,
                Functional.Constant(new TensorShape(2, 2), new[] { 1.12345f, 2.23456f, 3.34567f, 4.45678f })));
        }

        static float[] Run(Model model)
        {
            using var input = new Tensor<float>(new TensorShape(1, 2), new[] { 1f, 2f });
            using var worker = new Worker(model, BackendType.CPU);
            worker.Schedule(input);
            return ((Tensor<float>)worker.PeekOutput()).DownloadToArray();
        }

        [TestCase(false)] [TestCase(true)]
        public void RoundTripPreservesOutputAndDoesNotMutateSource(bool float16)
        {
            var source = SmallModel();
            var before = Run(source);
            var path = Path.Combine(directory, "nested/model.sentis");
            TextModelFile.Save(source, path, float16);
            Assert.That(File.Exists(path), Is.True);
            Assert.That(Run(TextModelFile.Load(path)), Is.EqualTo(before).Within(float16 ? 0.01 : 1e-6));
            Assert.That(Run(source), Is.EqualTo(before).Within(1e-6));
            Assert.That(Directory.GetFiles(Path.GetDirectoryName(path)), Has.Length.EqualTo(1));
        }

        [Test] public void InvalidArgumentsAreRejectedBeforeWriting()
        {
            Assert.Throws<ArgumentNullException>(() => TextModelFile.Save(null, "unused.sentis"));
            Assert.Throws<ArgumentException>(() => TextModelFile.Save(SmallModel(), ""));
            Assert.Throws<ArgumentException>(() => TextModelFile.Save(SmallModel(), Path.Combine(directory, "wrong.pt2")));
            Assert.That(Directory.Exists(directory), Is.False);
        }

        [Test] public void ExistingDestinationIsReplacedByValidModel()
        {
            Directory.CreateDirectory(directory);
            var path = Path.Combine(directory, "model.sentis");
            File.WriteAllText(path, "old data");
            var model = SmallModel();
            TextModelFile.Save(model, path);
            Assert.That(Run(TextModelFile.Load(path)), Is.EqualTo(Run(model)).Within(1e-6));
        }

        [Test] public void MissingOrCorruptModelCannotLoad()
        {
            Directory.CreateDirectory(directory);
            Assert.Throws<FileNotFoundException>(() => TextModelFile.Load(Path.Combine(directory, "missing.sentis")));
            var path = Path.Combine(directory, "bad.sentis");
            File.WriteAllBytes(path, new byte[] { 1, 2 });
            LogAssert.Expect(LogType.Error, new Regex("Failed to load serialized .sentis model"));
            Assert.Catch(() => TextModelFile.Load(path));
        }
    }
}
