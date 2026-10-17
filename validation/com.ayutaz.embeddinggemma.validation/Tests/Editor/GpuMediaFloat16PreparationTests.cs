using System;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.ExceptionServices;
using System.Security.Cryptography;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using Unity.InferenceEngine;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class GpuMediaFloat16PreparationTests
    {
        string directory, source, destination, sourceHash, sourceModel, coefficient;
        const string Producer = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
        static string Hash(string path)
        { using var stream = File.OpenRead(path); using var hash = SHA256.Create(); return BitConverter.ToString(hash.ComputeHash(stream)).Replace("-", "").ToLowerInvariant(); }
        [SetUp] public void Setup()
        {
            directory = Path.Combine(Path.GetTempPath(), "embeddinggemma-float16-" + Guid.NewGuid().ToString("N"));
            source = Path.Combine(directory, "source"); destination = Path.Combine(directory, "derived");
            Directory.CreateDirectory(directory);
            var graph = new FunctionalGraph(); var input = graph.AddInput<float>(new TensorShape(1, 2), "input");
            var model = graph.Compile(Functional.MatMul(input, Functional.Constant(new TensorShape(2, 3), new[] {.1f, -.25f, .75f, 1.5f, 2f, -3f})));
            var writer = new GpuMediaBundleWriter(source, new string('a', 40)); sourceModel = writer.StoreModel(model);
            var tokenizer = Path.Combine(directory, "tokenizer.json"); File.WriteAllText(tokenizer, "{}");
            var json = writer.StoreFile(tokenizer); coefficient = writer.StoreWeights(new[] {.1f, -.25f, -0f});
            foreach (var name in new[] {"text", "text-alias"})
                writer.CommitModule(name, new JObject { ["kind"] = "text", ["model"] = sourceModel, ["tokenizer"] = json });
            sourceHash = Hash(Path.Combine(source, "bundle.json"));
        }
        [TearDown] public void Cleanup() { if (Directory.Exists(directory)) Directory.Delete(directory, true); }
        JObject Prepare(string output = null, string expected = null, string producer = Producer)
        {
            var type = AppDomain.CurrentDomain.GetAssemblies().Select(a => a.GetType("EmbeddingGemma.Validation.Tests.GpuMediaFloat16Preparation")).FirstOrDefault(t => t != null);
            Assert.That(type, Is.Not.Null, "Reproducible Float16 weight derivation is missing.");
            try { return (JObject)type.GetMethod("Prepare").Invoke(null, new object[] {source, output ?? destination, producer, expected ?? sourceHash}); }
            catch (TargetInvocationException error) { ExceptionDispatchInfo.Capture(error.InnerException).Throw(); throw; }
        }
        [Test] public void QuantizesModelWeightsPreservesCoefficientBitsAndDeduplicatesSharedModules()
        {
            var before = Directory.GetFiles(source, "*", SearchOption.AllDirectories).ToDictionary(path => path, Hash);
            var result = Prepare(); var bundle = GpuMediaBundle.Open(destination);
            var converted = (string)bundle.Module("text")["model"];
            Assert.That(converted, Is.Not.EqualTo(sourceModel));
            Assert.That((string)bundle.Module("text-alias")["model"], Is.EqualTo(converted));
            Assert.That(ModelLoader.Load(bundle.VerifyFile(converted)).constants.Any(c => c.dataType == DataType.Short), Is.True);
            Assert.That(File.ReadAllBytes(bundle.VerifyFile(coefficient)), Is.EqualTo(File.ReadAllBytes(Path.Combine(source, coefficient))));
            Assert.That((string)result["precision"], Is.EqualTo("float16-weights"));
            Assert.That((bool)result["inference_executed"], Is.False);
            Assert.That((int)result["module_count"], Is.EqualTo(2)); Assert.That((int)result["model_count"], Is.EqualTo(1));
            foreach (var row in before) Assert.That(Hash(row.Key), Is.EqualTo(row.Value), "Source files must remain unchanged.");
        }
        [TestCase(false)] [TestCase(true)] public void WrongDescriptorOrCorruptObjectCannotCreateOutput(bool corrupt)
        {
            if (corrupt) using (var stream = File.Open(Path.Combine(source, sourceModel), FileMode.Append)) stream.WriteByte(1);
            Assert.Throws<InvalidDataException>(() => Prepare(expected: corrupt ? sourceHash : new string('0', 64)));
            Assert.That(Directory.Exists(destination), Is.False);
        }
        [TestCase("same")] [TestCase("nested")] [TestCase("ancestor")]
        public void OutputCannotOverlapItsSource(string kind)
        {
            var output = kind == "same" ? source : kind == "nested" ? Path.Combine(source, "derived") : directory;
            Assert.Throws<ArgumentException>(() => Prepare(output)); Assert.That(Hash(Path.Combine(source, "bundle.json")), Is.EqualTo(sourceHash));
        }
        [Test] public void ForeignOutputIsNotAdopted()
        {
            Directory.CreateDirectory(destination); var path = Path.Combine(destination, "keep.txt"); File.WriteAllText(path, "keep");
            Assert.Throws<IOException>(() => Prepare()); Assert.That(File.ReadAllText(path), Is.EqualTo("keep"));
        }
        [Test] public void RetryPreservesPublishedDescriptorAndRejectsDifferentProducer()
        {
            Prepare(); var path = Path.Combine(destination, "bundle.json"); var before = File.ReadAllBytes(path);
            Prepare(); Assert.That(File.ReadAllBytes(path), Is.EqualTo(before));
            Assert.Throws<InvalidDataException>(() => Prepare(producer: new string('c', 40)));
            Assert.That(File.ReadAllBytes(path), Is.EqualTo(before));
        }
        [Test] public void InterruptedOwnedPreparationResumesItsVerifiedObjects()
        {
            var writer = new GpuMediaBundleWriter(destination, Producer); writer.StoreFile(Path.Combine(source, coefficient));
            Assert.That(File.Exists(Path.Combine(destination, "bundle.json")), Is.False);
            Prepare(); var bundle = GpuMediaBundle.Open(destination);
            Assert.That(File.Exists(bundle.VerifyFile((string)bundle.Module("text")["model"])), Is.True);
        }
    }
}
