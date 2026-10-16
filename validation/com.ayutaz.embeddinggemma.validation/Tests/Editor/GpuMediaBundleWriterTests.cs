using System;
using System.IO;
using System.Linq;
using System.Reflection;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using Unity.InferenceEngine;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class GpuMediaBundleWriterTests
    {
        string directory, output;
        [SetUp] public void Setup() { directory = Path.Combine(Path.GetTempPath(), "embeddinggemma-bundle-writer-" + Guid.NewGuid().ToString("N")); Directory.CreateDirectory(directory); output = Path.Combine(directory, "bundle"); }
        [TearDown] public void Cleanup() { if (Directory.Exists(directory)) Directory.Delete(directory, true); }
        static Type Required()
        {
            var type = AppDomain.CurrentDomain.GetAssemblies().Select(a => a.GetType("EmbeddingGemma.Validation.Tests.GpuMediaBundleWriter")).FirstOrDefault(t => t != null);
            Assert.That(type, Is.Not.Null, "Qualified GPU recipes need a reusable immutable preparation writer."); return type;
        }
        object Create(string source = null) => Activator.CreateInstance(Required(), new object[] { output, source ?? new string('a', 40) });
        static object Call(object writer, string name, params object[] args) => Required().GetMethod(name).Invoke(writer, args);
        static Model Tiny()
        {
            var graph = new FunctionalGraph(); var input = graph.AddInput<float>(new TensorShape(1, 2), "input");
            return graph.Compile(input * Functional.Constant(new TensorShape(1, 2), new[] { 0.1f, -0.25f }));
        }
        string AddText(object writer)
        {
            var model = (string)Call(writer, "StoreModel", Tiny());
            var tokenizer = Path.Combine(directory, "tokenizer.json"); File.WriteAllText(tokenizer, "{}");
            var json = (string)Call(writer, "StoreFile", tokenizer);
            Call(writer, "CommitModule", "text", new JObject { ["kind"] = "text", ["model"] = model, ["tokenizer"] = json });
            return model;
        }
        [Test] public void ModelSerializationPreservesDirectWriterBytesAndReusesContent()
        {
            var writer = Create(); var model = Tiny(); var reference = Path.Combine(directory, "reference.sentis"); ModelWriter.Save(reference, model);
            var first = (string)Call(writer, "StoreModel", model); var second = (string)Call(writer, "StoreModel", model);
            Assert.That(first, Is.EqualTo(second)); Assert.That(File.ReadAllBytes(Path.Combine(output, first)), Is.EqualTo(File.ReadAllBytes(reference)));
        }
        [Test] public void OriginalFloatCoefficientBitsArePreserved()
        {
            var writer = Create(); var values = new[] { 0.1f, -0.25f, BitConverter.Int32BitsToSingle(unchecked((int)0x80000000)) };
            var file = (string)Call(writer, "StoreWeights", values);
            var expected = new byte[values.Length * 4]; Buffer.BlockCopy(values, 0, expected, 0, expected.Length);
            Assert.That(File.ReadAllBytes(Path.Combine(output, file)), Is.EqualTo(expected));
        }
        [Test] public void PublishedModulesRecordPreparationWithoutClaimingInference()
        {
            var writer = Create(); var model = AddText(writer); var bundle = GpuMediaBundle.Open(output);
            Assert.That(bundle.VerifyFile(model), Is.EqualTo(Path.Combine(output, model)));
            var data = JObject.Parse(File.ReadAllText(Path.Combine(output, "bundle.json")));
            Assert.That((bool)data["inference_executed"], Is.False); Assert.That((string)data["source_commit"], Is.EqualTo(new string('a', 40)));
        }
        [Test] public void ForeignOutputDirectoryIsNeverOverwritten()
        {
            Directory.CreateDirectory(output); var foreign = Path.Combine(output, "keep.txt"); File.WriteAllText(foreign, "keep");
            Assert.That(Assert.Throws<TargetInvocationException>(() => Create()).InnerException, Is.InstanceOf<IOException>());
            Assert.That(File.ReadAllText(foreign), Is.EqualTo("keep"));
        }
        [Test] public void DifferentProducerCannotResumeTheSamePreparation()
        {
            Create(); Assert.That(Assert.Throws<TargetInvocationException>(() => Create(new string('b', 40))).InnerException, Is.InstanceOf<InvalidDataException>());
        }
        [Test] public void InvalidRecipeCannotReplacePreviouslyPublishedModuleMetadata()
        {
            var writer = Create(); AddText(writer); var path = Path.Combine(output, "bundle.json"); var previous = File.ReadAllBytes(path);
            Assert.That(Assert.Throws<TargetInvocationException>(() => Call(writer, "CommitModule", "bad", new JObject { ["kind"] = "cpu" })).InnerException, Is.InstanceOf<InvalidDataException>());
            Assert.That(File.ReadAllBytes(path), Is.EqualTo(previous));
        }
    }
}
