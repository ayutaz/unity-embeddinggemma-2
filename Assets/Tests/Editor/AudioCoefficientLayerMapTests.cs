using System;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.ExceptionServices;
using System.Security.Cryptography;
using System.Collections.Generic;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using Unity.InferenceEngine;

namespace EmbeddingGemma.Tests
{
    public sealed class AudioCoefficientLayerMapTests
    {
        static Dictionary<string, int> Resolve(Model model, JArray anchors)
        {
            var type = AppDomain.CurrentDomain.GetAssemblies().Select(assembly => assembly.GetType("EmbeddingGemma.Tests.AudioCoefficientLayerMap")).FirstOrDefault(value => value != null);
            Assert.That(type, Is.Not.Null, "Coefficient-to-layer mapping is missing.");
            try { return (Dictionary<string, int>)type.GetMethod("Resolve").Invoke(null, new object[] { model, anchors }); }
            catch (TargetInvocationException error) { ExceptionDispatchInfo.Capture(error.InnerException).Throw(); throw; }
        }
        static JArray Anchors()
        {
            var bytes = new byte[8]; Buffer.BlockCopy(new[] { 1.5f, 2.25f }, 0, bytes, 0, bytes.Length);
            using var hash = SHA256.Create();
            return new JArray(new JObject { ["stage"] = "layer_0", ["coefficient_sha256"] = BitConverter.ToString(hash.ComputeHash(bytes)).Replace("-", "").ToLowerInvariant(),
                ["coefficient_bytes"] = 8, ["coefficient_dims"] = new JArray(2) });
        }
        static Model Fixture(bool duplicate)
        {
            var graph = new FunctionalGraph(); var x = graph.AddInput<float>(new TensorShape(1, 2), "input");
            var scale = Functional.Constant(new TensorShape(2), new[] { 1.5f, 2.25f });
            var first = x * scale;
            return graph.Compile(duplicate ? first + (x + 1f) * scale : first);
        }
        [Test] public void ExactCoefficientIdentityFindsOneOutputWithoutMutatingModel()
        {
            var model = Fixture(false); using var before = new MemoryStream(); ModelWriter.Save(before, model);
            var result = Resolve(model, Anchors());
            Assert.That(result["layer_0"], Is.EqualTo(model.outputs[0].index));
            using var after = new MemoryStream(); ModelWriter.Save(after, model);
            Assert.That(after.ToArray(), Is.EqualTo(before.ToArray()));
        }
        [Test] public void MissingCoefficientCannotProduceAMapping()
        {
            var anchors = Anchors(); anchors[0]["coefficient_sha256"] = new string('0', 64);
            Assert.Throws<InvalidOperationException>(() => Resolve(Fixture(false), anchors));
        }
        [Test] public void SharedCoefficientWithTwoConsumersCannotChooseOneArbitrarily()
        { Assert.Throws<InvalidOperationException>(() => Resolve(Fixture(true), Anchors())); }
        [Test] public void DuplicateStageNameCannotBeAccepted()
        {
            var anchors = Anchors(); anchors.Add(anchors[0].DeepClone());
            Assert.Throws<InvalidOperationException>(() => Resolve(Fixture(false), anchors));
        }
        static Dictionary<string, int> NormInputs(Model model)
        {
            var type = AppDomain.CurrentDomain.GetAssemblies().Select(assembly => assembly.GetType("EmbeddingGemma.Tests.AudioCoefficientLayerMap")).FirstOrDefault(value => value != null);
            var method = type?.GetMethod("ResolveNormInputs");
            Assert.That(method, Is.Not.Null, "Coefficient-bound norm input mapping is missing.");
            try { return (Dictionary<string, int>)method.Invoke(null, new object[] { model, Anchors() }); }
            catch (TargetInvocationException error) { ExceptionDispatchInfo.Capture(error.InnerException).Throw(); throw; }
        }
        [Test] public void NormInputIdentityFollowsSquaredReductionWithoutChangingModel()
        {
            var graph = new FunctionalGraph(); var x = graph.AddInput<float>(new TensorShape(1, 2), "input");
            var scale = Functional.Constant(new TensorShape(2), new[] { 1.5f, 2.25f });
            var inverse = Functional.Pow(Functional.ReduceMean(Functional.Pow(x, 2f), -1, true) + 1e-5f, -0.5f);
            var model = graph.Compile(x * inverse * scale);
            using var before = new MemoryStream(); ModelWriter.Save(before, model);
            Assert.That(NormInputs(model)["layer_0"], Is.EqualTo(model.inputs[0].index));
            using var after = new MemoryStream(); ModelWriter.Save(after, model);
            Assert.That(after.ToArray(), Is.EqualTo(before.ToArray()));
        }
        [Test] public void NormInputMappingRejectsPlainCoefficientMultiplication()
        { Assert.Throws<InvalidOperationException>(() => NormInputs(Fixture(false))); }
        [Test] public void NormInputMappingRejectsInverseWithoutSquaredReduction()
        {
            var graph = new FunctionalGraph(); var x = graph.AddInput<float>(new TensorShape(1, 2), "input");
            var scale = Functional.Constant(new TensorShape(2), new[] { 1.5f, 2.25f });
            var model = graph.Compile(x * Functional.Pow(x + 1e-5f, -0.5f) * scale);
            Assert.Throws<InvalidOperationException>(() => NormInputs(model));
        }
    }
}
