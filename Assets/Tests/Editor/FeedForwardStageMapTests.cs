using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.ExceptionServices;
using System.Security.Cryptography;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using Unity.InferenceEngine;

namespace EmbeddingGemma.Tests
{
    public sealed class FeedForwardStageMapTests
    {
        static Dictionary<string, int> Resolve(Model model)
        {
            var type = AppDomain.CurrentDomain.GetAssemblies().Select(assembly => assembly.GetType("EmbeddingGemma.Tests.AudioCoefficientLayerMap")).FirstOrDefault(value => value != null);
            var method = type?.GetMethod("ResolveFeedForward");
            Assert.That(method, Is.Not.Null, "Coefficient-bound feed-forward stage mapping is missing.");
            JArray anchors = new JArray();
            foreach (var item in new[] { ("pre_layer_norm", new[] { 1.5f, 2.25f }), ("post_layer_norm", new[] { 3.5f, 4.25f }) })
            {
                var bytes = new byte[8]; Buffer.BlockCopy(item.Item2, 0, bytes, 0, bytes.Length);
                using var hash = SHA256.Create();
                anchors.Add(new JObject { ["stage"] = item.Item1, ["coefficient_bytes"] = 8, ["coefficient_dims"] = new JArray(2),
                    ["coefficient_sha256"] = BitConverter.ToString(hash.ComputeHash(bytes)).Replace("-", "").ToLowerInvariant() });
            }
            try { return (Dictionary<string, int>)method.Invoke(null, new object[] { model, anchors }); }
            catch (TargetInvocationException error) { ExceptionDispatchInfo.Capture(error.InnerException).Throw(); throw; }
        }
        static Model Fixture(string problem = "none")
        {
            var graph = new FunctionalGraph(); var x = graph.AddInput<float>(new TensorShape(1, 2), "input");
            var preScale = Functional.Constant(new TensorShape(2), new[] { 1.5f, 2.25f });
            var postScale = Functional.Constant(new TensorShape(2), new[] { 3.5f, 4.25f });
            var normalized = problem == "plain_pre" ? x * preScale : x * Functional.Pow(Functional.ReduceMean(Functional.Pow(x, 2f), -1, true) + 1e-5f, -0.5f) * preScale;
            var first = Functional.MatMul(normalized, Functional.Constant(new TensorShape(2, 4), new[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f }));
            var activated = problem == "relu" ? Functional.Relu(first) : Functional.Swish(first);
            var second = Functional.MatMul(problem == "bypass" ? first : activated,
                Functional.Constant(new TensorShape(4, 2), new[] { 8f, 7f, 6f, 5f, 4f, 3f, 2f, 1f }));
            var post = second * Functional.Pow(Functional.ReduceMean(Functional.Pow(second, 2f), -1, true) + 1e-5f, -0.5f) * postScale;
            return graph.Compile(normalized, first, activated, second, post);
        }
        [Test] public void FiveStagesFollowBothNormIdentitiesAndPreserveModelBytes()
        {
            var model = Fixture(); using var before = new MemoryStream(); ModelWriter.Save(before, model);
            var result = Resolve(model);
            var names = new[] { "pre_layer_norm", "ffw_layer_1", "act_fn", "ffw_layer_2", "post_layer_norm" };
            for (var index = 0; index < names.Length; index++) Assert.That(result[names[index]], Is.EqualTo(model.outputs[index].index));
            using var after = new MemoryStream(); ModelWriter.Save(after, model); Assert.That(after.ToArray(), Is.EqualTo(before.ToArray()));
        }
        [TestCase("relu")] [TestCase("bypass")] [TestCase("plain_pre")]
        public void WrongActivationOrNormalizationCannotBeMappedAsOfficialFeedForward(string problem)
        { Assert.Throws<InvalidOperationException>(() => Resolve(Fixture(problem))); }
        static float[] Restore(float[] values)
        {
            var type = AppDomain.CurrentDomain.GetAssemblies().Select(assembly => assembly.GetType("EmbeddingGemma.Tests.AudioCoefficientLayerMap")).FirstOrDefault(value => value != null);
            var method = type?.GetMethod("RestoreHalfScale");
            Assert.That(method, Is.Not.Null, "Audited half-scale restoration is missing.");
            try { return (float[])method.Invoke(null, new object[] { values }); }
            catch (TargetInvocationException error) { ExceptionDispatchInfo.Capture(error.InnerException).Throw(); throw; }
        }
        [Test] public void HalfScaleRestoreReturnsExactIndependentValuesAndKeepsBorrowedInput()
        {
            var original = new[] { 1.5f, -2.25f, 0f, 1e-5f }; var scaled = original.Select(value => value * 0.5f).ToArray();
            var before = scaled.ToArray(); var restored = Restore(scaled);
            Assert.That(restored, Is.EqualTo(original)); Assert.That(restored, Is.Not.SameAs(scaled)); Assert.That(scaled, Is.EqualTo(before));
        }
        [TestCase(float.NaN)] [TestCase(float.PositiveInfinity)] [TestCase(float.Epsilon)] [TestCase(float.MaxValue)]
        public void HalfScaleRestoreRejectsNonfiniteSubnormalOrOverflowValues(float value)
        { Assert.Throws<ArgumentException>(() => Restore(new[] { value })); }
    }
}
