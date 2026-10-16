using System;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.ExceptionServices;
using System.Security.Cryptography;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEngine;

namespace EmbeddingGemma.Tests
{
    public sealed class AudioNormIsolationTests
    {
        static Model Create(Model source, JArray anchors, TensorShape shape)
        {
            var type = AppDomain.CurrentDomain.GetAssemblies().Select(assembly => assembly.GetType("EmbeddingGemma.Tests.AudioNormIsolation")).FirstOrDefault(value => value != null);
            Assert.That(type, Is.Not.Null, "Coefficient-bound norm isolation is missing.");
            try { return (Model)type.GetMethod("Create").Invoke(null, new object[] { source, anchors, shape }); }
            catch (TargetInvocationException error) { ExceptionDispatchInfo.Capture(error.InnerException).Throw(); throw; }
        }
        static JArray Anchors()
        {
            var bytes = new byte[8]; Buffer.BlockCopy(new[] { 1.5f, 2.25f }, 0, bytes, 0, bytes.Length);
            using var hash = SHA256.Create();
            return new JArray(new JObject { ["stage"] = "post_layer_norm", ["coefficient_bytes"] = 8,
                ["coefficient_sha256"] = BitConverter.ToString(hash.ComputeHash(bytes)).Replace("-", "").ToLowerInvariant(), ["coefficient_dims"] = new JArray(2) });
        }
        static Model Fixture()
        {
            var graph = new FunctionalGraph(); var x = graph.AddInput<float>(new TensorShape(1, 2), "original_input");
            var before = Functional.Swish(x * Functional.Constant(new TensorShape(2), new[] { 4.5f, 5.75f }));
            var scale = Functional.Constant(new TensorShape(2), new[] { 1.5f, 2.25f });
            var output = before * Functional.Pow(Functional.ReduceMean(Functional.Pow(before, 2f), -1, true) + 1e-5f, -0.5f) * scale;
            return graph.Compile(output);
        }
        [Test] public void BorrowedSliceContainsOnlyNormAndReferencedConstantsPreservingSourceBytes()
        {
            var source = Fixture(); using var before = new MemoryStream(); ModelWriter.Save(before, source);
            var slice = Create(source, Anchors(), new TensorShape(1, 2));
            Assert.That(slice.inputs.Count, Is.EqualTo(1)); Assert.That(slice.outputs.Count, Is.EqualTo(1));
            Assert.That(slice.inputs[0].index, Is.EqualTo(0));
            Assert.That(slice.layers.Any(layer => layer.GetType().Name == "Swish"), Is.False);
            Assert.That(slice.layers.Count, Is.EqualTo(6)); Assert.That(slice.constants.Count, Is.LessThan(source.constants.Count));
            Assert.That(slice.constants.All(value => source.constants.Contains(value)), Is.True);
            Assert.That(slice.layers.Count(value => !source.layers.Contains(value)), Is.EqualTo(2));
            Assert.That(slice.layers.Where(value => !source.layers.Contains(value)).All(value => value.inputs.Contains(0)), Is.True);
            using var after = new MemoryStream(); ModelWriter.Save(after, source); Assert.That(after.ToArray(), Is.EqualTo(before.ToArray()));
        }
        [TestCase(BackendType.CPU)] [TestCase(BackendType.GPUCompute)]
        public void ExactInputNormNumericsAreIndependentOfTheOriginalUpstream(BackendType backend)
        {
            var slice = Create(Fixture(), Anchors(), new TensorShape(1, 2));
            using var input = new Tensor<float>(new TensorShape(1, 2), new[] { 3f, 4f });
            if (backend == BackendType.GPUCompute) { Assert.That(SystemInfo.supportsComputeShaders, Is.True); ComputeTensorData.Pin(input); }
            using var worker = new Worker(slice, backend); worker.Schedule(input);
            var output = (Tensor<float>)worker.PeekOutput(); Assert.That(output.backendType, Is.EqualTo(backend));
            var actual = output.DownloadToArray(); var inverse = 1.0 / Math.Sqrt(12.5 + 1e-5);
            Assert.That(actual[0], Is.EqualTo(3 * inverse * 1.5).Within(1e-6));
            Assert.That(actual[1], Is.EqualTo(4 * inverse * 2.25).Within(1e-6));
        }
        [Test] public void MissingNormIdentityCannotProduceASlice()
        {
            var anchors = Anchors(); anchors[0]["coefficient_sha256"] = new string('0', 64);
            Assert.Throws<InvalidOperationException>(() => Create(Fixture(), anchors, new TensorShape(1, 2)));
        }
        [Test] public void EmptyInputShapeCannotProduceASlice()
        { Assert.Throws<ArgumentException>(() => Create(Fixture(), Anchors(), new TensorShape(1, 0))); }
    }
}
