using System;
using System.IO;
using System.Linq;
using NUnit.Framework;
using Unity.InferenceEngine;

namespace EmbeddingGemma.Tests
{
    public sealed class NativeInversePowerCandidateTests
    {
        [TestCase(BackendType.CPU)] [TestCase(BackendType.GPUCompute)]
        public void CandidateKeepsNativePowerWithoutMutatingBorrowedModel(BackendType backend)
        {
            var type = AppDomain.CurrentDomain.GetAssemblies().Select(assembly => assembly.GetType("EmbeddingGemma.Tests.NativeInversePowerCandidate"))
                .FirstOrDefault(value => value != null);
            Assert.That(type, Is.Not.Null, "Native inverse-power candidate is missing.");
            var shape = new TensorShape(1, 3);
            var graph = new FunctionalGraph(); var input = graph.AddInput<float>(shape, "input");
            var source = graph.Compile(Functional.Pow(input, -0.5f));
            Assert.That(source.layers.Count(layer => layer.GetType().Name == "Rsqrt"), Is.EqualTo(1));
            byte[] Bytes(Model model) { using var stream = new MemoryStream(); ModelWriter.Save(stream, model); return stream.ToArray(); }
            var before = Bytes(source);
            var candidate = (Model)type.GetMethod("Create").Invoke(null, new object[] { source });
            Assert.That(candidate, Is.Not.SameAs(source));
            Assert.That(candidate.layers.Count(layer => layer.GetType().Name == "Pow"), Is.EqualTo(1));
            Assert.That(candidate.layers.Any(layer => layer.GetType().Name == "Rsqrt"), Is.False);
            Assert.That(Bytes(source), Is.EqualTo(before), "Borrowed model bytes and weights must stay unchanged.");
            using var tensor = new Tensor<float>(shape, new[] { 0.25f, 1f, 4f });
            using var worker = new Worker(candidate, backend);
            worker.Schedule(tensor);
            var output = (Tensor<float>)worker.PeekOutput();
            Assert.That(output.backendType, Is.EqualTo(backend));
            Assert.That(output.DownloadToArray(), Is.EqualTo(new[] { 2f, 1f, 0.5f }));
        }
    }
}
