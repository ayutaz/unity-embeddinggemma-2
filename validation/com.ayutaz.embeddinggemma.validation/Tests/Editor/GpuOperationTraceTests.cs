using System;
using System.Collections.Generic;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using Unity.InferenceEngine;

namespace EmbeddingGemma.Validation.Tests
{
    /// <summary>Transparent diagnostic contracts with fake operations, not GPU acceptance.</summary>
    public sealed class GpuOperationTraceTests
    {
        sealed class Unary : IGpuTensorOperation
        {
            public bool Disposed; public int Calls;
            public BackendType Backend => BackendType.GPUCompute;
            public Tensor<float> Execute(Tensor<float> input) { Calls++; return input; }
            public void Dispose() { Disposed = true; }
        }
        sealed class Binary : IGpuBinaryTensorOperation
        {
            public readonly InvalidOperationException Failure = new InvalidOperationException("controlled failure");
            public BackendType Backend => BackendType.GPUCompute;
            public Tensor<float> Execute(Tensor<float> input) => throw new NotSupportedException();
            public Tensor<float> Execute(Tensor<float> left, Tensor<float> right) => throw Failure;
            public void Dispose() { }
        }
        [Test]
        public void UnaryTracePreservesArityBorrowedResourcesAndExactReturnWithoutReadingValues()
        {
            using var source = new Unary(); var events = new List<JObject>();
            using var input = new Tensor<float>(new TensorShape(1, 2), new[] {12345f, -54321f});
            var traced = GpuOperationTrace.Wrap(source, 7, events.Add);
            Assert.That(traced, Is.Not.InstanceOf<IGpuBinaryTensorOperation>());
            Assert.That(traced.Backend, Is.EqualTo(source.Backend));
            Assert.That(traced.Execute(input), Is.SameAs(input)); traced.Dispose();
            Assert.That(source.Disposed, Is.False); Assert.That(source.Calls, Is.EqualTo(1));
            Assert.That(input.dataOnBackend, Is.Not.Null);
            Assert.That(events.Count, Is.EqualTo(2)); Assert.That(events[0].Value<string>("event"), Is.EqualTo("begin"));
            Assert.That(events[1].Value<string>("event"), Is.EqualTo("submitted"));
            Assert.That(events[1].Value<int>("operation_index"), Is.EqualTo(7));
            Assert.That(events[1].Value<long>("cpu_wall_ns"), Is.GreaterThanOrEqualTo(0));
            foreach (var row in events) Assert.That(row.Value<bool>("gpu_completion_verified"), Is.False);
            var json = JArray.FromObject(events).ToString();
            Assert.That(json, Does.Not.Contain("12345").And.Not.Contain("54321"));
            Assert.That(json, Does.Not.Contain("gpu_elapsed_ns"), "A CPU submission trace cannot claim GPU time.");
        }
        [Test]
        public void BinaryTraceKeepsOriginalExceptionAndDoesNotMisreportSubmission()
        {
            using var source = new Binary(); var events = new List<JObject>();
            using var input = new Tensor<float>(new TensorShape(1, 1), new[] {1f});
            using var traced = GpuOperationTrace.Wrap(source, 2, events.Add);
            Assert.That(traced, Is.InstanceOf<IGpuBinaryTensorOperation>());
            Assert.That(Assert.Throws<InvalidOperationException>(() => ((IGpuBinaryTensorOperation)traced).Execute(input, input)), Is.SameAs(source.Failure));
            Assert.That(events.Count, Is.EqualTo(2)); Assert.That(events[1].Value<string>("event"), Is.EqualTo("exception"));
            Assert.That(events[1].Value<string>("error_type"), Is.EqualTo(typeof(InvalidOperationException).FullName));
            Assert.That(events[1].Value<bool>("gpu_completion_verified"), Is.False);
        }
        [Test] public void MissingOperationIsRejected()
        { Assert.Throws<ArgumentNullException>(() => GpuOperationTrace.Wrap(null, 0, _ => {})); }
        [Test] public void MissingObserverIsRejected()
        { using var source = new Unary(); Assert.Throws<ArgumentNullException>(() => GpuOperationTrace.Wrap(source, 0, null)); }
        [Test] public void NegativeOperationIndexIsRejected()
        { using var source = new Unary(); Assert.Throws<ArgumentOutOfRangeException>(() => GpuOperationTrace.Wrap(source, -1, _ => {})); }
    }
}
