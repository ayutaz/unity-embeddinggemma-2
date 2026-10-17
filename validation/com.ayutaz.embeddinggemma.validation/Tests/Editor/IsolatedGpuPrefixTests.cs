using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using NUnit.Framework;
using Newtonsoft.Json.Linq;
using Unity.InferenceEngine;
using UnityEngine.TestTools;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class IsolatedGpuPrefixTests
    {
        static Model Segment(string input)
        {
            var graph = new FunctionalGraph(); var x = graph.AddInput<float>(new TensorShape(1, 2), input);
            return graph.Compile(x + 1f);
        }
        [UnityTest] public IEnumerator OriginalCacheRetirementReleasesDeadBoundariesWithoutChangingTheResult()
        {
            var models = new[] { Segment("input"), Segment("next") };
            using var first = new GpuScalarDivide(2); using var second = new GpuScalarDivide(2);
            using var input = new Tensor<float>(new TensorShape(1, 2), new[] { 1f, 2f }); ComputeTensorData.Pin(input);
            var rows = new List<JObject>();
            var task = IsolatedGpuPrefix.ExecuteAsync(models,
                models.Select(m => new[] { m.outputs[0].name }).ToArray(), new[] { "next", "final" },
                new IGpuTensorOperation[] { first, second }, input, rows.Add, retireCompletedCache: true);
            while (!task.IsCompleted) { UnityEditor.EditorApplication.QueuePlayerLoopUpdate(); yield return null; }
            using var result = task.GetAwaiter().GetResult();
            Assert.That(result.DownloadToArray(), Is.EqualTo(new[] { 1f, 1.25f }));
            Assert.That(input.DownloadToArray(), Is.EqualTo(new[] { 1f, 2f }));
            Assert.That(rows.Count(r => (string)r["event"] == "original_worker_cache_retired"), Is.EqualTo(2));
            Assert.That(rows.Count(r => (string)r["event"] == "original_backend_cache_retired"), Is.EqualTo(2));
            Assert.That(rows.Where(r => (string)r["event"] == "dead_boundaries_retired").Sum(r => (int)r["retired_count"]), Is.GreaterThan(0));
        }
        [UnityTest] public IEnumerator EightSmallOriginalPairsCanBeIsolatedWithoutLaterModels()
        {
            var models = Enumerable.Range(0, 8).Select(i => Segment(i == 0 ? "input" : "next" + (i - 1))).ToArray();
            var operations = Enumerable.Range(0, 8).Select(_ => new GpuScalarDivide(2)).ToArray();
            using var input = new Tensor<float>(new TensorShape(1, 2), new[] { 1f, 2f }); ComputeTensorData.Pin(input);
            try
            {
                var task = IsolatedGpuPrefix.ExecuteAsync(models, models.Select(m => new[] { m.outputs[0].name }).ToArray(),
                    Enumerable.Range(0, 8).Select(i => "next" + i).ToArray(), operations, input, _ => { });
                while (!task.IsCompleted) { UnityEditor.EditorApplication.QueuePlayerLoopUpdate(); yield return null; }
                using var result = task.GetAwaiter().GetResult();
                Assert.That(result.DownloadToArray(), Is.EqualTo(new[] { 1f, 1f + 1f / 256 }));
            }
            finally { foreach (var operation in operations) operation.Dispose(); }
        }
        [UnityTest] public IEnumerator OriginalGpuSegmentsAndOperationsReturnOwnedOutputAfterEveryFence()
        {
            var models = new[] { Segment("input"), Segment("next") };
            using var first = new GpuScalarDivide(2); using var second = new GpuScalarDivide(2);
            using var input = new Tensor<float>(new TensorShape(1, 2), new[] { 1f, 2f }); ComputeTensorData.Pin(input);
            var rows = new List<JObject>();
            var task = IsolatedGpuPrefix.ExecuteAsync(models,
                new[] { new[] { models[0].outputs[0].name }, new[] { models[1].outputs[0].name } },
                new[] { "next", "final" }, new IGpuTensorOperation[] { first, second }, input, rows.Add);
            while (!task.IsCompleted) { UnityEditor.EditorApplication.QueuePlayerLoopUpdate(); yield return null; }
            using var result = task.GetAwaiter().GetResult();
            Assert.That(result.backendType, Is.EqualTo(BackendType.GPUCompute));
            Assert.That(result.DownloadToArray(), Is.EqualTo(new[] { 1f, 1.25f }));
            Assert.That(input.DownloadToArray(), Is.EqualTo(new[] { 1f, 2f }));
            Assert.That(rows.Where(r => (string)r["event"] == "fence_passed").Select(r => (int)r["stage"]), Is.EqualTo(new[] { 0, 1 }));
            Assert.That(rows.All(r => !(bool)r["full_model_qualified"]), Is.True);
        }
        [UnityTest] public IEnumerator CancellationWaitsForSubmittedGpuWorkAndKeepsBorrowedInputAlive()
        {
            var model = Segment("input"); using var operation = new GpuScalarDivide(2);
            using var input = new Tensor<float>(new TensorShape(1, 2), new[] { 1f, 2f }); ComputeTensorData.Pin(input);
            using var cancellation = new CancellationTokenSource(); var rows = new List<JObject>();
            var task = IsolatedGpuPrefix.ExecuteAsync(new[] { model }, new[] { new[] { model.outputs[0].name } },
                new[] { "final" }, new IGpuTensorOperation[] { operation }, input, row =>
                { rows.Add(row); if ((string)row["event"] == "operation_submitted") cancellation.Cancel(); }, cancellation.Token);
            while (!task.IsCompleted) { UnityEditor.EditorApplication.QueuePlayerLoopUpdate(); yield return null; }
            Assert.Throws<OperationCanceledException>(() => task.GetAwaiter().GetResult());
            Assert.That(rows.Any(r => (string)r["event"] == "fence_passed"), Is.True);
            Assert.That(input.DownloadToArray(), Is.EqualTo(new[] { 1f, 2f }));
        }
    }
}
