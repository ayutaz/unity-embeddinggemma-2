using System;
using System.Collections;
using System.Diagnostics;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.TestTools;

namespace EmbeddingGemma.Tests
{
    public sealed class GpuProjectionPipelineAsyncTests
    {
        sealed class Operation : IGpuTensorOperation
        {
            readonly GpuBlockedMatmul identity = new GpuBlockedMatmul(2, 2, new[] { 1f, 0f, 0f, 1f });
            public bool Fail;
            public int Calls;
            public Tensor<float> LastOutput;
            public BackendType Backend => BackendType.GPUCompute;
            public Tensor<float> Execute(Tensor<float> input)
            {
                Calls++; if (Fail) throw new InvalidOperationException("Controlled asynchronous operation failure.");
                return LastOutput = identity.Execute(input);
            }
            public void Dispose() => identity.Dispose();
        }
        static Model Rename(Model model, params string[] names)
        {
            for (var i = 0; i < names.Length; i++) model.outputs[i] = new Model.Output { index = model.outputs[i].index, name = names[i] };
            return model;
        }
        static GpuProjectionPipeline Pipeline(Operation operation,bool release=true,bool nativeGemm=false)
        {
            Assert.That(SystemInfo.supportsGraphicsFence, Is.True, "Actual graphics-fence support required; no skip.");
            var first = new FunctionalGraph(); var x = first.AddInput<float>(new TensorShape(1, 2), "input");
            var work = nativeGemm ? x * x + Functional.MatMul(x, Functional.Constant(new TensorShape(2, 2), new[] { 1f, 0f, 0f, 1f })) : x * x + x;
            var before = Rename(first.Compile(x, work), "input", "work");
            if (nativeGemm) Assert.That(before.layers.Exists(layer => layer.GetType().Name == "MatMul" || layer.GetType().Name == "MatMul2D"), Is.True, "Exercise a real ML-device GEMM operator cache.");
            var last = new FunctionalGraph(); var residual = last.AddInput<float>(new TensorShape(1, 2), "input"); var projected = last.AddInput<float>(new TensorShape(1, 2), "projected");
            var after = Rename(last.Compile(projected * projected + residual), "final");
            return new GpuProjectionPipeline(new[] { before, after }, new[] { new[] { "work" } }, new[] { "projected" }, new IGpuTensorOperation[] { operation }, release);
        }
        static Task<Tensor<float>> Start(GpuProjectionPipeline pipeline, Tensor<float> input, CancellationToken token = default)
        {
            var method = typeof(GpuProjectionPipeline).GetMethod("ExecuteAsync", new[] { typeof(Tensor<float>), typeof(CancellationToken) });
            Assert.That(method, Is.Not.Null, "Staged GPU-fence execution is missing.");
            return (Task<Tensor<float>>)method.Invoke(pipeline, new object[] { input, token });
        }
        static IEnumerator Wait(Task task)
        {
            var timer = Stopwatch.StartNew();
            while (!task.IsCompleted) { Assert.That(timer.Elapsed.TotalSeconds, Is.LessThan(10), "Small GPU scheduling test did not complete."); yield return null; }
        }
        static Tensor<float> Input()
        {
            var input = new Tensor<float>(new TensorShape(1, 2), new[] { -3f, 4f }); ComputeTensorData.Pin(input); return input;
        }

        [UnityTest]
        public IEnumerator AsyncOptInRecreatesCompiledOperatorContextOnlyAfterGpuCompletion()
        {
            Assert.That(SystemInfo.supportsMachineLearning, Is.True, "Actual ML-device GPU required for this context lifetime gate; no skip.");
            using var operation = new Operation(); using var input = Input();
            const BindingFlags fields = BindingFlags.Instance | BindingFlags.NonPublic;
            var type = typeof(Worker).Assembly.GetType("Unity.InferenceEngine.GfxDeviceBackend");
            var contextField = type.GetField("m_GfxContext", fields); Assert.That(contextField, Is.Not.Null);
            foreach (var release in new[] { false, true })
            {
                using var pipeline = Pipeline(operation, release, true);
                var workers = (Worker[])typeof(GpuProjectionPipeline).GetField("workers", fields).GetValue(pipeline);
                var backend = typeof(Worker).GetField("m_Backend", fields).GetValue(workers[0]); Assert.That(type.IsInstanceOfType(backend), Is.True);
                var before = contextField.GetValue(backend); var pending = Start(pipeline, input);
                Assert.That(contextField.GetValue(backend), Is.SameAs(before), "A queued GPU stage still owns its compiled operators.");
                yield return Wait(pending); using var result = pending.GetAwaiter().GetResult();
                if (release) Assert.That(contextField.GetValue(backend), Is.Not.SameAs(before), "Completed compiled-operator caches must be released when opted in.");
                else Assert.That(contextField.GetValue(backend), Is.SameAs(before));
                var repeat = Start(pipeline, input); yield return Wait(repeat); using var repeated = repeat.GetAwaiter().GetResult();
                Assert.That(result.DownloadToArray(), Is.EqualTo(new[] { 33f, 404f })); Assert.That(repeated.DownloadToArray(), Is.EqualTo(new[] { 33f, 404f }));
            }
            Assert.That(input.DownloadToArray(), Is.EqualTo(new[] { -3f, 4f }));
        }

        [UnityTest]
        public IEnumerator AsyncKeepsSynchronousBitsBorrowedInputsAndIndependentOwnedResults()
        {
            using var operation = new Operation(); using var pipeline = Pipeline(operation); using var input = Input(); using var baseline = pipeline.Execute(input);
            var pending = Start(pipeline, input); Assert.That(pending.IsCompleted, Is.False, "Give the Unity loop control between GPU stages.");
            yield return Wait(pending); using var output = pending.GetAwaiter().GetResult();
            var repeated = Start(pipeline, input); yield return Wait(repeated); using var other = repeated.GetAwaiter().GetResult();
            pipeline.Dispose(); pipeline.Dispose(); Assert.That(output.backendType, Is.EqualTo(BackendType.GPUCompute));
            Assert.That(output.DownloadToArray(), Is.EqualTo(baseline.DownloadToArray())); Assert.That(other.DownloadToArray(), Is.EqualTo(new[] { 33f, 404f }));
            Assert.That(input.DownloadToArray(), Is.EqualTo(new[] { -3f, 4f })); using var borrowed = operation.Execute(input); Assert.That(borrowed.DownloadToArray(), Is.EqualTo(new[] { -3f, 4f }));
        }

        [UnityTest]
        public IEnumerator CancellationReleasesOwnedIntermediatesAndAllowsReuse()
        {
            using var operation = new Operation(); using var pipeline = Pipeline(operation); using var input = Input();
            using var before = new CancellationTokenSource(); before.Cancel(); var rejected = Start(pipeline, input, before.Token);
            Assert.Throws<OperationCanceledException>(() => rejected.GetAwaiter().GetResult()); Assert.That(operation.Calls, Is.Zero);
            using var during = new CancellationTokenSource(); var pending = Start(pipeline, input, during.Token); var intermediate = operation.LastOutput;
            Assert.That(intermediate, Is.Not.Null); during.Cancel(); yield return Wait(pending);
            Assert.Throws<OperationCanceledException>(() => pending.GetAwaiter().GetResult()); Assert.That(intermediate.dataOnBackend, Is.Null);
            var reuse = Start(pipeline, input); yield return Wait(reuse); using var result = reuse.GetAwaiter().GetResult();
            Assert.That(result.DownloadToArray(), Is.EqualTo(new[] { 33f, 404f })); Assert.That(input.DownloadToArray(), Is.EqualTo(new[] { -3f, 4f }));
        }

        [UnityTest]
        public IEnumerator OverlapIsRejectedAndDisposalAbortsPendingCallWithoutOwningInput()
        {
            using var operation = new Operation(); using var pipeline = Pipeline(operation); using var input = Input(); var pending = Start(pipeline, input);
            Assert.Throws<InvalidOperationException>(() => pipeline.Execute(input)); var overlap = Start(pipeline, input);
            Assert.Throws<InvalidOperationException>(() => overlap.GetAwaiter().GetResult()); pipeline.Dispose(); yield return Wait(pending);
            Assert.Throws<ObjectDisposedException>(() => pending.GetAwaiter().GetResult()); pipeline.Dispose();
            Assert.That(input.DownloadToArray(), Is.EqualTo(new[] { -3f, 4f })); Assert.That(operation.LastOutput.dataOnBackend, Is.Null);
            using var borrowed = operation.Execute(input); Assert.That(borrowed.DownloadToArray(), Is.EqualTo(new[] { -3f, 4f }));
        }

        [UnityTest]
        public IEnumerator OperationFailureDoesNotPoisonLaterAsyncExecution()
        {
            using var operation = new Operation { Fail = true }; using var pipeline = Pipeline(operation); using var input = Input(); var failed = Start(pipeline, input);
            yield return Wait(failed); Assert.Throws<InvalidOperationException>(() => failed.GetAwaiter().GetResult()); operation.Fail = false;
            var pending = Start(pipeline, input); yield return Wait(pending); using var output = pending.GetAwaiter().GetResult(); pipeline.Dispose();
            Assert.That(output.DownloadToArray(), Is.EqualTo(new[] { 33f, 404f })); Assert.That(input.DownloadToArray(), Is.EqualTo(new[] { -3f, 4f }));
        }
    }
}
