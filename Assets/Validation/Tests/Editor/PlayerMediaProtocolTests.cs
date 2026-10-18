using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using UnityEngine.TestTools;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class PlayerMediaProtocolTests
    {
        sealed class Module : IPlayerMediaModule
        {
            public readonly List<string> Encoded = new();
            public readonly List<string> Events = new();
            public bool FailAudit, Corrupt, FailClose, IgnoreCancellation;
            public bool IsReleased { get; private set; }
            public async Task<float[]> EmbedAsync(PlayerMediaCase row, CancellationToken token)
            {
                Events.Add(token.IsCancellationRequested ? "pre-cancel" : "embed");
                await Task.Yield();
                if (!IgnoreCancellation) token.ThrowIfCancellationRequested();
                Encoded.Add(row.Id);
                var vector = (float[])row.Expected.Clone();
                if (Corrupt) { vector[0] = 0; vector[1] = 1; }
                return vector;
            }
            public JObject InspectGpuExecution()
            {
                Events.Add("audit");
                if (FailAudit) throw new InvalidOperationException("Synthetic GPU audit rejection.");
                return new JObject { ["success"] = true, ["worker_count"] = 2 };
            }
            public Task CloseAsync()
            {
                Events.Add("close");
                if (FailClose) throw new InvalidOperationException("Synthetic release rejection.");
                IsReleased = true; return Task.CompletedTask;
            }
        }
        static PlayerMediaCase[] Cases() => new[] { "first", "second" }.Select(id =>
        {
            var vector = new float[768]; vector[0] = 1;
            return new PlayerMediaCase { Id = id, Expected = vector };
        }).ToArray();
        static IEnumerator Await(Task task)
        { while (!task.IsCompleted) yield return null; task.GetAwaiter().GetResult(); }

        [UnityTest] public IEnumerator CancellationRecoveryEveryCaseWarmRepetitionsAndReleaseAreRequired()
        {
            var module = new Module(); var snapshots = new List<JObject>();
            var task = PlayerMediaProtocol.Run(Cases(), "fp32", () => Task.FromResult<IPlayerMediaModule>(module),
                value => snapshots.Add((JObject)value.DeepClone()));
            yield return Await(task);
            var result = task.Result;
            Assert.That((bool)result["success"], Is.True);
            Assert.That((bool)result["gpu_verified"], Is.False, "Injected execution must never certify actual GPU execution.");
            Assert.That((string)result["validation_mode"], Is.EqualTo("injected_contract"));
            Assert.That((bool)result["full_multimodal_acceptance"], Is.False);
            Assert.That((bool)result["pre_cancel_verified"], Is.True);
            Assert.That((bool)result["pending_call_cancel_verified"], Is.True);
            Assert.That((bool)result["session_released"], Is.True);
            Assert.That(module.Encoded, Is.EqualTo(new[] { "first", "second", "first", "first", "first" }));
            Assert.That(((JArray)result["cases"]).Count, Is.EqualTo(2));
            Assert.That(((JArray)result["warm"]).Count, Is.EqualTo(3));
            Assert.That(((JArray)result["cases"])[0]["embedding"].Count(), Is.EqualTo(768));
            Assert.That(module.Events.First(), Is.EqualTo("audit"));
            Assert.That(module.Events.Last(), Is.EqualTo("close"));
            Assert.That(snapshots.Any(row => (string)row["phase"] == "loading" && !(bool)row["success"]), Is.True);
        }
        [UnityTest] public IEnumerator FailedGpuAuditCannotProceedToInferenceAndStillCloses()
        {
            var module = new Module { FailAudit = true };
            var task = PlayerMediaProtocol.Run(Cases(), "fp32", () => Task.FromResult<IPlayerMediaModule>(module), _ => { });
            yield return Await(task);
            Assert.That((bool)task.Result["success"], Is.False);
            Assert.That(module.Encoded, Is.Empty);
            Assert.That(module.IsReleased, Is.True);
            Assert.That((string)task.Result["error_type"], Is.EqualTo("InvalidOperationException"));
        }
        [UnityTest] public IEnumerator IgnoredCancellationFailsWithoutBeingRelabeledAsRecovery()
        {
            var module = new Module { IgnoreCancellation = true };
            var task = PlayerMediaProtocol.Run(Cases(), "fp32", () => Task.FromResult<IPlayerMediaModule>(module), _ => { });
            yield return Await(task);
            Assert.That((bool)task.Result["success"], Is.False);
            Assert.That((bool)task.Result["pre_cancel_verified"], Is.False);
            Assert.That(module.IsReleased, Is.True);
        }
        [UnityTest] public IEnumerator InvalidVectorFailsAndKeepsPartialEvidence()
        {
            var module = new Module { Corrupt = true };
            var task = PlayerMediaProtocol.Run(Cases(), "fp32", () => Task.FromResult<IPlayerMediaModule>(module), _ => { });
            yield return Await(task);
            Assert.That((bool)task.Result["success"], Is.False);
            Assert.That((bool)task.Result["session_released"], Is.True);
            Assert.That((string)task.Result["failed_phase"], Is.EqualTo("case_first"));
            Assert.That(((JArray)task.Result["cases"]).Count, Is.EqualTo(1));
        }
        [UnityTest] public IEnumerator ReleaseFailureCannotBecomeSuccess()
        {
            var module = new Module { FailClose = true };
            var task = PlayerMediaProtocol.Run(Cases(), "float16", () => Task.FromResult<IPlayerMediaModule>(module), _ => { });
            yield return Await(task);
            Assert.That((bool)task.Result["success"], Is.False);
            Assert.That((bool)task.Result["session_released"], Is.False);
            Assert.That((string)task.Result["release_error_type"], Is.EqualTo("InvalidOperationException"));
        }
        [Test] public void MissingCasesAndUnrecognizedPrecisionFailBeforeLoading()
        {
            var loads = 0;
            Task<IPlayerMediaModule> Load() { loads++; return Task.FromResult<IPlayerMediaModule>(new Module()); }
            Assert.ThrowsAsync<ArgumentException>(async () => await PlayerMediaProtocol.Run(Array.Empty<PlayerMediaCase>(), "fp32", Load, _ => { }));
            Assert.ThrowsAsync<ArgumentException>(async () => await PlayerMediaProtocol.Run(Cases(), "bf16", Load, _ => { }));
            Assert.That(loads, Is.Zero);
        }
        [UnityTest] public IEnumerator FailureWritingTheClosingReceiptMustStillReleaseTheSession()
        {
            var module = new Module();
            var task = PlayerMediaProtocol.Run(Cases(), "fp32", () => Task.FromResult<IPlayerMediaModule>(module),
                report => { if ((string)report["phase"] == "closing") throw new IOException("Synthetic full output device."); });
            while (!task.IsCompleted) yield return null;
            Assert.That(module.IsReleased, Is.True, "A failed receipt write cannot retain GPU resources.");
            Assert.That(task.IsFaulted, Is.True);
            Assert.That(task.Exception.InnerException, Is.InstanceOf<IOException>());
        }
    }
}
