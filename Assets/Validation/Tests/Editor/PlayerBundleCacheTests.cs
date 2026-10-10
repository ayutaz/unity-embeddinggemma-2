using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using UnityEngine.TestTools;
using Unity.InferenceEngine;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class PlayerBundleCacheTests
    {
        string directory, source, cache;
        JObject receipt;
        readonly string[] names = { "model-fp32.sentis", "model-float16.sentis", "tokenizer.json", "reference.json", "search-reference.json" };
        const string Jar = "jar:file:///data/app/sample.apk!/assets/EmbeddingGemmaValidation";
        [SetUp] public void Setup()
        {
            directory = Path.Combine(Path.GetTempPath(), "embeddinggemma-cache-test-" + Guid.NewGuid().ToString("N"));
            source = Path.Combine(directory, "日本語 source"); cache = Path.Combine(directory, "cache"); Directory.CreateDirectory(source);
            File.WriteAllText(Path.Combine(source, "model-fp32.sentis"), "fp32"); File.WriteAllText(Path.Combine(source, "model-float16.sentis"), "float16");
            File.WriteAllText(Path.Combine(source, "tokenizer.json"), "{}"); var reference = PlayerValidationTests.Reference();
            File.WriteAllText(Path.Combine(source, "reference.json"), PlayerValidationTests.ReferenceJson(reference, false).ToString());
            File.WriteAllText(Path.Combine(source, "search-reference.json"), PlayerValidationTests.ReferenceJson(reference, true).ToString());
            receipt = new JObject { ["success"] = true, ["model_revision"] = PlayerValidation.ModelRevision, ["model_source"] = new string('a', 40),
                ["search_source"] = new string('b', 40), ["model_origin_sha256"] = new string('c', 64), ["unity_preparation_version"] = "6000.3.16f1" };
            SaveReceipt();
        }
        void SaveReceipt()
        {
            var files = new JArray();
            foreach (var name in names)
            {
                using var hash = SHA256.Create(); using var stream = File.OpenRead(Path.Combine(source, name));
                files.Add(new JObject { ["name"] = name, ["bytes"] = stream.Length, ["sha256"] = BitConverter.ToString(hash.ComputeHash(stream)).Replace("-", "").ToLowerInvariant() });
            }
            receipt["files"] = files; File.WriteAllText(Path.Combine(source, "bundle.json"), receipt.ToString());
        }
        [TearDown] public void Cleanup() { Directory.Delete(directory, true); }
        static void Drain(IEnumerator routine)
        {
            using var cleanup = routine as IDisposable;
            while (routine.MoveNext()) if (routine.Current is IEnumerator child) Drain(child);
        }
        IEnumerator Copy(string url, string output) { File.Copy(Path.Combine(source, url.Substring(url.LastIndexOf('/') + 1)), output); yield break; }
        PlayerBundleStage Resolve(List<string> requests = null, Func<string, string, IEnumerator> transfer = null)
        {
            var result = new PlayerBundleStage();
            Drain(PlayerBundleCache.Resolve(Jar, cache, result, (url, output) => { requests?.Add(url); return (transfer ?? Copy)(url, output); }));
            return result;
        }
        PlayerBundleStage Seed() { var result = Resolve(); Assert.That(result.auditPassed, Is.True, result.error); return result; }
        [Test] public void WarmCacheFetchesOnlyManifestAndStillAuditsEveryFile()
        {
            var coldCalls = new List<string>(); var cold = Resolve(coldCalls); Assert.That(cold.auditPassed, Is.True, cold.error);
            Assert.That(coldCalls, Has.Count.EqualTo(6)); Assert.That(cold.cacheReused, Is.False);
            var warmCalls = new List<string>(); var warm = Resolve(warmCalls);
            Assert.That(warm.auditPassed, Is.True, warm.error); Assert.That(warm.cacheReused, Is.True);
            Assert.That(warmCalls, Is.EqualTo(new[] { Jar + "/bundle.json" })); Assert.That(warm.directory, Is.EqualTo(cold.directory));
            Assert.That(PlayerBundleLoader.Load(warm.directory).Reference.Cases, Has.Length.EqualTo(15));
            Assert.That(Directory.GetDirectories(cache).Select(Path.GetFileName), Is.EqualTo(new[] { "active" }));
        }
        [Test] public void EqualLengthAndMtimeCorruptionIsRepairedBeforeReuse()
        {
            var cold = Seed(); var path = Path.Combine(cold.directory, "model-fp32.sentis"); var time = File.GetLastWriteTimeUtc(path);
            File.WriteAllText(path, "oops"); File.SetLastWriteTimeUtc(path, time);
            var calls = new List<string>(); var repaired = Resolve(calls);
            Assert.That(repaired.auditPassed, Is.True, repaired.error); Assert.That(repaired.cacheReused, Is.False);
            Assert.That(repaired.cacheRejectedError, Does.Contain("SHA-256")); Assert.That(calls, Has.Count.EqualTo(6));
            Assert.That(File.ReadAllText(path), Is.EqualTo("fp32"));
        }
        [Test] public void ChangedManifestPublishesOnlyAfterCandidateAuditAndBoundsStorage()
        {
            var cold = Seed(); var oldPath = Path.Combine(cold.directory, "model-float16.sentis");
            File.WriteAllText(Path.Combine(source, "model-float16.sentis"), "changed"); SaveReceipt();
            var changed = Resolve(transfer: (url, output) => { Assert.That(File.ReadAllText(oldPath), Is.EqualTo("float16"), "Keep previous active bytes throughout transfer."); return Copy(url, output); });
            Assert.That(changed.auditPassed, Is.True, changed.error); Assert.That(changed.cacheReused, Is.False);
            Assert.That(File.ReadAllText(oldPath), Is.EqualTo("changed"));
            Assert.That(Directory.GetDirectories(cache).Select(Path.GetFileName), Is.EqualTo(new[] { "active" }));
        }
        [Test] public void FailedUpdatePreservesOldActiveAndNextAttemptDoesNotAccumulateVersions()
        {
            var cold = Seed(); File.WriteAllText(Path.Combine(source, "model-float16.sentis"), "changed"); SaveReceipt();
            IEnumerator Fail(string path) { File.WriteAllText(path, "part"); yield return null; throw new IOException("interrupted update"); }
            var failed = Resolve(transfer: (url, output) => url.EndsWith("model-float16.sentis") ? Fail(output) : Copy(url, output));
            Assert.That(failed.auditPassed, Is.False); Assert.That(failed.error, Does.Contain("interrupted update"));
            Assert.That(File.ReadAllText(Path.Combine(cold.directory, "model-float16.sentis")), Is.EqualTo("float16"));
            Assert.That(Directory.GetDirectories(cache).Length, Is.LessThanOrEqualTo(2));
            var retry = Resolve(); Assert.That(retry.auditPassed, Is.True, retry.error);
            Assert.That(Directory.GetDirectories(cache).Select(Path.GetFileName), Is.EqualTo(new[] { "active" }));
        }
        [Test] public void InvalidManifestStopsBeforeWeightsAndKeepsOldCache()
        {
            var cold = Seed(); receipt["model_revision"] = "other"; File.WriteAllText(Path.Combine(source, "bundle.json"), receipt.ToString());
            var calls = new List<string>(); var failed = Resolve(calls);
            Assert.That(failed.auditPassed, Is.False); Assert.That(failed.error, Does.Contain("pinned")); Assert.That(calls, Has.Count.EqualTo(1));
            Assert.That(File.ReadAllText(Path.Combine(cold.directory, "model-fp32.sentis")), Is.EqualTo("fp32"));
        }
        [Test] public void UnknownNonemptyDirectoryIsNeverAdoptedOrDeleted()
        {
            Directory.CreateDirectory(cache); var sentinel = Path.Combine(cache, "user-data.txt"); File.WriteAllText(sentinel, "keep");
            var calls = new List<string>(); var failed = Resolve(calls);
            Assert.That(failed.auditPassed, Is.False); Assert.That(failed.error, Does.Contain("owned")); Assert.That(calls, Is.Empty);
            Assert.That(File.ReadAllText(sentinel), Is.EqualTo("keep"));
        }
        [Test] public void UnexpectedFilesPreventCleanupInsteadOfBeingDeleted()
        {
            var cold = Seed(); var sentinel = Path.Combine(cold.directory, "keep.txt"); File.WriteAllText(sentinel, "keep");
            var calls = new List<string>(); var failed = Resolve(calls);
            Assert.That(failed.auditPassed, Is.False); Assert.That(calls, Is.Empty); Assert.That(File.ReadAllText(sentinel), Is.EqualTo("keep"));
        }
        [Test] public void AConcurrentCacheOwnerPreventsTransferAndReplacement()
        {
            var cold = Seed();
            using var lease = new FileStream(Path.Combine(cache, ".lock"), FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None);
            var calls = new List<string>(); var failed = Resolve(calls);
            Assert.That(failed.auditPassed, Is.False); Assert.That(calls, Is.Empty);
            Assert.That(File.ReadAllText(Path.Combine(cold.directory, "model-fp32.sentis")), Is.EqualTo("fp32"));
        }
        [Test] public void InterruptedPromotionRestoresPreviousActiveBeforeRetry()
        {
            var cold = Seed(); Directory.Move(cold.directory, Path.Combine(cache, "previous"));
            var calls = new List<string>(); var restored = Resolve(calls);
            Assert.That(restored.auditPassed, Is.True, restored.error); Assert.That(restored.cacheReused, Is.True);
            Assert.That(calls, Has.Count.EqualTo(1)); Assert.That(Directory.Exists(Path.Combine(cache, "previous")), Is.False);
        }
        [UnityTest] public IEnumerator ActualFileUrlWarmCacheReturnsOneManifestTransfer()
        {
            var url = new Uri(source + Path.DirectorySeparatorChar).AbsoluteUri.TrimEnd('/');
            var cold = new PlayerBundleStage(); yield return PlayerBundleCache.Resolve(url, cache, cold);
            Assert.That(cold.auditPassed, Is.True, cold.error); Assert.That(cold.transferredFiles, Is.EqualTo(6));
            var warm = new PlayerBundleStage(); yield return PlayerBundleCache.Resolve(url, cache, warm);
            Assert.That(warm.auditPassed, Is.True, warm.error); Assert.That(warm.cacheReused, Is.True); Assert.That(warm.transferredFiles, Is.EqualTo(1));
        }
        [Test] public void MalformedCachedManifestIsRepairedWithoutClaimingReuse()
        {
            var cold = Seed(); File.WriteAllText(Path.Combine(cold.directory, "bundle.json"), "broken manifest");
            var calls = new List<string>(); var repaired = Resolve(calls);
            Assert.That(repaired.auditPassed, Is.True, repaired.error); Assert.That(repaired.cacheReused, Is.False);
            Assert.That(repaired.cacheRejectedError, Is.Not.Empty); Assert.That(calls, Has.Count.EqualTo(6));
        }
        sealed class Provider : IPlayerEmbedder
        {
            public BackendType Backend { get; set; }
            public float[] EmbedRaw(string text) => PlayerValidationTests.Vector();
            public float[] EmbedQuery(string text) => PlayerValidationTests.Vector();
            public float[] EmbedDocument(string text, string title = null) => PlayerValidationTests.Vector();
            public void Dispose() { }
        }
        [Test] public void ExecutionReusesAuditedCacheAndHoldsItsLeaseThroughoutInference()
        {
            for (var run = 0; run < 2; run++)
            {
                PlayerRunReport result = null; var requests = 0;
                var config = new PlayerRunConfiguration { Bundle = Jar, Output = Path.Combine(directory, "results.json"), RunId = new string('a', 32) };
                Drain(PlayerValidationExecution.Run(config, PlayerRunProtocolTests.Build(), cache, _ => { }, value => result = value,
                    (url, output) => { requests++; return Copy(url, output); }, encode: _ => (new int[128], new int[128]),
                    create: (bundle, _, backend) =>
                    {
                        Assert.That(bundle.Directory, Is.EqualTo(Path.Combine(cache, "active")));
                        Assert.Throws<IOException>(() => { using var competing = new FileStream(Path.Combine(cache, ".lock"), FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None); });
                        return new Provider { Backend = backend };
                    }));
                Assert.That(result.success, Is.True, result.error); Assert.That(result.gpuVerified, Is.False);
                Assert.That(result.staging.auditPassed, Is.True); Assert.That(result.staging.cacheReused, Is.EqualTo(run == 1));
                Assert.That(requests, Is.EqualTo(run == 0 ? 6 : 1));
                using var released = new FileStream(Path.Combine(cache, ".lock"), FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None);
            }
        }
    }
}
