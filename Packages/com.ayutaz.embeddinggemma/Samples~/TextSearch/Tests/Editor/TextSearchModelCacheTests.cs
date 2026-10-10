using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.TestTools;

namespace EmbeddingGemma.Samples.Tests
{
    public sealed class TextSearchModelCacheTests
    {
        string directory, source, root;
        List<string> requests;
        const string Jar = "jar:file:///sample.apk!/assets/EmbeddingGemmaTextSearch";
        [SetUp] public void Setup()
        {
            directory = Path.Combine(Path.GetTempPath(), "embeddinggemma-sample-cache-" + Guid.NewGuid().ToString("N"));
            source = Path.Combine(directory, "source"); root = Path.Combine(directory, "cache"); Directory.CreateDirectory(source);
            requests = new List<string>();
            File.WriteAllBytes(Path.Combine(source, "model-fp32.sentis"), new byte[] { 1, 2, 3, 4 });
            File.WriteAllBytes(Path.Combine(source, "model-float16.sentis"), new byte[] { 5, 6 });
            File.WriteAllText(Path.Combine(source, "tokenizer.json"), "{\"tiny_fixture\":true}");
            Manifest();
        }
        [TearDown] public void Cleanup() { Directory.Delete(directory, true); }
        void Manifest()
        {
            var files = new JArray();
            foreach (var name in new[] { "model-fp32.sentis", "model-float16.sentis", "tokenizer.json" })
            {
                var path = Path.Combine(source, name); using var hash = SHA256.Create();
                files.Add(new JObject { ["name"] = name, ["bytes"] = new FileInfo(path).Length,
                    ["sha256"] = BitConverter.ToString(hash.ComputeHash(File.ReadAllBytes(path))).Replace("-", "").ToLowerInvariant() });
            }
            File.WriteAllText(Path.Combine(source, "preparation.json"), new JObject { ["success"] = true, ["error"] = "", ["unityVersion"] = Application.unityVersion,
                ["sourceCommit"] = new string('a', 40), ["searchSourceCommit"] = new string('b', 40), ["modelSha256"] = new string('c', 64), ["files"] = files }.ToString());
        }
        IEnumerator Copy(string uri, string path)
        {
            var name = uri.Substring(uri.LastIndexOf('/') + 1); requests.Add(name);
            File.Copy(Path.Combine(source, name), path); yield return null;
        }
        static void Drain(IEnumerator routine)
        {
            var stack = new Stack<IEnumerator>(); stack.Push(routine);
            try { while (stack.Count > 0) { var current = stack.Peek(); if (!current.MoveNext()) { (stack.Pop() as IDisposable)?.Dispose(); continue; } if (current.Current is IEnumerator child) stack.Push(child); } }
            finally { while (stack.Count > 0) (stack.Pop() as IDisposable)?.Dispose(); }
        }
        TextSearchModelStage Resolve(string model = "model-fp32.sentis", Func<string, string, IEnumerator> copy = null)
        {
            var result = new TextSearchModelStage(); Drain(TextSearchModelCache.Resolve(Jar + "/" + model, Jar + "/tokenizer.json", root, result, copy ?? Copy)); return result;
        }
        [Test] public void UrlDefaultPathsKeepForwardSlashes()
            => Assert.That(TextSearchModelCache.Combine(Jar, "model-fp32.sentis"), Is.EqualTo(Jar + "/model-fp32.sentis"));
        [Test] public void CopiesOnlySelectedPrecisionAndWarmRunTransfersOnlyManifest()
        {
            using (var first = Resolve())
            {
                Assert.That(first.Success, Is.True, first.Error); Assert.That(first.CacheReused, Is.False);
                Assert.That(requests, Is.EqualTo(new[] { "preparation.json", "model-fp32.sentis", "tokenizer.json" }));
                Assert.That(first.ModelPath, Is.EqualTo(Path.Combine(root, "active", "model-fp32.sentis")));
                Assert.That(File.Exists(Path.Combine(root, "active", "model-float16.sentis")), Is.False);
            }
            requests.Clear(); using var warm = Resolve(); Assert.That(warm.Success, Is.True, warm.Error); Assert.That(warm.CacheReused, Is.True);
            Assert.That(requests, Is.EqualTo(new[] { "preparation.json" })); Assert.That(warm.TransferredFiles, Is.EqualTo(1));
            Assert.That(Directory.GetDirectories(root).Select(Path.GetFileName), Is.EqualTo(new[] { "active" }));
        }
        [Test] public void ChangedPrecisionReplacesTheOwnedCacheAndDoesNotKeepBothWeights()
        {
            using (var first = Resolve()) Assert.That(first.Success, Is.True, first.Error);
            requests.Clear(); using var second = Resolve("model-float16.sentis"); Assert.That(second.Success, Is.True, second.Error);
            Assert.That(File.Exists(Path.Combine(root, "active", "model-fp32.sentis")), Is.False);
            Assert.That(File.Exists(second.ModelPath), Is.True);
            Assert.That(requests, Is.EqualTo(new[] { "preparation.json", "model-float16.sentis", "tokenizer.json" }));
        }
        [Test] public void SameLengthAndMtimeCorruptionIsRepairedBeforeReturningPaths()
        {
            using (var first = Resolve()) Assert.That(first.Success, Is.True, first.Error);
            var path = Path.Combine(root, "active", "model-fp32.sentis"); var stamp = File.GetLastWriteTimeUtc(path);
            File.WriteAllBytes(path, new byte[] { 4, 3, 2, 1 }); File.SetLastWriteTimeUtc(path, stamp);
            requests.Clear(); using var repaired = Resolve(); Assert.That(repaired.Success, Is.True, repaired.Error);
            Assert.That(repaired.CacheReused, Is.False); Assert.That(repaired.CacheRejectedError, Does.Contain("SHA-256"));
            Assert.That(File.ReadAllBytes(path), Is.EqualTo(new byte[] { 1, 2, 3, 4 })); Assert.That(requests.Count, Is.EqualTo(3));
        }
        [TestCase("names")] [TestCase("hash")] [TestCase("success")] [TestCase("source")]
        public void InvalidManifestIsRejectedBeforeAnyWeightTransfer(string failure)
        {
            var path = Path.Combine(source, "preparation.json"); var manifest = JObject.Parse(File.ReadAllText(path));
            if (failure == "names") manifest["files"][0]["name"] = "../outside.sentis";
            if (failure == "hash") manifest["files"][0]["sha256"] = "bad";
            if (failure == "success") manifest["success"] = false;
            if (failure == "source") manifest["sourceCommit"] = "main";
            File.WriteAllText(path, manifest.ToString()); using var stage = Resolve(); Assert.That(stage.Success, Is.False);
            Assert.That(stage.Error, Does.Contain("Invalid preparation")); Assert.That(requests, Is.EqualTo(new[] { "preparation.json" }));
        }
        [Test] public void AFailedUpdateKeepsTheLastVerifiedActiveAndRetryIsBounded()
        {
            using (var first = Resolve()) Assert.That(first.Success, Is.True, first.Error);
            File.WriteAllBytes(Path.Combine(source, "model-fp32.sentis"), new byte[] { 8, 8, 8, 8 }); Manifest();
            IEnumerator Fail(string uri, string path) { if (uri.EndsWith("tokenizer.json", StringComparison.Ordinal)) throw new IOException("interrupted"); yield return Copy(uri, path); }
            using (var failed = Resolve(copy: Fail)) { Assert.That(failed.Success, Is.False); Assert.That(failed.Error, Does.Contain("interrupted")); }
            Assert.That(File.ReadAllBytes(Path.Combine(root, "active", "model-fp32.sentis")), Is.EqualTo(new byte[] { 1, 2, 3, 4 }));
            Assert.That(Directory.GetDirectories(root).Length, Is.LessThanOrEqualTo(2));
            using var retry = Resolve(); Assert.That(retry.Success, Is.True, retry.Error); Assert.That(Directory.GetDirectories(root).Length, Is.EqualTo(1));
        }
        [Test] public void UnownedDataIsPreservedWithoutTransfer()
        {
            Directory.CreateDirectory(root); File.WriteAllText(Path.Combine(root, "user.txt"), "keep");
            using var stage = Resolve(); Assert.That(stage.Success, Is.False); Assert.That(stage.Error, Does.Contain("not owned"));
            Assert.That(requests, Is.Empty); Assert.That(File.ReadAllText(Path.Combine(root, "user.txt")), Is.EqualTo("keep"));
        }
        [Test] public void ALeasePreventsAnotherResolutionUntilTheCallerReleasesIt()
        {
            using var first = Resolve(); Assert.That(first.Success, Is.True, first.Error);
            requests.Clear(); using (var second = Resolve()) { Assert.That(second.Success, Is.False); Assert.That(second.Error, Does.Contain("IOException")); Assert.That(requests, Is.Empty); }
            first.Dispose(); using var third = Resolve(); Assert.That(third.Success, Is.True, third.Error);
        }
        [Test] public void CancellationDisposesTheTransferAndReleasesTheLock()
        {
            var disposed = false;
            IEnumerator Wait(string uri, string path) { try { while (true) yield return null; } finally { disposed = true; } }
            var stage = new TextSearchModelStage(); var operation = TextSearchModelCache.Resolve(Jar + "/model-fp32.sentis", Jar + "/tokenizer.json", root, stage, Wait);
            Assert.That(operation.MoveNext(), Is.True); (operation as IDisposable)?.Dispose(); stage.Dispose();
            Assert.That(disposed, Is.True); Assert.That(stage.Success, Is.False);
            using var next = Resolve(); Assert.That(next.Success, Is.True, next.Error);
        }
        [Test] public void NetworkUrlsAndDifferentTokenizerSourcesAreRejectedBeforeTransfer()
        {
            foreach (var inputs in new[] { new[] { "https://example.com/model-fp32.sentis", "https://example.com/tokenizer.json" }, new[] { Jar + "/model-fp32.sentis", Jar + "/other/tokenizer.json" } })
            {
                using var stage = new TextSearchModelStage(); Drain(TextSearchModelCache.Resolve(inputs[0], inputs[1], root, stage, Copy));
                Assert.That(stage.Success, Is.False); Assert.That(stage.Error, Does.Contain("Unsupported model source")); Assert.That(requests, Is.Empty);
            }
        }
        [Test] public void UnknownFilesInOwnedStorageAreNotDeleted()
        {
            using (var first = Resolve()) Assert.That(first.Success, Is.True, first.Error);
            var path = Path.Combine(root, "active", "user.txt"); File.WriteAllText(path, "keep");
            requests.Clear(); using var stage = Resolve(); Assert.That(stage.Success, Is.False); Assert.That(stage.Error, Does.Contain("Unexpected cache"));
            Assert.That(requests, Is.Empty); Assert.That(File.ReadAllText(path), Is.EqualTo("keep"));
        }
        sealed class Provider : ITextEmbedder
        {
            public int Disposals;
            public BackendType Backend => BackendType.CPU;
            public float[] EmbedQuery(string text) => Vector();
            public float[] EmbedDocument(string text, string title = null) => Vector();
            static float[] Vector() { var result = new float[768]; result[0] = 1; return result; }
            public void Dispose() { Disposals++; }
        }
        [Test] public void SampleLoadsOnlyResolvedPathsAndHoldsLeaseUntilWorkerRelease()
        {
            var owner = new GameObject("Sample cache contract"); var corpus = new TextAsset("{\"documents\":[{\"id\":\"a\",\"text\":\"cat\"}]}"); var provider = new Provider();
            try
            {
                var sample = owner.AddComponent<TextSearchSample>(); sample.Corpus = corpus; sample.Backend = BackendType.CPU;
                sample.ModelPath = Jar + "/model-fp32.sentis"; sample.TokenizerPath = Jar + "/tokenizer.json";
                var calls = 0;
                Drain(sample.PrepareDocumentsFromStreamingAssets(root, Copy, (model, tokenizer) =>
                {
                    calls++; Assert.That(model, Is.EqualTo(Path.Combine(root, "active", "model-fp32.sentis")));
                    Assert.That(tokenizer, Is.EqualTo(Path.Combine(root, "active", "tokenizer.json")));
                    Assert.That(File.ReadAllBytes(model), Is.EqualTo(new byte[] { 1, 2, 3, 4 })); return provider;
                }));
                Assert.That(sample.IsReady, Is.True, sample.Error); Assert.That(sample.IsPreparing, Is.False); Assert.That(calls, Is.EqualTo(1));
                Assert.That(sample.ModelPath, Is.EqualTo(Jar + "/model-fp32.sentis"), "Keep the caller's source, not a stale cache path.");
                using (var blocked = Resolve()) Assert.That(blocked.Success, Is.False);
                sample.ReleaseDocuments(); Assert.That(provider.Disposals, Is.EqualTo(1));
                using var next = Resolve(); Assert.That(next.Success, Is.True, next.Error);
            }
            finally { UnityEngine.Object.DestroyImmediate(owner); UnityEngine.Object.DestroyImmediate(corpus); }
        }
        [Test] public void SampleCancellationPreventsLateProviderCreationAndDisposesTransport()
        {
            var owner = new GameObject("Sample cancellation contract"); var disposed = false; var calls = 0;
            IEnumerator Wait(string uri, string path) { try { while (true) yield return null; } finally { disposed = true; } }
            try
            {
                var sample = owner.AddComponent<TextSearchSample>(); sample.ModelPath = Jar + "/model-fp32.sentis"; sample.TokenizerPath = Jar + "/tokenizer.json";
                var operation = sample.PrepareDocumentsFromStreamingAssets(root, Wait, (_, _) => { calls++; return new Provider(); });
                Assert.That(operation.MoveNext(), Is.True); Assert.That(sample.IsPreparing, Is.True);
                sample.ReleaseDocuments(); Assert.That(operation.MoveNext(), Is.False);
                Assert.That(disposed, Is.True); Assert.That(calls, Is.Zero); Assert.That(sample.IsPreparing, Is.False); Assert.That(sample.IsReady, Is.False);
                using var next = Resolve(); Assert.That(next.Success, Is.True, next.Error);
            }
            finally { UnityEngine.Object.DestroyImmediate(owner); }
        }
        [Test] public void SampleTransferFailureIsVisibleAndNeverCreatesTheWorker()
        {
            var owner = new GameObject("Sample transfer failure contract"); var calls = 0;
            IEnumerator Fail(string uri, string path) { if (uri.EndsWith("model-fp32.sentis", StringComparison.Ordinal)) throw new IOException("interrupted model transfer"); yield return Copy(uri, path); }
            try
            {
                var sample = owner.AddComponent<TextSearchSample>(); sample.ModelPath = Jar + "/model-fp32.sentis"; sample.TokenizerPath = Jar + "/tokenizer.json";
                Drain(sample.PrepareDocumentsFromStreamingAssets(root, Fail, (_, _) => { calls++; return new Provider(); }));
                Assert.That(sample.IsReady, Is.False); Assert.That(sample.IsPreparing, Is.False); Assert.That(sample.Error, Does.Contain("interrupted model transfer")); Assert.That(calls, Is.Zero);
                using var next = Resolve(); Assert.That(next.Success, Is.True, next.Error);
            }
            finally { UnityEngine.Object.DestroyImmediate(owner); }
        }
        [TestCase(false)] [TestCase(true)] public void SampleProviderFailureEndsProgressAndReleasesTheCache(bool mismatch)
        {
            var owner = new GameObject("Sample provider failure contract"); var corpus = new TextAsset("{\"documents\":[{\"id\":\"a\",\"text\":\"cat\"}]}"); var provider = new Provider();
            try
            {
                var sample = owner.AddComponent<TextSearchSample>(); sample.Corpus = corpus; sample.ModelPath = Jar + "/model-fp32.sentis"; sample.TokenizerPath = Jar + "/tokenizer.json";
                Drain(sample.PrepareDocumentsFromStreamingAssets(root, Copy, (_, _) => mismatch ? provider : throw new InvalidOperationException("provider load failed")));
                Assert.That(sample.IsPreparing, Is.False); Assert.That(sample.IsReady, Is.False); Assert.That(sample.Error, Is.Not.Empty);
                Assert.That(sample.Status, Is.EqualTo("モデルを準備できませんでした。"));
                Assert.That(provider.Disposals, Is.EqualTo(mismatch ? 1 : 0));
                using var next = Resolve(); Assert.That(next.Success, Is.True, next.Error);
            }
            finally { UnityEngine.Object.DestroyImmediate(owner); UnityEngine.Object.DestroyImmediate(corpus); }
        }
        [UnityTest] public IEnumerator RealUnicodeFileUrlTransfersSelectedFilesAndReusesCache()
        {
            var renamed = Path.Combine(directory, "日本語 bundle"); Directory.Move(source, renamed); source = renamed;
            var uri = new Uri(source + Path.DirectorySeparatorChar).AbsoluteUri.TrimEnd('/');
            using (var first = new TextSearchModelStage())
            {
                yield return TextSearchModelCache.Resolve(uri + "/model-float16.sentis", uri + "/tokenizer.json", root, first);
                Assert.That(first.Success, Is.True, first.Error); Assert.That(first.TransferredFiles, Is.EqualTo(3)); Assert.That(first.CacheReused, Is.False);
                Assert.That(File.ReadAllBytes(first.ModelPath), Is.EqualTo(new byte[] { 5, 6 }));
            }
            using var warm = new TextSearchModelStage();
            yield return TextSearchModelCache.Resolve(uri + "/model-float16.sentis", uri + "/tokenizer.json", root, warm);
            Assert.That(warm.Success, Is.True, warm.Error); Assert.That(warm.CacheReused, Is.True); Assert.That(warm.TransferredFiles, Is.EqualTo(1));
            Assert.That(warm.HashBackends.Keys, Is.EquivalentTo(new[] { "model-float16.sentis", "tokenizer.json" }));
        }
        [UnityTest] public IEnumerator RealMissingFileUrlFailsWithoutPublishedCacheOrPartialFile()
        {
            File.Delete(Path.Combine(source, "model-fp32.sentis"));
            var uri = new Uri(source + Path.DirectorySeparatorChar).AbsoluteUri.TrimEnd('/'); using var stage = new TextSearchModelStage();
            yield return TextSearchModelCache.Resolve(uri + "/model-fp32.sentis", uri + "/tokenizer.json", root, stage);
            Assert.That(stage.Success, Is.False); Assert.That(stage.Error, Is.Not.Empty);
            Assert.That(Directory.Exists(Path.Combine(root, "active")), Is.False); Assert.That(Directory.GetFiles(root, "*.part", SearchOption.AllDirectories), Is.Empty);
        }
        [TestCase(false)] [TestCase(true)] public void InterruptedPublicationRecoversTheLastValidBundle(bool damagedActive)
        {
            using (var first = Resolve()) Assert.That(first.Success, Is.True, first.Error);
            var active = Path.Combine(root, "active"); var previous = Path.Combine(root, "previous");
            if (!damagedActive) Directory.Move(active, previous);
            else
            {
                Directory.CreateDirectory(previous); foreach (var file in Directory.GetFiles(active)) File.Copy(file, Path.Combine(previous, Path.GetFileName(file)));
                File.WriteAllBytes(Path.Combine(active, "model-fp32.sentis"), new byte[] { 9, 9, 9, 9 });
            }
            Directory.CreateDirectory(Path.Combine(root, "pending")); File.WriteAllText(Path.Combine(root, "pending", "model-fp32.sentis.part"), "partial");
            requests.Clear(); using var recovered = Resolve(); Assert.That(recovered.Success, Is.True, recovered.Error); Assert.That(recovered.CacheReused, Is.True);
            Assert.That(requests, Is.EqualTo(new[] { "preparation.json" })); Assert.That(Directory.GetDirectories(root).Length, Is.EqualTo(1));
        }
    }
}
