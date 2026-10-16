using System;
using System.Collections;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Security.Cryptography;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using UnityEngine.TestTools;

namespace EmbeddingGemma.Tests
{
    public sealed class GpuMediaBundleCacheTests
    {
        string root, source, cache, firstModel, secondModel, tokenizer; JObject descriptor;
        string Store(byte[] bytes, string extension)
        {
            using var hash = SHA256.Create(); var sha = BitConverter.ToString(hash.ComputeHash(bytes)).Replace("-", "").ToLowerInvariant();
            var name = "objects/" + sha + extension; File.WriteAllBytes(Path.Combine(source, name), bytes);
            descriptor["files"][name] = new JObject { ["bytes"] = bytes.Length, ["sha256"] = sha }; return name;
        }
        [SetUp] public void Setup()
        {
            root = Path.Combine(Path.GetTempPath(), "embeddinggemma-media-cache-" + Guid.NewGuid().ToString("N"));
            source = Path.Combine(root, "source"); cache = Path.Combine(root, "cache"); Directory.CreateDirectory(Path.Combine(source, "objects"));
            descriptor = new JObject { ["schema_version"] = 1, ["backend"] = "GPUCompute", ["sentis_version"] = "2.6.1",
                ["model_revision"] = GpuMediaBundle.PinnedRevision, ["source_commit"] = new string('a', 40), ["files"] = new JObject(), ["modules"] = new JObject() };
            firstModel = Store(new byte[] { 1, 2, 3 }, ".sentis"); secondModel = Store(new byte[] { 4, 5, 6 }, ".sentis"); tokenizer = Store(new byte[] { 7 }, ".json");
            descriptor["modules"]["first"] = new JObject { ["kind"] = "text", ["model"] = firstModel, ["tokenizer"] = tokenizer };
            descriptor["modules"]["second"] = new JObject { ["kind"] = "text", ["model"] = secondModel, ["tokenizer"] = tokenizer };
            Publish();
        }
        void Publish() => File.WriteAllText(Path.Combine(source, "bundle.json"), descriptor.ToString());
        [TearDown] public void Cleanup() { if (Directory.Exists(root)) Directory.Delete(root, true); }
        Task Acquire(string module = "first", CancellationToken token = default)
        {
            var type = typeof(TextEmbedder).Assembly.GetType("EmbeddingGemma.GpuMediaBundleCache");
            Assert.That(type, Is.Not.Null, "A reusable module cache must audit prepared content and hold an exclusive lease.");
            return (Task)type.GetMethod("AcquireAsync").Invoke(null, new object[] { source, module, cache, token });
        }
        static IEnumerator Wait(Task task) { while (!task.IsCompleted) yield return null; }
        static object Result(Task task) { task.GetAwaiter().GetResult(); return task.GetType().GetProperty("Result").GetValue(task); }
        static T Property<T>(object lease, string name) => (T)lease.GetType().GetProperty(name).GetValue(lease);
        static void Close(object lease) => ((IDisposable)lease).Dispose();

        [UnityTest] public IEnumerator ColdCopiesOnlySelectedObjectsWarmAuditsAndModuleSwitchReusesSharedTokenizer()
        {
            var cold = Acquire(); yield return Wait(cold); var first = Result(cold);
            Assert.That(Property<int>(first, "TransferredFiles"), Is.EqualTo(3));
            Assert.That(File.Exists(Path.Combine(cache, secondModel)), Is.False); Close(first);
            var warm = Acquire(); yield return Wait(warm); var reused = Result(warm);
            Assert.That(Property<int>(reused, "TransferredFiles"), Is.EqualTo(1));
            Assert.That(Property<int>(reused, "ReusedObjects"), Is.EqualTo(2)); Close(reused);
            var switched = Acquire("second"); yield return Wait(switched); var second = Result(switched);
            Assert.That(Property<int>(second, "TransferredFiles"), Is.EqualTo(2));
            Assert.That(Property<int>(second, "ReusedObjects"), Is.EqualTo(1));
            Property<GpuMediaBundle>(second, "Bundle").VerifyFile(secondModel); Close(second);
        }
        [UnityTest] public IEnumerator SameLengthCorruptionRepairsOnlyTheRejectedObject()
        {
            var cold = Acquire(); yield return Wait(cold); Close(Result(cold)); File.WriteAllBytes(Path.Combine(cache, firstModel), new byte[] { 9, 9, 9 });
            var repaired = Acquire(); yield return Wait(repaired); var lease = Result(repaired);
            Assert.That(Property<int>(lease, "TransferredFiles"), Is.EqualTo(2)); Assert.That(Property<int>(lease, "ReusedObjects"), Is.EqualTo(1));
            Assert.That(File.ReadAllBytes(Path.Combine(cache, firstModel)), Is.EqualTo(new byte[] { 1, 2, 3 })); Close(lease);
        }
        [UnityTest] public IEnumerator ActiveLeasePreventsAnotherModuleFromRepairingOrLoadingTheCache()
        {
            var cold = Acquire(); yield return Wait(cold); var lease = Result(cold);
            try { var concurrent = Acquire("second"); yield return Wait(concurrent); Assert.That(concurrent.Exception?.InnerException, Is.InstanceOf<IOException>()); }
            finally { Close(lease); }
            var retry = Acquire("second"); yield return Wait(retry); Close(Result(retry));
        }
        [UnityTest] public IEnumerator CancelledPreparationReleasesItsLockAndCanBeRetried()
        {
            using var cancellation = new CancellationTokenSource(); var pending = Acquire(token: cancellation.Token); cancellation.Cancel();
            yield return Wait(pending); Assert.That(pending.IsCanceled, Is.True);
            var retry = Acquire(); yield return Wait(retry); Close(Result(retry));
        }
        [UnityTest] public IEnumerator MissingSourceObjectDoesNotPublishSuccessAndRetryRecovers()
        {
            File.Delete(Path.Combine(source, firstModel)); var broken = Acquire(); yield return Wait(broken);
            Assert.That(broken.Exception?.InnerException, Is.InstanceOf<IOException>());
            File.WriteAllBytes(Path.Combine(source, firstModel), new byte[] { 1, 2, 3 });
            var retry = Acquire(); yield return Wait(retry); Close(Result(retry));
        }
        [UnityTest] public IEnumerator ForeignDirectoryIsUntouched()
        {
            Directory.CreateDirectory(cache); File.WriteAllText(Path.Combine(cache, "keep.txt"), "user data");
            var attempt = Acquire(); yield return Wait(attempt); Assert.That(attempt.Exception?.InnerException, Is.InstanceOf<IOException>());
            Assert.That(Directory.GetFileSystemEntries(cache).Select(Path.GetFileName), Is.EqualTo(new[] { "keep.txt" }));
            Assert.That(File.ReadAllText(Path.Combine(cache, "keep.txt")), Is.EqualTo("user data"));
        }
        [UnityTest] public IEnumerator ChangedProducerDescriptorReusesAuditedIdenticalObjects()
        {
            var first = Acquire(); yield return Wait(first); Close(Result(first)); descriptor["source_commit"] = new string('b', 40); Publish();
            var next = Acquire(); yield return Wait(next); var lease = Result(next);
            Assert.That(Property<int>(lease, "TransferredFiles"), Is.EqualTo(1)); Assert.That(Property<int>(lease, "ReusedObjects"), Is.EqualTo(2)); Close(lease);
        }
        [UnityTest] public IEnumerator OriginalAudioCoefficientObjectsMayKeepTheirBinExtension()
        {
            var coefficient = Store(new byte[] { 0, 0, 128, 63 }, ".bin");
            descriptor["modules"]["audio"] = new JObject { ["kind"] = "audio", ["valid_audio_frames"] = 1119,
                ["preprocessor"] = firstModel, ["backbone"] = firstModel, ["tokenizer"] = tokenizer, ["tokenizer_config"] = tokenizer,
                ["models"] = new JArray(firstModel, firstModel), ["operations"] = new JArray(new JObject { ["kind"] = "sequential_scale",
                    ["inputs"] = new JArray("input"), ["output"] = "output", ["first_scale"] = .25, ["weight_file"] = coefficient }) };
            Publish(); var pending = Acquire("audio"); yield return Wait(pending); var lease = Result(pending);
            Assert.That(Property<int>(lease, "TransferredFiles"), Is.EqualTo(4));
            Property<GpuMediaBundle>(lease, "Bundle").VerifyFile(coefficient); Close(lease);
        }
    }
}
