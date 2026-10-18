using System;
using System.IO;
using System.Reflection;
using System.Security.Cryptography;
using System.Threading.Tasks;
using Newtonsoft.Json.Linq;
using NUnit.Framework;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class PlayerMixedCacheIntegrationTests
    {
        string root, source, cache, model;
        [SetUp] public void Setup()
        {
            root = Path.Combine(Path.GetTempPath(), "embeddinggemma-mixed-cache-integration-" + Guid.NewGuid().ToString("N"));
            source = Path.Combine(root, "source"); cache = Path.Combine(root, "run-cache");
            Assert.That(Directory.Exists(root), Is.False); Directory.CreateDirectory(Path.Combine(source, "objects"));
            var files = new JObject();
            string Store(byte[] bytes, string extension)
            {
                using var hash = SHA256.Create(); var sha = BitConverter.ToString(hash.ComputeHash(bytes)).Replace("-", "").ToLowerInvariant();
                var name = "objects/" + sha + extension; File.WriteAllBytes(Path.Combine(source, name), bytes);
                files[name] = new JObject { ["bytes"] = bytes.Length, ["sha256"] = sha }; return name;
            }
            model = Store(new byte[] { 1, 2, 3 }, ".sentis"); var tokenizer = Store(new byte[] { 7 }, ".json");
            File.WriteAllText(Path.Combine(source, "bundle.json"), new JObject {
                ["schema_version"] = 1, ["backend"] = "GPUCompute", ["sentis_version"] = "2.6.1",
                ["model_revision"] = GpuMediaBundle.PinnedRevision, ["source_commit"] = new string('a', 40), ["files"] = files,
                ["modules"] = new JObject { ["text"] = new JObject { ["kind"] = "text", ["model"] = model, ["tokenizer"] = tokenizer } }
            }.ToString());
        }
        [TearDown] public void Cleanup() { if (Directory.Exists(root)) Directory.Delete(root, true); }
        static MethodInfo Required(string name)
        {
            var method = typeof(PlayerMixedCachePolicy).GetMethod(name);
            Assert.That(method, Is.Not.Null, "Mixed execution must respect the existing prepared-cache owner and descriptor accounting."); return method;
        }
        string Prepared() => (string)Required("PreparedCacheDirectory").Invoke(null, new object[] { cache });
        static JObject Evidence(GpuMediaBundleLease lease, bool warm)
            => (JObject)Required("AcquisitionEvidence").Invoke(null, new object[] { "text", lease, warm });
        [Test] public void QualificationIsOutsideTheSdkOwnedPreparedDirectory()
        {
            Assert.That(Prepared(), Is.EqualTo(Path.Combine(cache, "prepared")));
            Assert.That(Path.GetDirectoryName(Path.Combine(cache, PlayerMixedCachePolicy.QualificationName)), Is.Not.EqualTo(Prepared()));
            Assert.That(Directory.Exists(cache), Is.False);
        }
        [Test] public async Task ActualWarmCacheReadsOneDescriptorWithoutTransferringAnyModelObject()
        {
            var prepared = Prepared();
            using (var cold = await GpuMediaBundleCache.AcquireAsync(source, "text", prepared)) { }
            File.WriteAllText(Path.Combine(cache, PlayerMixedCachePolicy.QualificationName), "private cold identity link");
            using var warm = await GpuMediaBundleCache.AcquireAsync(source, "text", prepared);
            var report = Evidence(warm, true);
            Assert.That(warm.TransferredFiles, Is.EqualTo(1)); Assert.That((int)report["descriptor_reads"], Is.EqualTo(1));
            Assert.That((int)report["transferred_objects"], Is.Zero); Assert.That((int)report["reused_objects"], Is.EqualTo(2));
            Assert.That((bool)report["gpu_execution_certified"], Is.False);
        }
        [Test] public async Task RepairDuringActualWarmAcquisitionIsRejected()
        {
            var prepared = Prepared();
            using (var cold = await GpuMediaBundleCache.AcquireAsync(source, "text", prepared)) { }
            File.WriteAllBytes(Path.Combine(prepared, model), new byte[] { 9, 9, 9 });
            using var repaired = await GpuMediaBundleCache.AcquireAsync(source, "text", prepared);
            Assert.That(repaired.TransferredFiles, Is.EqualTo(2));
            Assert.That(Assert.Throws<TargetInvocationException>(() => Evidence(repaired, true)).InnerException, Is.InstanceOf<InvalidOperationException>());
        }
        [Test] public async Task ColdReportsDescriptorAndObjectTransfersSeparately()
        {
            using var cold = await GpuMediaBundleCache.AcquireAsync(source, "text", Prepared()); var report = Evidence(cold, false);
            Assert.That((int)report["transferred_files"], Is.EqualTo(3)); Assert.That((int)report["descriptor_reads"], Is.EqualTo(1));
            Assert.That((int)report["transferred_objects"], Is.EqualTo(2)); Assert.That((long)report["transferred_bytes"], Is.EqualTo(cold.TransferredBytes));
        }
        [TestCase("zero_files")] [TestCase("negative_reuse")] [TestCase("zero_bytes")]
        public void InvalidAcquisitionAccountingCannotBecomeWarmSuccess(string problem)
        {
            Required("AcquisitionEvidence"); var lease = new GpuMediaBundleLease();
            typeof(GpuMediaBundleLease).GetProperty("TransferredFiles").SetValue(lease, problem == "zero_files" ? 0 : 1);
            typeof(GpuMediaBundleLease).GetProperty("ReusedObjects").SetValue(lease, problem == "negative_reuse" ? -1 : 0);
            typeof(GpuMediaBundleLease).GetProperty("TransferredBytes").SetValue(lease, problem == "zero_bytes" ? 0L : 10L);
            Assert.That(Assert.Throws<TargetInvocationException>(() => Evidence(lease, true)).InnerException, Is.InstanceOf<ArgumentException>());
        }
    }
}
