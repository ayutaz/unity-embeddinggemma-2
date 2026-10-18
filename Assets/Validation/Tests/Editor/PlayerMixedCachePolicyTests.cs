using System;
using System.IO;
using System.Linq;
using System.Reflection;
using Newtonsoft.Json.Linq;
using NUnit.Framework;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class PlayerMixedCachePolicyTests
    {
        string root, cache, header;
        [SetUp] public void Setup()
        {
            root = Path.Combine(Path.GetTempPath(), "embeddinggemma-mixed-cache-" + Guid.NewGuid().ToString("N"));
            Assert.That(Directory.Exists(root), Is.False); Directory.CreateDirectory(root); cache = Path.Combine(root, "cache"); header = Path.Combine(cache, "mixed-cold-complete.json");
        }
        [TearDown] public void Cleanup()
        {
            if (File.Exists(header)) File.Delete(header); if (Directory.Exists(cache)) Directory.Delete(cache); Directory.Delete(root);
        }
        PlayerMixedConfiguration Config(string phase) => PlayerMixedConfiguration.Parse(new[] {
            "--embeddinggemma-mixed-bundle", Path.Combine(root, "models"), "--embeddinggemma-mixed-fixtures", Path.Combine(root, "fixtures"),
            "--embeddinggemma-mixed-cache", cache, "--embeddinggemma-mixed-output", Path.Combine(root, "results", "result.json"),
            "--embeddinggemma-mixed-run-id", new string('a', 32), "--embeddinggemma-mixed-phase", phase,
            "--embeddinggemma-mixed-precision", "fp32", "--embeddinggemma-mixed-descriptor", PlayerMixedConfiguration.Fp32Descriptor });
        static JObject Header() => new JObject { ["schema_version"] = 1, ["success"] = true, ["gpu_verified"] = true, ["phase"] = "cold",
            ["source_commit"] = new string('c', 40), ["precision"] = "fp32", ["bundle_descriptor_sha256"] = PlayerMixedConfiguration.Fp32Descriptor,
            ["run_id"] = new string('b', 32), ["cold_report_sha256"] = new string('d', 64),
            ["base_media_vectors"] = 36, ["warm_media_vectors"] = 45, ["mixed_documents"] = 18, ["mixed_queries"] = 4, ["mixed_rank_and_score_pairs"] = 72 };
        JObject Validate(string phase)
        {
            var type = AppDomain.CurrentDomain.GetAssemblies().Select(a => a.GetType("EmbeddingGemma.Validation.PlayerMixedCachePolicy")).FirstOrDefault(t => t != null);
            Assert.That(type, Is.Not.Null, "Independent warm execution must bind to a complete cold-process receipt.");
            return (JObject)type.GetMethod("Validate").Invoke(null, new object[] { Config(phase), new PlayerBuildInfo { codeCommit = new string('c', 40) } });
        }
        static void Reject(Action action) => Assert.That(Assert.Throws<TargetInvocationException>(() => action()).InnerException,
            Is.InstanceOf<ArgumentException>().Or.InstanceOf<IOException>().Or.InstanceOf<InvalidOperationException>());
        [Test] public void ColdRequiresANewCacheWithoutCreatingOne()
        { var result = Validate("cold"); Assert.That((bool)result["cache_existed"], Is.False); Assert.That(Directory.Exists(cache), Is.False); }
        [Test] public void AnExistingCacheCannotBeRelabeledCold()
        { Directory.CreateDirectory(cache); Reject(() => Validate("cold")); }
        [Test] public void WarmWithoutCompleteColdEvidenceIsRejected()
        { Directory.CreateDirectory(cache); Reject(() => Validate("warm")); }
        [Test] public void WarmRetainsColdIdentityButDoesNotCertifyGpuFromJsonFlags()
        {
            Directory.CreateDirectory(cache); File.WriteAllText(header, Header().ToString()); var result = Validate("warm");
            Assert.That((string)result["cold_run_id"], Is.EqualTo(new string('b', 32))); Assert.That((string)result["cold_report_sha256"], Is.EqualTo(new string('d', 64)));
            Assert.That((bool)result["cache_transfers_must_be_zero"], Is.True); Assert.That((bool)result["gpu_execution_certified"], Is.False);
            Assert.That(((string)result["cold_qualification_sha256"]).Length, Is.EqualTo(64));
        }
        [TestCase("source")] [TestCase("descriptor")] [TestCase("precision")] [TestCase("phase")]
        [TestCase("same_run_id")] [TestCase("bad_hash")] [TestCase("failed")] [TestCase("gpu_false")]
        [TestCase("partial_media")] [TestCase("malformed")] [TestCase("oversized")]
        public void ChangedPartialOrReplayedColdEvidenceCannotQualifyWarm(string problem)
        {
            var value = Header(); Directory.CreateDirectory(cache);
            if (problem == "source") value["source_commit"] = new string('e', 40);
            if (problem == "descriptor") value["bundle_descriptor_sha256"] = new string('e', 64);
            if (problem == "precision") value["precision"] = "float16";
            if (problem == "phase") value["phase"] = "warm";
            if (problem == "same_run_id") value["run_id"] = new string('a', 32);
            if (problem == "bad_hash") value["cold_report_sha256"] = "unknown";
            if (problem == "failed") value["success"] = false;
            if (problem == "gpu_false") value["gpu_verified"] = false;
            if (problem == "partial_media") value["base_media_vectors"] = 35;
            File.WriteAllText(header, problem == "malformed" ? "{}" : problem == "oversized" ? new string(' ', 1024 * 1024 + 1) : value.ToString());
            Reject(() => Validate("warm"));
        }
    }
}
