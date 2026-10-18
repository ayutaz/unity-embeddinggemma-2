using System;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Threading.Tasks;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using UnityEngine;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class PlayerMixedExecutionTests
    {
        string root, output;
        [SetUp] public void Setup()
        {
            root = Path.Combine(Path.GetTempPath(), "embeddinggemma-mixed-entry-" + Guid.NewGuid().ToString("N"));
            Assert.That(Directory.Exists(root), Is.False); Directory.CreateDirectory(root); output = Path.Combine(root, "results", "result.json");
        }
        [TearDown] public void Cleanup()
        {
            if (File.Exists(output)) File.Delete(output);
            var results = Path.Combine(root, "results"); if (Directory.Exists(results)) Directory.Delete(results);
            Directory.Delete(root); // Nonrecursive: unexpected data is preserved and causes the test to fail.
        }
        static Type Required()
        {
            var type = AppDomain.CurrentDomain.GetAssemblies().Select(a => a.GetType("EmbeddingGemma.Validation.PlayerMixedExecution")).FirstOrDefault(t => t != null);
            Assert.That(type, Is.Not.Null, "Complete Player execution needs an explicit real-process entry."); return type;
        }
        string[] Args() => new[] { "Player.exe", "-force-d3d12", "--embeddinggemma-mixed-bundle", Path.Combine(root, "models"),
            "--embeddinggemma-mixed-fixtures", Path.Combine(root, "fixtures"), "--embeddinggemma-mixed-cache", Path.Combine(root, "cache"),
            "--embeddinggemma-mixed-output", output, "--embeddinggemma-mixed-run-id", new string('a', 32),
            "--embeddinggemma-mixed-phase", "cold", "--embeddinggemma-mixed-precision", "fp32", "--embeddinggemma-mixed-descriptor", PlayerMixedConfiguration.Fp32Descriptor };
        static PlayerBuildInfo Build() => new PlayerBuildInfo { codeCommit = new string('a', 40), unityVersion = Application.unityVersion,
            sentisVersion = "2.6.1", target = "StandaloneWindows64", scriptingBackend = "IL2CPP", stripping = "High",
            sourceSha256 = new JObject { ["Assets/Validation/Runtime/fixture.cs"] = new string('b', 64) } };
        static Task<bool> Run(PlayerBuildInfo build, string[] args) => (Task<bool>)Required().GetMethod("Run").Invoke(null, new object[] { build, args });
        [TestCase("empty", false)] [TestCase("legacy", false)] [TestCase("mixed", true)] [TestCase("mixed_unknown", true)]
        public void LegacyAndOrdinaryInvocationsKeepTheirExistingRoute(string kind, bool expected)
        {
            var args = kind == "empty" ? Array.Empty<string>() : kind == "legacy" ? new[] { "--embeddinggemma-media-module", "audio-1119" }
                : kind == "mixed" ? new[] { "--embeddinggemma-mixed-phase", "cold" } : new[] { "--embeddinggemma-mixed-unknown" };
            var method = Required().GetMethod("IsRequested"); Assert.That((bool)method.Invoke(null, new object[] { args }), Is.EqualTo(expected));
            if (kind == "empty") Assert.That((bool)method.Invoke(null, new object[] { null }), Is.False);
        }
        [Test] public async Task EditorCannotClaimActualReleasePlayerOrStartModelPreparation()
        {
            Assert.That(await Run(Build(), Args()), Is.False); var report = JObject.Parse(File.ReadAllText(output));
            Assert.That((bool)report["success"], Is.False); Assert.That((bool)report["gpu_verified"], Is.False);
            Assert.That((string)report["error_type"], Is.EqualTo("NotSupportedException")); Assert.That((bool)report["is_editor"], Is.True);
            Assert.That(Directory.Exists(Path.Combine(root, "models")), Is.False); Assert.That(Directory.Exists(Path.Combine(root, "cache")), Is.False);
            Assert.That(report.ToString(), Does.Not.Contain(root)); Assert.That(report["error"], Is.Null);
        }
        [Test] public void ExistingReportBytesAreNeverTruncatedOrReused()
        {
            Directory.CreateDirectory(Path.GetDirectoryName(output)); File.WriteAllText(output, "original evidence");
            Assert.ThrowsAsync<IOException>(async () => await Run(Build(), Args())); Assert.That(File.ReadAllText(output), Is.EqualTo("original evidence"));
        }
        [Test] public void MalformedArgumentsCannotCreateAReportOrCache()
        {
            var args = Args().Concat(new[] { "--embeddinggemma-mixed-allow-cpu", "true" }).ToArray();
            Assert.ThrowsAsync<ArgumentException>(async () => await Run(Build(), args));
            Assert.That(File.Exists(output), Is.False); Assert.That(Directory.Exists(Path.GetDirectoryName(output)), Is.False);
            Assert.That(Directory.Exists(Path.Combine(root, "cache")), Is.False);
        }
    }
}
