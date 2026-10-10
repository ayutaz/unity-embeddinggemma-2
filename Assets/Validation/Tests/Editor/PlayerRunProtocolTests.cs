using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using Unity.InferenceEngine;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class PlayerRunProtocolTests
    {
        string directory;
        [SetUp] public void Setup() { directory = Path.Combine(Path.GetTempPath(), "embeddinggemma-run-" + Guid.NewGuid().ToString("N")); Directory.CreateDirectory(directory); }
        [TearDown] public void Cleanup() { Directory.Delete(directory, true); }
        PlayerRunConfiguration Config() => new PlayerRunConfiguration { Bundle = directory, Output = Path.Combine(directory, "results.json"), RunId = new string('a', 32) };
        public static PlayerBuildInfo Build() => new PlayerBuildInfo { codeCommit = new string('a', 40), unityVersion = "6000.3.16f1", sentisVersion = "2.6.1",
            target = "StandaloneWindows64", scriptingBackend = "Mono2x", stripping = "Disabled", sourceSha256 = new JObject { ["Assets/Validation/Runtime/example.cs"] = new string('b', 64) } };
        static PlayerBundle Bundle() => new PlayerBundle { Reference = PlayerValidationTests.Reference(), Receipt = new JObject { ["success"] = true } };
        sealed class Provider : IPlayerEmbedder
        {
            public BackendType Backend { get; set; }
            public float[] EmbedRaw(string text) => PlayerValidationTests.Vector();
            public float[] EmbedQuery(string text) => PlayerValidationTests.Vector();
            public float[] EmbedDocument(string text, string title = null) => PlayerValidationTests.Vector();
            public void Dispose() { }
        }
        [Test] public void AcceptsPinnedBuildProvenance() { Build().Validate(); }
        [TestCase("commit")] [TestCase("hash")] [TestCase("unity")] [TestCase("backend")]
        public void InvalidBuildProvenanceIsRejected(string failure)
        {
            var info = Build(); if (failure == "commit") info.codeCommit = "main";
            if (failure == "hash") info.sourceSha256["Assets/Validation/Runtime/example.cs"] = "missing";
            if (failure == "unity") info.unityVersion = "6000.3.15f1";
            if (failure == "backend") info.scriptingBackend = "unknown";
            Assert.Throws<ArgumentException>(() => info.Validate());
        }
        [Test] public void ParsesExplicitPathsWithUnicodeAndStableRunId()
        {
            var output = Path.Combine(directory, "日本語 folder", "results.json");
            var config = PlayerRunConfiguration.Parse(new[] { "-batchmode", "--embeddinggemma-bundle", directory, "--embeddinggemma-output", output, "--embeddinggemma-run-id", new string('b', 32) }, "unused", "unused");
            Assert.That(config.Bundle, Is.EqualTo(directory)); Assert.That(config.Output, Is.EqualTo(output)); Assert.That(config.RunId, Is.EqualTo(new string('b', 32)));
        }
        [TestCase("duplicate")] [TestCase("missing")] [TestCase("run_id")] [TestCase("unknown")]
        public void InvalidArgumentsCannotStartValidation(string failure)
        {
            var args = failure == "duplicate" ? new[] { "--embeddinggemma-bundle", directory, "--embeddinggemma-bundle", directory } :
                failure == "missing" ? new[] { "--embeddinggemma-output" } :
                failure == "run_id" ? new[] { "--embeddinggemma-run-id", "old" } : new[] { "--embeddinggemma-typo", "value" };
            Assert.Throws<ArgumentException>(() => PlayerRunConfiguration.Parse(args, directory, Path.Combine(directory, "results.json")));
        }
        [Test] public void WritesAReportToANewFile()
        {
            var file = new PlayerReportFile(Config().Output); file.Save(new PlayerRunReport { success = false, phase = "failed", error = "model missing" });
            var result = JObject.Parse(File.ReadAllText(Config().Output)); Assert.That((bool)result["success"], Is.False); Assert.That((string)result["error"], Is.EqualTo("model missing"));
        }
        [Test] public void CannotOverwriteAPreviousSuccessfulRun()
        {
            const string old = "{\"success\":true,\"runId\":\"previous\"}"; File.WriteAllText(Config().Output, old);
            Assert.Throws<IOException>(() => new PlayerReportFile(Config().Output)); Assert.That(File.ReadAllText(Config().Output), Is.EqualTo(old));
        }
        [Test] public void ReportsProgressAsIncompleteUntilAllFourConditionsFinish()
        {
            var snapshots = new List<PlayerRunReport>();
            var result = PlayerRunProtocol.Run(Config(), Build(), report => snapshots.Add(JsonConvert.DeserializeObject<PlayerRunReport>(JsonConvert.SerializeObject(report))),
                _ => Bundle(), _ => (new int[128], new int[128]), (_, _, backend) => new Provider { Backend = backend });
            Assert.That(result.success, Is.True, result.error); Assert.That(result.tokenRows, Is.EqualTo(25)); Assert.That(result.conditions, Has.Count.EqualTo(4));
            Assert.That(snapshots.Take(snapshots.Count - 1).All(report => !report.success), Is.True); Assert.That(snapshots.Last().success, Is.True);
            Assert.That(result.validationMode, Is.EqualTo("injected_contract")); Assert.That(result.gpuVerified, Is.False); Assert.That(result.isEditor, Is.True);
        }
        [Test] public void FailedGpuConditionsAreRetainedAlongsideSuccessfulCpuConditions()
        {
            var result = PlayerRunProtocol.Run(Config(), Build(), _ => { }, _ => Bundle(), _ => (new int[128], new int[128]),
                (_, _, backend) => backend == BackendType.GPUCompute ? throw new NotSupportedException("no GPU") : new Provider { Backend = backend });
            Assert.That(result.success, Is.False); Assert.That(result.conditions, Has.Count.EqualTo(4)); Assert.That(result.conditions.Count(condition => condition.success), Is.EqualTo(2));
            Assert.That(result.conditions.Where(condition => condition.requestedBackend == "GPUCompute").All(condition => condition.actualBackend == null), Is.True);
        }
        [Test] public void TokenFailurePreventsAnyInference()
        {
            var calls = 0; var bundle = Bundle(); bundle.Reference.Cases[0].input_ids[0] = 1;
            var result = PlayerRunProtocol.Run(Config(), Build(), _ => { }, _ => bundle, _ => (new int[128], new int[128]), (_, _, _) => { calls++; return new Provider(); });
            Assert.That(result.success, Is.False); Assert.That(result.error, Does.Contain("Token")); Assert.That(calls, Is.Zero);
        }
        [Test] public void BundleFailureIsSavedAsFailure()
        {
            PlayerRunReport last = null;
            var result = PlayerRunProtocol.Run(Config(), Build(), report => last = report, _ => throw new FileNotFoundException("missing bundle"));
            Assert.That(result.success, Is.False); Assert.That(result.error, Does.Contain("missing bundle")); Assert.That(last.completedUtc, Is.Not.Empty);
        }
        [Test] public void ResultWriteFailureCannotBecomeSuccess()
        {
            var calls = 0;
            var result = PlayerRunProtocol.Run(Config(), Build(), _ => throw new IOException("write failed"), _ => Bundle(), _ => (new int[128], new int[128]), (_, _, _) => { calls++; return new Provider(); });
            Assert.That(result.success, Is.False); Assert.That(result.error, Does.Contain("write failed")); Assert.That(calls, Is.Zero);
        }
    }
}
