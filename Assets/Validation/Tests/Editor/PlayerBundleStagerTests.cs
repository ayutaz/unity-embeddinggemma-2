using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEngine.TestTools;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class PlayerBundleStagerTests
    {
        string root, source, destination;
        readonly string[] names = { "bundle.json", "model-fp32.sentis", "model-float16.sentis", "tokenizer.json", "reference.json", "search-reference.json" };
        [SetUp] public void Setup()
        {
            root = Path.Combine(Path.GetTempPath(), "embeddinggemma-stage-" + Guid.NewGuid().ToString("N"));
            source = Path.Combine(root, "日本語 source folder"); destination = Path.Combine(root, "extracted");
            Directory.CreateDirectory(source);
            File.WriteAllText(Path.Combine(source, "model-fp32.sentis"), "fp32");
            File.WriteAllText(Path.Combine(source, "model-float16.sentis"), "float16");
            File.WriteAllText(Path.Combine(source, "tokenizer.json"), "{}");
            var reference = PlayerValidationTests.Reference();
            File.WriteAllText(Path.Combine(source, "reference.json"), PlayerValidationTests.ReferenceJson(reference, false).ToString());
            File.WriteAllText(Path.Combine(source, "search-reference.json"), PlayerValidationTests.ReferenceJson(reference, true).ToString());
            var files = new JArray();
            foreach (var name in names.Skip(1))
            {
                using var hash = SHA256.Create(); using var stream = File.OpenRead(Path.Combine(source, name));
                files.Add(new JObject { ["name"] = name, ["bytes"] = stream.Length,
                    ["sha256"] = BitConverter.ToString(hash.ComputeHash(stream)).Replace("-", "").ToLowerInvariant() });
            }
            File.WriteAllText(Path.Combine(source, "bundle.json"), new JObject { ["success"] = true, ["files"] = files,
                ["model_revision"] = PlayerValidation.ModelRevision, ["model_source"] = new string('a', 40), ["search_source"] = new string('b', 40),
                ["model_origin_sha256"] = new string('c', 64), ["unity_preparation_version"] = "6000.3.16f1" }.ToString());
        }
        [TearDown] public void Cleanup() { Directory.Delete(root, true); }
        string FileUrl => new Uri(source + Path.DirectorySeparatorChar).AbsoluteUri.TrimEnd('/');
        const string JarUrl = "jar:file:///data/app/test/base.apk!/assets/EmbeddingGemmaValidation";
        static void Drain(IEnumerator routine)
        {
            using var cleanup = routine as IDisposable;
            while (routine.MoveNext()) if (routine.Current is IEnumerator inner) Drain(inner);
        }
        IEnumerator CopyFixture(string url, string output)
        {
            File.Copy(Path.Combine(source, url.Substring(url.LastIndexOf('/') + 1)), output);
            yield break;
        }
        [Test] public void FilesystemBundlesAreAuditedInPlaceWithoutCopying()
        {
            var stage = new PlayerBundleStage(); var calls = 0;
            Drain(PlayerBundleStager.Stage(source, destination, stage, (_, _) => { calls++; return CopyFixture("/bundle.json", "unused"); }));
            Assert.That(stage.transferCompleted, Is.True, stage.error); Assert.That(stage.directory, Is.EqualTo(source));
            Assert.That(calls, Is.Zero); Assert.That(Directory.Exists(destination), Is.False);
            Assert.That(PlayerBundleLoader.Load(stage.directory).Reference.Cases, Has.Length.EqualTo(15));
        }
        [Test] public void JarTransportCopiesOnlyFixedBundleFilesBeforeFilesystemAudit()
        {
            var urls = new List<string>(); var stage = new PlayerBundleStage();
            Drain(PlayerBundleStager.Stage(JarUrl, destination, stage, (url, path) => { urls.Add(url); return CopyFixture(url, path); }));
            Assert.That(stage.transferCompleted, Is.True, stage.error); Assert.That(stage.transferredFiles, Is.EqualTo(6));
            Assert.That(urls, Is.EqualTo(names.Select(name => JarUrl + "/" + name).ToArray()));
            Assert.That(Directory.GetFiles(destination).Select(Path.GetFileName).OrderBy(name => name), Is.EqualTo(names.OrderBy(name => name)));
            Assert.That(PlayerBundleLoader.Load(stage.directory).Reference.Queries, Has.Length.EqualTo(4));
        }
        [UnityTest] public IEnumerator ActualUnityWebRequestStreamsUnicodeFileUrlsToDisk()
        {
            var stage = new PlayerBundleStage();
            yield return PlayerBundleStager.Stage(FileUrl, destination, stage);
            Assert.That(stage.transferCompleted, Is.True, stage.error); Assert.That(stage.transferredFiles, Is.EqualTo(6));
            foreach (var name in names) Assert.That(File.ReadAllBytes(Path.Combine(destination, name)), Is.EqualTo(File.ReadAllBytes(Path.Combine(source, name))), name);
            Assert.That(PlayerBundleLoader.Load(stage.directory).Reference.Cases, Has.Length.EqualTo(15));
        }
        [UnityTest] public IEnumerator ActualMissingFileFailsAndLeavesNoPartialFile()
        {
            File.Delete(Path.Combine(source, "model-float16.sentis"));
            var stage = new PlayerBundleStage();
            yield return PlayerBundleStager.Stage(FileUrl, destination, stage);
            Assert.That(stage.transferCompleted, Is.False); Assert.That(stage.error, Does.Contain("model-float16.sentis"));
            Assert.That(File.Exists(Path.Combine(destination, "model-float16.sentis")), Is.False);
            Assert.That(Directory.GetFiles(destination, "*.part"), Is.Empty);
        }
        [Test] public void InterruptedTransferCannotPromotePartialBytesOrContinue()
        {
            var stage = new PlayerBundleStage(); var calls = 0;
            IEnumerator Fail(string path) { File.WriteAllText(path, "partial"); yield return null; throw new IOException("interrupted"); }
            Drain(PlayerBundleStager.Stage(JarUrl, destination, stage, (_, path) => { calls++; return Fail(path); }));
            Assert.That(stage.transferCompleted, Is.False); Assert.That(stage.error, Does.Contain("interrupted")); Assert.That(calls, Is.EqualTo(1));
            Assert.That(Directory.GetFiles(destination), Is.Empty);
        }
        [Test] public void ExistingDestinationIsNeverOverwritten()
        {
            Directory.CreateDirectory(destination); var file = Path.Combine(destination, "bundle.json"); File.WriteAllText(file, "previous successful data");
            var stage = new PlayerBundleStage(); var calls = 0;
            Drain(PlayerBundleStager.Stage(JarUrl, destination, stage, (url, path) => { calls++; return CopyFixture(url, path); }));
            Assert.That(stage.transferCompleted, Is.False); Assert.That(stage.error, Does.Contain("empty"));
            Assert.That(File.ReadAllText(file), Is.EqualTo("previous successful data")); Assert.That(calls, Is.Zero);
        }
        [TestCase("https://example.invalid/bundle")] [TestCase("jar:https://example.invalid/app.apk!/assets")]
        public void UnsupportedSourcesDoNotCreateAStagingDirectory(string url)
        {
            var stage = new PlayerBundleStage(); var calls = 0;
            Drain(PlayerBundleStager.Stage(url, destination, stage, (_, _) => { calls++; return CopyFixture("/bundle.json", "unused"); }));
            Assert.That(stage.transferCompleted, Is.False); Assert.That(stage.error, Does.Contain("source"));
            Assert.That(calls, Is.Zero); Assert.That(Directory.Exists(destination), Is.False);
        }
        PlayerRunConfiguration Config() => new PlayerRunConfiguration { Bundle = JarUrl, Output = Path.Combine(root, "results.json"), RunId = new string('a', 32) };
        sealed class Provider : IPlayerEmbedder
        {
            public BackendType Backend { get; set; }
            public float[] EmbedRaw(string text) => PlayerValidationTests.Vector();
            public float[] EmbedQuery(string text) => PlayerValidationTests.Vector();
            public float[] EmbedDocument(string text, string title = null) => PlayerValidationTests.Vector();
            public void Dispose() { }
        }
        [Test] public void ExecutionAuditsTheExtractedDirectoryBeforeAllConditions()
        {
            var config = Config(); var snapshots = new List<PlayerRunReport>(); PlayerRunReport result = null; var calls = 0;
            Drain(PlayerValidationExecution.Run(config, PlayerRunProtocolTests.Build(), destination,
                report => snapshots.Add(Newtonsoft.Json.JsonConvert.DeserializeObject<PlayerRunReport>(Newtonsoft.Json.JsonConvert.SerializeObject(report))),
                report => result = report, CopyFixture, PlayerBundleLoader.Load, _ => (new int[128], new int[128]),
                (bundle, _, backend) => { Assert.That(bundle.Directory, Is.EqualTo(destination)); calls++; return new Provider { Backend = backend }; }));
            Assert.That(result.success, Is.True, result.error); Assert.That(result.gpuVerified, Is.False); Assert.That(calls, Is.EqualTo(4));
            Assert.That(config.Bundle, Is.EqualTo(JarUrl), "The requested source must not be lost through mutation.");
            Assert.That(snapshots.First().phase, Is.EqualTo("bundle_staging")); Assert.That(snapshots.Take(snapshots.Count - 1).All(report => !report.success), Is.True);
            Assert.That(result.staging.source, Is.EqualTo(JarUrl)); Assert.That(result.staging.transferredFiles, Is.EqualTo(6));
        }
        [Test] public void FailedStagingSavesFailureAndNeverStartsAuditOrInference()
        {
            PlayerRunReport saved = null, result = null; var calls = 0;
            IEnumerator Fail() { yield return null; throw new IOException("storage unavailable"); }
            Drain(PlayerValidationExecution.Run(Config(), PlayerRunProtocolTests.Build(), destination, report => saved = report, report => result = report,
                (_, _) => Fail(), _ => { calls++; throw new Exception("must not audit"); }));
            Assert.That(result.success, Is.False); Assert.That(result.gpuVerified, Is.False); Assert.That(calls, Is.Zero);
            Assert.That(result.error, Does.Contain("storage unavailable")); Assert.That(saved.phase, Is.EqualTo("failed")); Assert.That(saved.completedUtc, Is.Not.Empty);
        }
        [Test] public void TransferredBytesStillRequireFullHashAuditBeforeInference()
        {
            File.WriteAllText(Path.Combine(source, "model-fp32.sentis"), "oops"); PlayerRunReport result = null; var calls = 0;
            Drain(PlayerValidationExecution.Run(Config(), PlayerRunProtocolTests.Build(), destination, _ => { }, report => result = report, CopyFixture,
                PlayerBundleLoader.Load, _ => (new int[128], new int[128]), (_, _, _) => { calls++; return new Provider(); }));
            Assert.That(result.success, Is.False); Assert.That(result.error, Does.Contain("SHA-256")); Assert.That(calls, Is.Zero);
            Assert.That(result.staging.transferCompleted, Is.True, "Transfer completion must not imply audited inference success.");
        }
        [Test] public void InvalidBuildAndReportWriteFailureCannotStartTransfers()
        {
            var build = PlayerRunProtocolTests.Build(); build.codeCommit = "main"; var calls = 0; PlayerRunReport result = null;
            Drain(PlayerValidationExecution.Run(Config(), build, destination, _ => { }, report => result = report,
                (url, path) => { calls++; return CopyFixture(url, path); }));
            Assert.That(result.success, Is.False); Assert.That(calls, Is.Zero);
            Drain(PlayerValidationExecution.Run(Config(), PlayerRunProtocolTests.Build(), destination, _ => throw new IOException("report locked"), report => result = report,
                (url, path) => { calls++; return CopyFixture(url, path); }));
            Assert.That(result.success, Is.False); Assert.That(result.error, Does.Contain("report locked")); Assert.That(calls, Is.Zero);
        }
    }
}
