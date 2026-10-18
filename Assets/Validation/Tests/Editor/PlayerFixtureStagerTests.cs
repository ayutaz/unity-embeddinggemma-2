using System;
using System.Collections;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using UnityEngine.TestTools;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class PlayerFixtureStagerTests
    {
        string root, destination, source;
        static readonly byte[] Data = { 1, 2, 3, 4 };
        [SetUp] public void SetUp()
        {
            root = Path.Combine(Path.GetTempPath(), "embeddinggemma-fixture-stage-" + Guid.NewGuid().ToString("N"));
            source = Path.Combine(root, "source"); Directory.CreateDirectory(source); destination = Path.Combine(root, "staged");
        }
        [TearDown] public void TearDown() { if (Directory.Exists(root)) Directory.Delete(root, true); }
        static string Sha(byte[] bytes) { using var hash = SHA256.Create(); return BitConverter.ToString(hash.ComputeHash(bytes)).Replace("-", "").ToLowerInvariant(); }
        static JObject Row(string name = "audio/zero.bin") => new JObject { ["path"] = name, ["bytes"] = Data.Length, ["sha256"] = Sha(Data) };
        static JObject Manifest() => new JObject { ["schema_version"] = 1, ["files"] = new JArray(Row()) };
        static byte[] Bytes(JObject manifest) => Encoding.UTF8.GetBytes(manifest.ToString());
        IEnumerator Start(JObject report, Func<string, string, IEnumerator> transfer = null, JObject manifest = null, string location = null, string digest = null)
        {
            var type = typeof(PlayerBundleStager).Assembly.GetType("EmbeddingGemma.Validation.PlayerFixtureStager");
            Assert.That(type, Is.Not.Null, "Device fixtures need a pinned local materialization entry.");
            var bytes = Bytes(manifest ?? Manifest());
            return (IEnumerator)type.GetMethod("Stage", BindingFlags.Public | BindingFlags.Static).Invoke(null,
                new object[] { location ?? "jar:file:///validation.apk!/assets/fixtures", destination, bytes, digest ?? Sha(bytes), report, transfer });
        }
        static void Drain(IEnumerator routine) { try { while (routine.MoveNext()) { if (routine.Current is IEnumerator nested) Drain(nested); } } finally { (routine as IDisposable)?.Dispose(); } }
        static IEnumerator Transfer(string _, string output) { File.WriteAllBytes(output, Data); yield return null; }

        [Test] public void VerifiedFilesBecomeLocalWithoutGrantingNumericalOrGpuAcceptance()
        {
            var report = new JObject(); var manifest = Manifest(); ((JArray)manifest["files"]).Add(Row("roles/raw/reference.json"));
            Drain(Start(report, Transfer, manifest));
            Assert.That((bool)report["success"], Is.True); Assert.That((string)report["phase"], Is.EqualTo("complete"));
            Assert.That((string)report["scope"], Is.EqualTo("local_streaming_fixture_materialization"));
            Assert.That((string)report["transport"], Is.EqualTo("injected_transport"));
            Assert.That((int)report["verified_files"], Is.EqualTo(2)); Assert.That((long)report["verified_bytes"], Is.EqualTo(8));
            Assert.That((bool)report["original_fixture_validated"], Is.False); Assert.That((bool)report["gpu_verified"], Is.False);
            Assert.That(File.ReadAllBytes(Path.Combine(destination, "audio/zero.bin")), Is.EqualTo(Data));
            Assert.That(File.ReadAllBytes(Path.Combine(destination, "roles/raw/reference.json")), Is.EqualTo(Data));
            Assert.That(report.ToString(), Does.Not.Contain(root)); Assert.That(Directory.GetFiles(destination, "*.part", SearchOption.AllDirectories), Is.Empty);
        }
        [UnityTest] public IEnumerator RealLocalFileTransportVerifiesTheDownloadedBytes()
        {
            Directory.CreateDirectory(Path.Combine(source, "audio")); File.WriteAllBytes(Path.Combine(source, "audio/zero.bin"), Data);
            var report = new JObject(); yield return Start(report, location: new Uri(source + Path.DirectorySeparatorChar).AbsoluteUri.TrimEnd('/'));
            Assert.That((bool)report["success"], Is.True); Assert.That((string)report["transport"], Is.EqualTo("UnityWebRequest.DownloadHandlerFile"));
            Assert.That((string)report["files"][0]["sha256"], Is.EqualTo(Sha(Data))); Assert.That((bool)report["gpu_verified"], Is.False);
        }
        [TestCase("../escape.bin")] [TestCase("a/../escape.bin")] [TestCase("/escape.bin")] [TestCase("a\\b.bin")]
        [TestCase("https://remote/file.bin")] [TestCase("a//b.bin")] [TestCase("a/.")] [TestCase("a/file.part")] [TestCase("a/model.onnx")]
        public void UnsafeOrNonFixturePathsFailBeforeTransfer(string name)
        {
            var manifest = Manifest(); manifest["files"][0]["path"] = name; var called = false;
            IEnumerator Unexpected(string _, string __) { called = true; yield break; }
            Assert.Throws<ArgumentException>(() => Drain(Start(new JObject(), Unexpected, manifest)));
            Assert.That(called, Is.False); Assert.That(Directory.Exists(destination), Is.False); Assert.That(File.Exists(Path.Combine(root, "escape.bin")), Is.False);
        }
        [TestCase("digest")] [TestCase("duplicate")] [TestCase("empty")] [TestCase("oversize")] [TestCase("unknown_field")] [TestCase("hierarchy")]
        public void ManifestMustBePinnedBoundedUniqueAndUnambiguous(string change)
        {
            var manifest = Manifest();
            if (change == "duplicate") ((JArray)manifest["files"]).Add(Row("AUDIO/zero.bin"));
            if (change == "empty") manifest["files"] = new JArray();
            if (change == "oversize") manifest["files"][0]["bytes"] = 64L * 1024 * 1024 + 1;
            if (change == "unknown_field") manifest["unexpected"] = true;
            if (change == "hierarchy") manifest["files"] = new JArray(Row("a.json"), Row("a.json/b.bin"));
            var called = false; IEnumerator Unexpected(string _, string __) { called = true; yield break; }
            Assert.Throws<ArgumentException>(() => Drain(Start(new JObject(), Unexpected, manifest, digest: change == "digest" ? new string('0', 64) : null)));
            Assert.That(called, Is.False); Assert.That(Directory.Exists(destination), Is.False);
        }
        [TestCase("https://example.invalid/fixtures")] [TestCase("http://example.invalid/fixtures")]
        [TestCase("file://remote-share/fixtures")] [TestCase("jar:https://example.invalid/archive!/fixtures")] [TestCase("relative/fixtures")]
        public void OnlyLocalStreamingSourcesAreAllowed(string location)
        {
            Assert.Throws<NotSupportedException>(() => Drain(Start(new JObject(), Transfer, location: location)));
            Assert.That(Directory.Exists(destination), Is.False);
        }
        [TestCase(false)] [TestCase(true)] public void CorruptOrTruncatedTransfersNeverPublishTheFinalName(bool wrongSize)
        {
            var report = new JObject();
            IEnumerator Corrupt(string _, string output) { File.WriteAllBytes(output, wrongSize ? new byte[] { 1 } : new byte[] { 9, 2, 3, 4 }); yield return null; }
            Drain(Start(report, Corrupt)); Assert.That((bool)report["success"], Is.False); Assert.That((string)report["error_type"], Is.EqualTo("InvalidDataException"));
            Assert.That((int)report["verified_files"], Is.Zero); Assert.That(File.Exists(Path.Combine(destination, "audio/zero.bin")), Is.False);
            Assert.That(Directory.GetFiles(destination, "*.part", SearchOption.AllDirectories), Is.Empty);
        }
        [TestCase(false)] [TestCase(true)] public void ExistingDestinationsAreNeverReusedOrOverwritten(bool asFile)
        {
            if (asFile) File.WriteAllText(destination, "previous evidence");
            else { Directory.CreateDirectory(destination); File.WriteAllText(Path.Combine(destination, "keep.json"), "previous evidence"); }
            Assert.Throws<IOException>(() => Drain(Start(new JObject(), Transfer)));
            Assert.That(File.ReadAllText(asFile ? destination : Path.Combine(destination, "keep.json")), Is.EqualTo("previous evidence"));
        }
        [Test] public void ExistingReportObjectIsNeverClearedOrReused()
        {
            var report = new JObject { ["prior_evidence"] = true };
            Assert.Throws<ArgumentException>(() => Drain(Start(report, Transfer)));
            Assert.That(report.ToString(), Does.Contain("prior_evidence")); Assert.That(Directory.Exists(destination), Is.False);
        }
        [Test] public void DisposalCancelsOwnedTransferAndRemovesOnlyItsPartial()
        {
            var report = new JObject(); var disposed = false;
            IEnumerator Wait(string _, string output) { try { File.WriteAllBytes(output, Data); yield return null; yield return null; } finally { disposed = true; } }
            var routine = Start(report, Wait); Assert.That(routine.MoveNext(), Is.True); (routine as IDisposable).Dispose();
            Assert.That(disposed, Is.True); Assert.That((bool)report["success"], Is.False); Assert.That((string)report["phase"], Is.EqualTo("cancelled"));
            Assert.That(File.Exists(Path.Combine(destination, "audio/zero.bin")), Is.False); Assert.That(Directory.GetFiles(destination, "*.part", SearchOption.AllDirectories), Is.Empty);
        }
        [Test] public void TransferExceptionsRecordTypesWithoutPublishingPrivateMessages()
        {
            var report = new JObject(); IEnumerator Fail(string _, string output) { File.WriteAllBytes(output, Data); yield return null; throw new IOException(root); }
            Drain(Start(report, Fail)); Assert.That((bool)report["success"], Is.False); Assert.That((string)report["phase"], Is.EqualTo("failed"));
            Assert.That((string)report["error_type"], Is.EqualTo("IOException")); Assert.That(report.ToString(), Does.Not.Contain(root));
            Assert.That(Directory.GetFiles(destination, "*.part", SearchOption.AllDirectories), Is.Empty);
        }
    }
}
