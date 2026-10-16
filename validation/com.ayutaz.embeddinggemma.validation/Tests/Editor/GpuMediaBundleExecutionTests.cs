using System;
using System.Collections;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;

namespace EmbeddingGemma.Validation.Tests
{
    /// <summary>Opt-in real prepared-module execution; missing artifacts fail rather than skip.</summary>
    public sealed class GpuMediaBundleExecutionTests
    {
        static string Cache(string path) => Path.GetFullPath(Path.Combine(Application.dataPath, "../../", path));
        static string Hash(string path)
        { using var stream = File.OpenRead(path); using var hash = SHA256.Create(); return BitConverter.ToString(hash.ComputeHash(stream)).Replace("-", "").ToLowerInvariant(); }

        [UnityTest] public IEnumerator PreparedImageModuleUsesOriginalPngAndPinnedFinalDocumentReference()
        {
            using var cancellation = new CancellationTokenSource();
            var task = RunImage(cancellation.Token); long peak = 0;
            while (!task.IsCompleted)
            {
                peak = Math.Max(peak, MediaExecutionMemory.Peak());
                if (peak > 12L * 1024 * 1024 * 1024) cancellation.Cancel();
                UnityEditor.EditorApplication.QueuePlayerLoopUpdate(); yield return null;
            }
            var output = Environment.GetEnvironmentVariable("EG_MEDIA_EXECUTION_REPORT");
            if (!string.IsNullOrEmpty(output) && File.Exists(output))
            {
                var report = JObject.Parse(File.ReadAllText(output)); report["host_native_peak_bytes"] = peak;
                File.WriteAllText(output, report.ToString());
            }
            task.GetAwaiter().GetResult(); Assert.That(peak, Is.GreaterThan(0).And.LessThanOrEqualTo(12L * 1024 * 1024 * 1024));
        }
        static async Task RunImage(CancellationToken token)
        {
            var directory = Environment.GetEnvironmentVariable("EG_MEDIA_BUNDLE_OUTPUT");
            var module = Environment.GetEnvironmentVariable("EG_MEDIA_BUNDLE_MODULE");
            var output = Environment.GetEnvironmentVariable("EG_MEDIA_EXECUTION_REPORT");
            Assert.That(directory, Is.Not.Null.And.Not.Empty); Assert.That(output, Is.Not.Null.And.Not.Empty);
            Assert.That(File.Exists(output), Is.False, "Preserve previous execution evidence.");
            var bundle = GpuMediaBundle.Open(directory); var metadata = bundle.Module(module);
            Assert.That((string)metadata["kind"], Is.EqualTo("image"));
            var report = new JObject { ["success"] = false, ["module"] = module,
                ["execution_source"] = Environment.GetEnvironmentVariable("EG_ACCEPTANCE_SOURCE"),
                ["bundle_descriptor_sha256"] = Hash(Path.Combine(directory, "bundle.json")),
                ["bundle_producer_source"] = JObject.Parse(File.ReadAllText(Path.Combine(directory, "bundle.json")))["source_commit"],
                ["actual_backend"] = "GPUCompute", ["cases"] = new JArray() };
            void Save() => File.WriteAllText(output, report.ToString());
            GpuMediaSession session = null;
            try
            {
                report["phase"] = "load_verified_prepared_module"; Save(); session = await bundle.LoadAsync(module, token);
                var referencePath = Cache("numerical-acceptance-state/image-role-reference-real/image-role-reference/document/image-embedding-reference.json");
                Assert.That(Hash(referencePath), Is.EqualTo("90692ad38c6d7d844dc6b6e79c7d5cfa719ea59866946dbf94a40bca10ac9eac"));
                var reference = JObject.Parse(File.ReadAllText(referencePath));
                var imagesRoot = Cache("k/artifacts/vision-encoder");
                var source = JObject.Parse(File.ReadAllText(Path.Combine(imagesRoot, "vision-reference.json")));
                var audit = JObject.Parse(File.ReadAllText(Path.Combine(imagesRoot, "vision-validation.json")));
                foreach (var row in source["cases"].Where(row => "image-" + (string)row["profile"] == module))
                {
                    var id = (string)row["id"]; var image = (string)row["image"]; var imagePath = Path.Combine(imagesRoot, image);
                    Assert.That(Hash(imagePath), Is.EqualTo((string)audit["files"][image]["sha256"]));
                    var expected = reference["cases"].Single(value => (string)value["id"] == id)["embedding"].Values<float>().ToArray();
                    var texture = new Texture2D(2, 2, TextureFormat.RGBA32, false, true);
                    try
                    {
                        Assert.That(ImageConversion.LoadImage(texture, File.ReadAllBytes(imagePath), true), Is.True);
                        report["phase"] = "encode_" + id; Save(); var actual = await session.EmbedImageAsync(texture, cancellationToken: token);
                        Assert.That(actual.Length, Is.EqualTo(768));
                        var norm = actual.Sum(value => (double)value * value);
                        var cosine = actual.Select((value, index) => (double)value * expected[index]).Sum() /
                            Math.Sqrt(norm * expected.Sum(value => (double)value * value));
                        var capture = Path.ChangeExtension(output, null) + "-" + id + ".bin"; Assert.That(File.Exists(capture), Is.False);
                        var bytes = new byte[actual.Length * 4]; Buffer.BlockCopy(actual, 0, bytes, 0, bytes.Length); File.WriteAllBytes(capture, bytes);
                        ((JArray)report["cases"]).Add(new JObject { ["id"] = id, ["cosine"] = cosine, ["norm_squared"] = norm,
                            ["capture"] = Path.GetFileName(capture), ["capture_sha256"] = Hash(capture), ["input_png_sha256"] = Hash(imagePath),
                            ["reference_sha256"] = Hash(referencePath), ["acceptance_cosine_minimum"] = .999 }); Save();
                        Assert.That(cosine, Is.GreaterThanOrEqualTo(.999)); Assert.That(Math.Abs(norm - 1), Is.LessThanOrEqualTo(.002));
                    }
                    finally { UnityEngine.Object.DestroyImmediate(texture); }
                }
                Assert.That(((JArray)report["cases"]).Count, Is.EqualTo(2));
                report["phase"] = "close"; Save(); await session.CloseAsync(); Assert.That(session.IsClosed, Is.True);
                report["session_closed"] = true; report["success"] = true;
            }
            catch (Exception error) { report["error"] = error.ToString(); throw; }
            finally { session?.Dispose(); Save(); }
        }
    }
}
