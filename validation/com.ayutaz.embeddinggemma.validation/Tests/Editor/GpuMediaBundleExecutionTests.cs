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
        static IDisposable OptionalStageTiming(GpuMediaSession session, JObject report, Action save)
        {
            if (Environment.GetEnvironmentVariable("EG_MEDIA_STAGE_TIMING") != "1") return null;
            var rows = new JArray(); report["stage_measurements"] = rows;
            report["stage_measurement_scope"] = "Instrumented call intervals; process CPU includes other Editor threads; GPU timing is separate and unavailable here";
            return MediaStageTiming.Attach(session, row =>
            {
                row["module"] = session.ModuleName; row["case_phase"] = report["phase"]?.DeepClone();
                rows.Add(row); save();
            });
        }
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
                using var stageTiming = OptionalStageTiming(session, report, Save);
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

        [UnityTest] public IEnumerator PreparedAudioModuleUsesOriginalPcmAndPinnedFinalDocumentReference()
        {
            using var cancellation = new CancellationTokenSource(); var task = RunAudio(cancellation.Token); long peak = 0;
            while (!task.IsCompleted)
            {
                peak = Math.Max(peak, MediaExecutionMemory.Peak()); if (peak > 12L * 1024 * 1024 * 1024) cancellation.Cancel();
                UnityEditor.EditorApplication.QueuePlayerLoopUpdate(); yield return null;
            }
            var output = Environment.GetEnvironmentVariable("EG_MEDIA_EXECUTION_REPORT");
            if (!string.IsNullOrEmpty(output) && File.Exists(output))
            { var report = JObject.Parse(File.ReadAllText(output)); report["host_native_peak_bytes"] = peak; File.WriteAllText(output, report.ToString()); }
            task.GetAwaiter().GetResult(); Assert.That(peak, Is.GreaterThan(0).And.LessThanOrEqualTo(12L * 1024 * 1024 * 1024));
        }
        static async Task RunAudio(CancellationToken token)
        {
            var directory = Environment.GetEnvironmentVariable("EG_MEDIA_BUNDLE_OUTPUT");
            var module = Environment.GetEnvironmentVariable("EG_MEDIA_BUNDLE_MODULE");
            var output = Environment.GetEnvironmentVariable("EG_MEDIA_EXECUTION_REPORT");
            Assert.That(directory, Is.Not.Null.And.Not.Empty); Assert.That(output, Is.Not.Null.And.Not.Empty); Assert.That(File.Exists(output), Is.False);
            var bundle = GpuMediaBundle.Open(directory); var metadata = bundle.Module(module); Assert.That((string)metadata["kind"], Is.EqualTo("audio"));
            var validFrames = (int)metadata["valid_audio_frames"];
            var referencePath = Cache("k/artifacts/audio-final-backbone/audio-embedding-reference.json");
            Assert.That(Hash(referencePath), Is.EqualTo("281d12573b431667e8841033db1c6a10dce4382ef98c9a5345c18044a1194001"));
            var reference = JObject.Parse(File.ReadAllText(referencePath));
            var pcmRoot = Cache("k/artifacts/audio-real-chain/audio-runtime-preprocessing"); var auditPath = Path.Combine(pcmRoot, "audio-validation.json");
            Assert.That(Hash(auditPath), Is.EqualTo("4e325f708e8d103a9de358a411380ddbb51cfe953984e9dcb73e291d919630fa"));
            var audit = JObject.Parse(File.ReadAllText(auditPath)); var rows = reference["cases"].Where(row => (string)row["profile"] == "frames-1119-valid-" + validFrames).ToArray();
            Assert.That(rows.Length, Is.EqualTo(validFrames == 1119 ? 5 : 1));
            foreach (var row in rows)
                foreach (var name in new[] { (string)row["id"] + "-pcm-fp32.bin", (string)row["id"] + "-samples-int32.bin" })
                {
                    Assert.That(new FileInfo(Path.Combine(pcmRoot, name)).Length, Is.EqualTo((long)audit["files"][name]["bytes"]));
                    Assert.That(Hash(Path.Combine(pcmRoot, name)), Is.EqualTo((string)audit["files"][name]["sha256"]));
                }
            var report = new JObject { ["success"] = false, ["scope"] = "prepared_audio_module_original_pcm_final_document_768",
                ["module"] = module, ["execution_source"] = Environment.GetEnvironmentVariable("EG_ACCEPTANCE_SOURCE"), ["reference_sha256"] = Hash(referencePath),
                ["bundle_descriptor_sha256"] = Hash(Path.Combine(directory, "bundle.json")), ["actual_backend"] = "GPUCompute", ["cases"] = new JArray(),
                ["encoder_full_reference_comparison"] = false, ["fixed_document_cosine_minimum"] = .999 };
            void Save() => File.WriteAllText(output, report.ToString()); GpuMediaSession session = null;
            try
            {
                report["phase"] = "load_verified_prepared_module"; Save(); session = await bundle.LoadAsync(module, token);
                using var stageTiming = OptionalStageTiming(session, report, Save);
                foreach (var row in rows)
                {
                    token.ThrowIfCancellationRequested(); var id = (string)row["id"];
                    var pcmBytes = File.ReadAllBytes(Path.Combine(pcmRoot, id + "-pcm-fp32.bin")); var maskBytes = File.ReadAllBytes(Path.Combine(pcmRoot, id + "-samples-int32.bin"));
                    var waveform = new float[pcmBytes.Length / 4]; var mask = new int[maskBytes.Length / 4];
                    Buffer.BlockCopy(pcmBytes, 0, waveform, 0, pcmBytes.Length); Buffer.BlockCopy(maskBytes, 0, mask, 0, maskBytes.Length);
                    var count = mask.Count(value => value == 1); Assert.That(waveform.Length, Is.EqualTo(179200));
                    Assert.That(mask, Is.EqualTo(Enumerable.Repeat(1, count).Concat(Enumerable.Repeat(0, 179200 - count)).ToArray()));
                    Assert.That(waveform.Skip(count), Is.All.EqualTo(0)); var input = new AudioPcmInput(waveform.Take(count).ToArray()); Assert.That(input.ValidFrameCount, Is.EqualTo(validFrames));
                    report["phase"] = "encode_" + id; Save(); var actual = await session.EmbedAudioAsync(input, cancellationToken: token);
                    var expected = row["embedding"].Values<float>().ToArray(); var cosine = TextSearchIndex.Cosine(actual, expected); var norm = actual.Sum(value => (double)value * value);
                    var capture = Path.ChangeExtension(output, null) + "-" + id + ".bin"; Assert.That(File.Exists(capture), Is.False);
                    var bytes = new byte[actual.Length * 4]; Buffer.BlockCopy(actual, 0, bytes, 0, bytes.Length); File.WriteAllBytes(capture, bytes);
                    ((JArray)report["cases"]).Add(new JObject { ["id"] = id, ["cosine"] = cosine, ["norm_squared"] = norm, ["valid_audio_frames"] = validFrames,
                        ["capture"] = Path.GetFileName(capture), ["capture_sha256"] = Hash(capture), ["original_pcm_sha256"] = Hash(Path.Combine(pcmRoot, id + "-pcm-fp32.bin")) }); Save();
                    Assert.That(cosine, Is.GreaterThanOrEqualTo(.999)); Assert.That(Math.Abs(norm - 1), Is.LessThanOrEqualTo(.002));
                }
                report["phase"] = "close"; Save(); await session.CloseAsync(); MultimodalSearchSampleGpuTests.AssertReleased(session);
                report["closed_worker_kernel_and_tensor_fields_inspected"] = true; report["session_closed"] = true; report["success"] = true;
            }
            catch (Exception error) { report["error"] = error.ToString(); throw; }
            finally { session?.Dispose(); Save(); }
        }
    }
}
