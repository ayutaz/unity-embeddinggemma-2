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
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.TestTools;

namespace EmbeddingGemma.Validation.Tests
{
    /// <summary>Actual prepared text module, pinned query vectors; no CPU inference fallback or skip.</summary>
    public sealed class PreparedTextQueryReferenceTests
    {
        static string Hash(string path)
        { using var stream = File.OpenRead(path); using var hash = SHA256.Create(); return BitConverter.ToString(hash.ComputeHash(stream)).Replace("-", "").ToLowerInvariant(); }
        [UnityTest, Timeout(240000)] public IEnumerator ActualGpuTextModuleMatchesAllPinnedQueryVectorsAndReleasesWorkers()
        {
            using var cancellation = new CancellationTokenSource(); var task = Run(cancellation.Token); long peak = 0;
            while (!task.IsCompleted)
            {
                peak = Math.Max(peak, MediaExecutionMemory.Peak()); if (peak > GpuMediaExecutionPolicy.CurrentMemoryLimitBytes) cancellation.Cancel();
                UnityEditor.EditorApplication.QueuePlayerLoopUpdate(); yield return null;
            }
            task.GetAwaiter().GetResult();
            Assert.That(peak, Is.GreaterThan(0).And.LessThanOrEqualTo(GpuMediaExecutionPolicy.CurrentMemoryLimitBytes));
        }
        static async Task Run(CancellationToken token)
        {
            var directory = Environment.GetEnvironmentVariable("EG_MEDIA_BUNDLE_OUTPUT");
            var output = Environment.GetEnvironmentVariable("EG_TEXT_QUERY_REPORT");
            Assert.That(directory, Is.Not.Null.And.Not.Empty); Assert.That(output, Is.Not.Null.And.Not.Empty);
            Assert.That(File.Exists(output), Is.False);
            var descriptor = Path.Combine(directory, "bundle.json");
            GpuMediaExecutionPolicy.VerifyCurrent(directory);
            var referencePath = Path.GetFullPath(Path.Combine(Application.dataPath, "../../player-bundle/windows-unity21-20261012/search-reference.json"));
            Assert.That(Hash(referencePath), Is.EqualTo("f316d5199c920431c7f721d43b2cedbe4a6384fb0e0664511b75bc2e98fe80fe"));
            var queries = JObject.Parse(File.ReadAllText(referencePath))["queries"];
            var report = new JObject { ["success"] = false, ["module"] = "text", ["actual_backend"] = "GPUCompute",
                ["precision"] = GpuMediaExecutionPolicy.CurrentPrecision, ["host_memory_limit_bytes"] = GpuMediaExecutionPolicy.CurrentMemoryLimitBytes,
                ["execution_source"] = Environment.GetEnvironmentVariable("EG_ACCEPTANCE_SOURCE"), ["reference_sha256"] = Hash(referencePath),
                ["bundle_descriptor_sha256"] = Hash(descriptor), ["cases"] = new JArray(), ["stage_measurements"] = new JArray(),
                ["phase"] = "load", ["gpu_time_measured"] = false };
            void Save() => File.WriteAllText(output, report.ToString()); Save(); GpuMediaSession session = null;
            try
            {
                var load = Stopwatch.StartNew(); session = await GpuMediaBundle.Open(directory).LoadAsync("text", token);
                report["load_wall_seconds"] = load.Elapsed.TotalSeconds; Assert.That(session.Backend, Is.EqualTo(BackendType.GPUCompute));
                var measured = MediaStageTiming.Wrap<ITextEmbedder>(session, "text_query", row => { ((JArray)report["stage_measurements"]).Add(row); Save(); });
                foreach (var query in queries)
                {
                    token.ThrowIfCancellationRequested(); var id = (string)query["id"]; report["phase"] = "query_" + id; Save();
                    var actual = measured.EmbedQuery((string)query["text"]); var expected = query["embedding"].Values<float>().ToArray();
                    Assert.That(actual.Length, Is.EqualTo(768)); Assert.That(actual.All(v => !float.IsNaN(v) && !float.IsInfinity(v)), Is.True);
                    var cosine = TextSearchIndex.Cosine(actual, expected); var norm = actual.Sum(v => (double)v * v);
                    Assert.That(cosine, Is.GreaterThanOrEqualTo(GpuMediaExecutionPolicy.CurrentMinimumCosine)); Assert.That(Math.Abs(norm - 1), Is.LessThanOrEqualTo(.002));
                    var capture = Path.ChangeExtension(output, null) + "-" + id + ".bin"; Assert.That(File.Exists(capture), Is.False);
                    var bytes = new byte[actual.Length * 4]; Buffer.BlockCopy(actual, 0, bytes, 0, bytes.Length); File.WriteAllBytes(capture, bytes);
                    ((JArray)report["cases"]).Add(new JObject { ["id"] = id, ["cosine"] = cosine, ["norm_squared"] = norm,
                        ["capture"] = Path.GetFileName(capture), ["capture_sha256"] = Hash(capture) }); Save(); await Task.Yield();
                }
                Assert.That(((JArray)report["cases"]).Count, Is.EqualTo(queries.Count()));
                report["phase"] = "close"; Save(); await session.CloseAsync(); MultimodalSearchSampleGpuTests.AssertReleased(session);
                report["session_closed"] = true; report["closed_worker_fields_inspected"] = true;
                report["host_native_lifetime_peak_bytes"] = MediaExecutionMemory.Peak(); report["success"] = true;
            }
            catch (Exception error) { report["error"] = error.ToString(); throw; }
            finally { session?.Dispose(); Save(); }
        }
    }
}
