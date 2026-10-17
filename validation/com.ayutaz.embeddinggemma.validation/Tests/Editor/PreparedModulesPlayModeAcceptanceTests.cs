using System;
using System.Collections;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;

namespace EmbeddingGemma.Validation.Tests
{
    /// <summary>Delegate unchanged reference tests; qualify consecutive modules separately from isolated runs.</summary>
    public sealed class PreparedModulesPlayModeAcceptanceTests
    {
        [UnityTearDown] public IEnumerator Exit() { if (Application.isPlaying) yield return new ExitPlayMode(); }

        [UnityTest, Timeout(900000)]
        public IEnumerator AllSixPreparedModulesKeepCanonicalReferencesInOneGpuProcess()
        {
            yield return new EnterPlayMode();
            var prefix = Environment.GetEnvironmentVariable("EG_ALL_MODULE_REPORT_PREFIX");
            Assert.That(prefix, Is.Not.Null.And.Not.Empty);
            var output = prefix + "-collection.json";
            Assert.That(File.Exists(output), Is.False);
            var names = new[] { "text", "image-square", "image-landscape", "image-portrait", "audio-1119", "audio-839" };
            var counts = new[] { 4, 2, 2, 2, 5, 1 };
            var variables = new[] { "EG_MEDIA_BUNDLE_MODULE", "EG_MEDIA_EXECUTION_REPORT", "EG_TEXT_QUERY_REPORT" };
            var previous = variables.Select(Environment.GetEnvironmentVariable).ToArray();
            var oldBackground = Application.runInBackground;
            var oldRate = Application.targetFrameRate; var oldVsync = QualitySettings.vSyncCount;
            var cameraObject = new GameObject("Owned consecutive reference camera");
            var target = new RenderTexture(16, 16, 0);
            var camera = cameraObject.AddComponent<Camera>();
            camera.targetTexture = target; camera.clearFlags = CameraClearFlags.SolidColor; camera.cullingMask = 0;
            var report = new JObject { ["success"] = false,
                ["runtime_source"] = Environment.GetEnvironmentVariable("EG_ACCEPTANCE_SOURCE"),
                ["precision"] = GpuMediaExecutionPolicy.CurrentPrecision,
                ["scope"] = "Consecutive modules in one PlayMode process. Delegates original image/audio/text reference tests; not six isolated processes, native UI, public acquisition or a latency comparison.",
                ["modules"] = new JArray() };
            try
            {
                Application.runInBackground = true; Application.targetFrameRate = 60; QualitySettings.vSyncCount = 0;
                for (var index = 0; index < names.Length; index++)
                {
                    var module = names[index]; var path = prefix + "-" + module + ".json";
                    Assert.That(File.Exists(path), Is.False);
                    Environment.SetEnvironmentVariable(variables[0], module);
                    Environment.SetEnvironmentVariable(variables[1], path);
                    Environment.SetEnvironmentVariable(variables[2], path);
                    report["phase"] = module; File.WriteAllText(output, report.ToString());
                    IEnumerator original = module == "text"
                        ? new PreparedTextQueryReferenceTests().ActualGpuTextModuleMatchesAllPinnedQueryVectorsAndReleasesWorkers()
                        : module.StartsWith("audio-", StringComparison.Ordinal)
                            ? new GpuMediaBundleExecutionTests().PreparedAudioModuleUsesOriginalPcmAndPinnedFinalDocumentReference()
                            : new GpuMediaBundleExecutionTests().PreparedImageModuleUsesOriginalPngAndPinnedFinalDocumentReference();
                    // Preserve the canonical per-method limits as well as the overall limit.
                    var limit = module == "text" ? 240 : module.StartsWith("audio-", StringComparison.Ordinal) ? 900 : 480;
                    var timer = Stopwatch.StartNew();
                    try
                    {
                        while (original.MoveNext())
                        {
                            Assert.That(timer.Elapsed.TotalSeconds, Is.LessThan(limit), "Canonical per-module execution bound exceeded.");
                            UnityEditor.EditorApplication.QueuePlayerLoopUpdate(); yield return original.Current;
                        }
                    }
                    finally { (original as IDisposable)?.Dispose(); }
                    timer.Stop();
                    Assert.That(timer.Elapsed.TotalSeconds, Is.LessThan(limit));
                    var moduleReport = JObject.Parse(File.ReadAllText(path));
                    Assert.That((bool)moduleReport["success"] && (bool)moduleReport["session_closed"], Is.True);
                    Assert.That((string)moduleReport["actual_backend"], Is.EqualTo("GPUCompute"));
                    Assert.That((string)moduleReport["module"], Is.EqualTo(module));
                    Assert.That(((JArray)moduleReport["cases"]).Count, Is.EqualTo(counts[index]));
                    using var hash = SHA256.Create();
                    ((JArray)report["modules"]).Add(new JObject { ["module"] = module,
                        ["report"] = Path.GetFileName(path), ["cases"] = counts[index],
                        ["report_sha256"] = BitConverter.ToString(hash.ComputeHash(File.ReadAllBytes(path))).Replace("-", "").ToLowerInvariant(),
                        ["wall_seconds"] = timer.Elapsed.TotalSeconds });
                    File.WriteAllText(output, report.ToString());
                }
                Assert.That(((JArray)report["modules"]).Count, Is.EqualTo(6));
                report["reference_cases"] = ((JArray)report["modules"]).Sum(row => (int)row["cases"]);
                Assert.That((int)report["reference_cases"], Is.EqualTo(16));
                report["phase"] = "closed"; report["success"] = true;
            }
            finally
            {
                for (var index = 0; index < variables.Length; index++) Environment.SetEnvironmentVariable(variables[index], previous[index]);
                camera.targetTexture = null; UnityEngine.Object.DestroyImmediate(cameraObject);
                target.Release(); UnityEngine.Object.DestroyImmediate(target);
                Application.runInBackground = oldBackground; Application.targetFrameRate = oldRate; QualitySettings.vSyncCount = oldVsync;
                File.WriteAllText(output, report.ToString());
            }
        }
    }
}
