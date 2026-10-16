using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Security.Cryptography;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;

namespace EmbeddingGemma.Validation.Tests
{
    internal static class MediaExecutionMemory
    {
        [StructLayout(LayoutKind.Sequential)] struct Counters
        {
            public uint size, pageFaults;
            public UIntPtr peakWorkingSet, workingSet, peakPagedPool, pagedPool, peakNonPagedPool, nonPagedPool, pageFile, peakPageFile;
        }
        [DllImport("kernel32.dll")] static extern IntPtr GetCurrentProcess();
        [DllImport("psapi.dll", SetLastError = true)] static extern bool GetProcessMemoryInfo(IntPtr process, ref Counters counters, uint size);
        internal static long Peak()
        {
            long peak;
            if (SystemInfo.operatingSystemFamily == UnityEngine.OperatingSystemFamily.Windows)
            {
                var counters = new Counters { size = (uint)Marshal.SizeOf<Counters>() };
                if (!GetProcessMemoryInfo(GetCurrentProcess(), ref counters, counters.size)) throw new InvalidOperationException("Native memory counters unavailable.");
                peak = (long)counters.peakWorkingSet.ToUInt64();
            }
            else { using var process = Process.GetCurrentProcess(); process.Refresh(); peak = process.PeakWorkingSet64; }
            Assert.That(peak, Is.GreaterThan(0), "Missing memory counters cannot satisfy a memory guard."); return peak;
        }
    }
    /// <summary>Opt-in real imported sample, model cache, module switching and pinned cross-modal search.</summary>
    public sealed class MultimodalSearchSampleGpuTests
    {
        const BindingFlags Private = BindingFlags.Instance | BindingFlags.NonPublic;
        static string Cache(string path) => Path.GetFullPath(Path.Combine(Application.dataPath, "../../", path));
        static string Hash(string path)
        { using var stream = File.OpenRead(path); using var hash = SHA256.Create(); return BitConverter.ToString(hash.ComputeHash(stream)).Replace("-", "").ToLowerInvariant(); }
        static Type SampleType() => AppDomain.CurrentDomain.GetAssemblies().Select(assembly => assembly.GetType("EmbeddingGemma.Samples.MultimodalSearchSample")).Single(type => type != null);
        static T Property<T>(Component sample, string name) => (T)sample.GetType().GetProperty(name).GetValue(sample);
        static void Set(Component sample, string field, object value) => sample.GetType().GetField(field).SetValue(sample, value);
        static Task Call(Component sample, string method) => (Task)sample.GetType().GetMethod(method).Invoke(sample, method == "CloseAsync" ? Array.Empty<object>() : new object[] { CancellationToken.None });

        [UnityTest] public IEnumerator ActualImportedSampleRegistersMediaSearchesPinnedQueriesAndClosesItsCacheLease()
        {
            using var cancellation = new CancellationTokenSource(); long peak = 0; var task = Run(cancellation.Token);
            while (!task.IsCompleted)
            {
                peak = Math.Max(peak, MediaExecutionMemory.Peak());
                if (peak > 12L * 1024 * 1024 * 1024) cancellation.Cancel();
                UnityEditor.EditorApplication.QueuePlayerLoopUpdate(); yield return null;
            }
            var output = Environment.GetEnvironmentVariable("EG_MEDIA_SAMPLE_REPORT");
            if (!string.IsNullOrEmpty(output) && File.Exists(output))
            { var report = JObject.Parse(File.ReadAllText(output)); report["host_native_peak_bytes"] = peak; File.WriteAllText(output, report.ToString()); }
            task.GetAwaiter().GetResult(); Assert.That(peak, Is.GreaterThan(0).And.LessThanOrEqualTo(12L * 1024 * 1024 * 1024));
        }

        [UnityTest] public IEnumerator ImportedSampleRendersItsRealSearchResultsInPlayMode()
        {
            Assert.That(Environment.GetEnvironmentVariable("EG_MEDIA_SAMPLE_SCREENSHOT"), Is.Not.Null.And.Not.Empty);
            yield return new EnterPlayMode();
            using var cancellation = new CancellationTokenSource(); var task = Run(cancellation.Token); long peak = 0;
            while (!task.IsCompleted)
            {
                peak = Math.Max(peak, MediaExecutionMemory.Peak()); if (peak > 12L * 1024 * 1024 * 1024) cancellation.Cancel();
                yield return null;
            }
            var path = Environment.GetEnvironmentVariable("EG_MEDIA_SAMPLE_REPORT"); var report = JObject.Parse(File.ReadAllText(path));
            report["host_native_peak_bytes"] = peak; report["play_mode_execution"] = true; File.WriteAllText(path, report.ToString());
            task.GetAwaiter().GetResult(); Assert.That(peak, Is.GreaterThan(0).And.LessThanOrEqualTo(12L * 1024 * 1024 * 1024));
            yield return new ExitPlayMode();
        }

        static async Task Run(CancellationToken cancellation)
        {
            var bundlePath = Environment.GetEnvironmentVariable("EG_MEDIA_BUNDLE_OUTPUT");
            var cachePath = Environment.GetEnvironmentVariable("EG_MEDIA_SAMPLE_CACHE");
            var output = Environment.GetEnvironmentVariable("EG_MEDIA_SAMPLE_REPORT");
            var selections = (Environment.GetEnvironmentVariable("EG_MEDIA_SAMPLE_CASES") ?? "image-square:square-0,image-square:square-1").Split(',');
            var expectedWarm = Environment.GetEnvironmentVariable("EG_MEDIA_SAMPLE_WARM") == "1";
            Assert.That(bundlePath, Is.Not.Null.And.Not.Empty); Assert.That(cachePath, Is.Not.Null.And.Not.Empty); Assert.That(output, Is.Not.Null.And.Not.Empty);
            Assert.That(File.Exists(output), Is.False); var report = new JObject { ["success"] = false, ["scope"] = "actual_imported_sample_real_media_fixed_queries",
                ["runtime_source"] = Environment.GetEnvironmentVariable("EG_ACCEPTANCE_SOURCE"), ["cache_expected_warm"] = expectedWarm,
                ["unity"] = Application.unityVersion, ["gpu"] = SystemInfo.graphicsDeviceName, ["graphics_api"] = SystemInfo.graphicsDeviceType.ToString(),
                ["bundle_descriptor_sha256"] = Hash(Path.Combine(bundlePath, "bundle.json")), ["registrations"] = new JArray(), ["queries"] = new JArray() };
            void Save() => File.WriteAllText(output, report.ToString());
            var host = new GameObject("Actual GPU media search acceptance"); Component sample = null; GpuMediaSession previous = null;
            try
            {
                sample = host.AddComponent(SampleType()); Set(sample, "SourceDirectory", bundlePath); Set(sample, "CacheDirectory", cachePath);
                var expectedVectors = new List<float[]>();
                foreach (var selection in selections)
                {
                    cancellation.ThrowIfCancellationRequested(); var pair = selection.Split(':'); var module = pair[0]; var id = pair[1]; string path; float[] expected;
                    if (module.StartsWith("image-", StringComparison.Ordinal))
                    {
                        var referencePath = Cache("numerical-acceptance-state/image-role-reference-real/image-role-reference/document/image-embedding-reference.json");
                        Assert.That(Hash(referencePath), Is.EqualTo("90692ad38c6d7d844dc6b6e79c7d5cfa719ea59866946dbf94a40bca10ac9eac"));
                        var reference = JObject.Parse(File.ReadAllText(referencePath)); expected = reference["cases"].Single(row => (string)row["id"] == id)["embedding"].Values<float>().ToArray();
                        var images = Cache("k/artifacts/vision-encoder"); var source = JObject.Parse(File.ReadAllText(Path.Combine(images, "vision-reference.json")));
                        var image = (string)source["cases"].Single(row => (string)row["id"] == id)["image"]; path = Path.Combine(images, image);
                        var audit = JObject.Parse(File.ReadAllText(Path.Combine(images, "vision-validation.json"))); Assert.That(Hash(path), Is.EqualTo((string)audit["files"][image]["sha256"]));
                    }
                    else
                    {
                        var referencePath = Cache("k/artifacts/audio-final-backbone/audio-embedding-reference.json");
                        Assert.That(Hash(referencePath), Is.EqualTo("281d12573b431667e8841033db1c6a10dce4382ef98c9a5345c18044a1194001"));
                        var reference = JObject.Parse(File.ReadAllText(referencePath)); expected = reference["cases"].Single(row => (string)row["id"] == id)["embedding"].Values<float>().ToArray();
                        var pcmRoot = Cache("k/artifacts/audio-real-chain/audio-runtime-preprocessing"); var pcmName = id + "-waveform-fp32.bin"; var maskName = id + "-samples-int32.bin";
                        var audit = JObject.Parse(File.ReadAllText(Path.Combine(pcmRoot, "audio-validation.json")));
                        foreach (var name in new[] { pcmName, maskName }) Assert.That(Hash(Path.Combine(pcmRoot, name)), Is.EqualTo((string)audit["files"][name]["sha256"]));
                        var maskBytes = File.ReadAllBytes(Path.Combine(pcmRoot, maskName)); var mask = new int[maskBytes.Length / 4]; Buffer.BlockCopy(maskBytes, 0, mask, 0, maskBytes.Length);
                        var count = mask.Count(value => value == 1); Assert.That(mask, Is.EqualTo(Enumerable.Repeat(1, count).Concat(Enumerable.Repeat(0, 179200 - count)).ToArray()));
                        path = Path.ChangeExtension(output, null) + "-" + id + ".f32"; Assert.That(File.Exists(path), Is.False);
                        var pcm = File.ReadAllBytes(Path.Combine(pcmRoot, pcmName)); File.WriteAllBytes(path, pcm.Take(count * 4).ToArray());
                    }
                    Set(sample, "ModuleName", module); Set(sample, "MediaPath", path); report["phase"] = "register_" + id; Save();
                    if (expectedVectors.Count == 0 && Environment.GetEnvironmentVariable("EG_MEDIA_SAMPLE_CANCEL_FIRST") == "1")
                    {
                        var cancelled = Call(sample, "AddAsync");
                        while (!cancelled.IsCompleted && !Property<string>(sample, "Status").Contains("GPUで登録")) await Task.Yield();
                        Assert.That(cancelled.IsCompleted, Is.False, "Cancellation must reach an actual GPU-stage yield.");
                        sample.GetType().GetMethod("Cancel").Invoke(sample, null);
                        try { await cancelled; Assert.Fail("The GPU-stage cancellation must be observed."); } catch (OperationCanceledException) { }
                        Assert.That(Property<int>(sample, "DocumentCount"), Is.Zero);
                        report["cancelled_after_real_gpu_stage_yield"] = true; Save();
                    }
                    var pending = Call(sample, "AddAsync"); using (cancellation.Register(() => sample.GetType().GetMethod("Cancel").Invoke(sample, null))) await pending;
                    var current = (GpuMediaSession)sample.GetType().GetField("session", Private).GetValue(sample);
                    if (previous != null && !ReferenceEquals(previous, current)) Assert.That(previous.IsClosed, Is.True, "Switch must close previous GPU owners."); previous = current;
                    var vectors = (List<float[]>)sample.GetType().GetField("vectors", Private).GetValue(sample); var vector = vectors.Last();
                    var cosine = TextSearchIndex.Cosine(vector, expected); Assert.That(cosine, Is.GreaterThanOrEqualTo(.999)); expectedVectors.Add(expected);
                    var capture = Path.ChangeExtension(output, null) + "-" + id + "-vector.bin"; var bytes = new byte[vector.Length * 4]; Buffer.BlockCopy(vector, 0, bytes, 0, bytes.Length); File.WriteAllBytes(capture, bytes);
                    var transferred = Property<int>(sample, "LastTransferredFiles"); if (expectedWarm) Assert.That(transferred, Is.EqualTo(1));
                    ((JArray)report["registrations"]).Add(new JObject { ["id"] = id, ["module"] = module, ["cosine_to_pinned_document"] = cosine,
                        ["transferred_files_for_active_module"] = transferred, ["reused_objects"] = Property<int>(sample, "LastReusedObjects"),
                        ["capture"] = Path.GetFileName(capture), ["capture_sha256"] = Hash(capture), ["input_sha256"] = Hash(path) }); Save();
                }
                var queryPath = Cache("player-bundle/windows-unity21-20261012/search-reference.json");
                Assert.That(Hash(queryPath), Is.EqualTo("f316d5199c920431c7f721d43b2cedbe4a6384fb0e0664511b75bc2e98fe80fe"));
                var queries = JObject.Parse(File.ReadAllText(queryPath));
                foreach (var query in queries["queries"])
                {
                    cancellation.ThrowIfCancellationRequested(); Set(sample, "Query", (string)query["text"]); report["phase"] = "query_" + (string)query["id"]; Save();
                    await Call(sample, "SearchAsync"); Assert.That(previous.IsClosed, Is.True); var hits = Property<SearchHit[]>(sample, "Hits");
                    var queryVector = query["embedding"].Values<float>().ToArray();
                    var expectedHits = expectedVectors.Select((vector, index) => new { Id = "media-" + index, Score = TextSearchIndex.Cosine(queryVector, vector) })
                        .OrderByDescending(hit => hit.Score).ThenBy(hit => hit.Id, StringComparer.Ordinal).ToArray();
                    Assert.That(hits.Select(hit => hit.Document.Id), Is.EqualTo(expectedHits.Select(hit => hit.Id)));
                    var rows = new JArray();
                    for (var index = 0; index < hits.Length; index++)
                    {
                        // Qualified document vectors use cosine >= .999; report scores against the immutable Python vectors separately.
                        var difference = Math.Abs(hits[index].Score - expectedHits[index].Score);
                        Assert.That(difference, Is.LessThanOrEqualTo(.001));
                        rows.Add(new JObject { ["id"] = hits[index].Document.Id, ["score"] = hits[index].Score, ["pinned_score"] = expectedHits[index].Score, ["absolute_error"] = difference });
                    }
                    ((JArray)report["queries"]).Add(new JObject { ["id"] = (string)query["id"], ["hits"] = rows }); Save();
                    report["text_module_transferred_files"] = Property<int>(sample, "LastTransferredFiles");
                    report["text_module_reused_objects"] = Property<int>(sample, "LastReusedObjects");
                    var screenshotRoot = Environment.GetEnvironmentVariable("EG_MEDIA_SAMPLE_SCREENSHOT");
                    if (!string.IsNullOrWhiteSpace(screenshotRoot))
                    {
                        Assert.That(Application.isPlaying, Is.True, "A rendered sample screenshot requires actual PlayMode.");
                        sample.GetType().GetField("scroll", Private).SetValue(sample, new Vector2(0, 10000));
                        var screenshot = screenshotRoot + "-" + (string)query["id"] + ".png"; Assert.That(File.Exists(screenshot), Is.False);
                        for (var frame = 0; frame < 3; frame++) await Task.Yield();
                        ScreenCapture.CaptureScreenshot(screenshot);
                        for (var frame = 0; frame < 120 && !File.Exists(screenshot); frame++) await Task.Yield();
                        Assert.That(File.Exists(screenshot), Is.True); Assert.That(new FileInfo(screenshot).Length, Is.GreaterThan(1000));
                        var images = report["screenshots"] as JArray; if (images == null) report["screenshots"] = images = new JArray();
                        images.Add(new JObject { ["query"] = (string)query["id"], ["file"] = Path.GetFileName(screenshot), ["sha256"] = Hash(screenshot) }); Save();
                    }
                }
                if (expectedWarm) Assert.That(Property<int>(sample, "LastTransferredFiles"), Is.EqualTo(1));
                var finalSession = (GpuMediaSession)sample.GetType().GetField("session", Private).GetValue(sample);
                report["phase"] = "close"; Save(); await Call(sample, "CloseAsync"); Assert.That(finalSession.IsClosed, Is.True);
                using var unlocked = await GpuMediaBundleCache.AcquireAsync(bundlePath, "text", cachePath, cancellation);
                Assert.That(unlocked.TransferredFiles, Is.EqualTo(1)); report["cache_lease_released"] = true;
                Assert.That(Property<int>(sample, "DocumentCount"), Is.EqualTo(selections.Length)); report["session_closed"] = true; report["success"] = true;
            }
            catch (Exception error) { report["error"] = error.ToString(); throw; }
            finally { if (sample != null) await Call(sample, "CloseAsync"); UnityEngine.Object.DestroyImmediate(host); Save(); }
        }
    }
}
