using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
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
    /// <summary>Original files, isolated GPU prefix. Never loads backbone or claims full-model qualification.</summary>
    public sealed class IsolatedPreparedImagePrefixTests
    {
        static string Hash(string path)
        {
            using var stream = File.OpenRead(path); using var hash = SHA256.Create();
            return BitConverter.ToString(hash.ComputeHash(stream)).Replace("-", "").ToLowerInvariant();
        }
        [UnityTearDown] public IEnumerator Exit() { if (Application.isPlaying) yield return new ExitPlayMode(); }
        [UnityTest] public IEnumerator OriginalPinnedPrefixRunsWithoutLoadingBackboneOrUnusedSegments()
        {
            yield return new EnterPlayMode();
            using var cancellation = new CancellationTokenSource(); var task = Run(cancellation.Token);
            const long limit = 8L * 1024 * 1024 * 1024; long peak = 0; var guardTripped = false;
            while (!task.IsCompleted)
            {
                peak = Math.Max(peak, MediaExecutionMemory.Peak());
                if (peak > limit) { guardTripped = true; cancellation.Cancel(); }
                UnityEditor.EditorApplication.QueuePlayerLoopUpdate(); yield return null;
            }
            peak = Math.Max(peak, MediaExecutionMemory.Peak());
            var output = Environment.GetEnvironmentVariable("EG_ISOLATED_PREFIX_REPORT");
            if (!string.IsNullOrEmpty(output) && File.Exists(output))
            {
                var report = JObject.Parse(File.ReadAllText(output));
                report["host_native_lifetime_peak_bytes"] = peak; report["host_guard_limit_bytes"] = limit;
                report["host_guard_tripped"] = guardTripped || peak > limit;
                if (guardTripped || peak > limit) report["diagnostic_success"] = false;
                File.WriteAllText(output, report.ToString());
            }
            task.GetAwaiter().GetResult();
            Assert.That(guardTripped || peak > limit, Is.False, "Host-memory guard is not a successful diagnostic.");
        }
        static async Task Run(CancellationToken token)
        {
            var directory = Environment.GetEnvironmentVariable("EG_MEDIA_BUNDLE_OUTPUT");
            var image = Environment.GetEnvironmentVariable("EG_GPU_TRACE_IMAGE");
            var output = Environment.GetEnvironmentVariable("EG_ISOLATED_PREFIX_REPORT");
            var through = int.Parse(Environment.GetEnvironmentVariable("EG_GPU_TRACE_THROUGH") ?? "2");
            var retire = Environment.GetEnvironmentVariable("EG_GPU_PREFIX_RETIRE") == "1";
            Assert.That(through, Is.InRange(0, 7)); Assert.That(directory, Is.Not.Null.And.Not.Empty);
            Assert.That(output, Is.Not.Null.And.Not.Empty); Assert.That(File.Exists(output), Is.False);
            var events = Path.ChangeExtension(output, ".events.jsonl"); Assert.That(File.Exists(events), Is.False);
            var descriptor = Path.Combine(directory, "bundle.json");
            Assert.That(Hash(descriptor), Is.EqualTo("8363f9a63a95b307fbbb835f3551a41f6fa5a1d21aca3017670e6f678647e955"));
            Assert.That(Hash(image), Is.EqualTo("c2a1900a1656ce2d843e9d06801c62f927b98a51e9b95ae1159bb2c7515fa399"));
            var bundle = JObject.Parse(File.ReadAllText(descriptor)); var module = bundle["modules"]["image-square"];
            var modelFiles = module["models"].Take(through + 1).Values<string>().ToArray();
            var operations = module["operations"].Take(through + 1).Cast<JObject>().ToArray();
            var selected = modelFiles.Concat(operations.Where(o => o["weight_file"] != null).Select(o => (string)o["weight_file"])).Distinct().ToArray();
            var bytes = selected.Sum(name => (long)bundle["files"][name]["bytes"]);
            Assert.That(bytes, Is.LessThanOrEqualTo(64L * 1024 * 1024), "Keep this diagnostic bounded; do not silently load full modules.");
            var report = new JObject { ["diagnostic_success"] = false, ["full_model_qualified"] = false,
                ["scope"] = retire ? "isolated original prefix with pinned cache retirement and prefix liveness; no full session/backbone/final vector"
                    : "isolated original prefix retaining all intermediates; no full session/backbone/final vector",
                ["backbone_loaded"] = false, ["through_operation"] = through, ["selected_file_bytes"] = bytes,
                ["original_cache_retirement"] = retire,
                ["declared_runtime_source"] = Environment.GetEnvironmentVariable("EG_ACCEPTANCE_SOURCE"),
                ["runtime_assembly_sha256"] = Hash(typeof(GpuProjectionPipeline).Assembly.Location),
                ["validation_assembly_sha256"] = Hash(typeof(IsolatedGpuPrefix).Assembly.Location),
                ["bundle_descriptor_sha256"] = Hash(descriptor), ["image_sha256"] = Hash(image),
                ["unity"] = Application.unityVersion, ["graphics_api"] = SystemInfo.graphicsDeviceType.ToString(),
                ["gpu"] = SystemInfo.graphicsDeviceName, ["files"] = new JArray(), ["phase"] = "verify_selected_files" };
            void Save() => File.WriteAllText(output, report.ToString()); Save();
            var paths = new Dictionary<string, string>(); var ownedOperations = new List<IGpuTensorOperation>(); Texture2D texture = null;
            try
            {
                foreach (var name in selected)
                {
                    var path = Path.GetFullPath(Path.Combine(directory, name));
                    Assert.That(path.StartsWith(Path.GetFullPath(directory) + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase), Is.True);
                    Assert.That(new FileInfo(path).Length, Is.EqualTo((long)bundle["files"][name]["bytes"]));
                    Assert.That(Hash(path), Is.EqualTo((string)bundle["files"][name]["sha256"])); paths.Add(name, path);
                    ((JArray)report["files"]).Add(new JObject { ["file"] = name, ["sha256"] = Hash(path) });
                }
                var models = modelFiles.Select(name => ModelLoader.Load(paths[name])).ToArray();
                var factory = typeof(GpuMediaBundle).GetMethod("CreateOperation", BindingFlags.Static | BindingFlags.NonPublic);
                Assert.That(factory, Is.Not.Null);
                foreach (var operation in operations)
                    ownedOperations.Add((IGpuTensorOperation)factory.Invoke(null, new object[] { operation, paths }));
                texture = new Texture2D(2, 2, TextureFormat.RGBA32, false, true);
                Assert.That(ImageConversion.LoadImage(texture, File.ReadAllBytes(image), true), Is.True);
                using var preprocessing = new ImagePreprocessor(BackendType.GPUCompute);
                using var processed = preprocessing.Process(texture);
                report["phase"] = "isolated_original_prefix"; Save();
                using var result = await IsolatedGpuPrefix.ExecuteAsync(models,
                    operations.Select(o => o["inputs"].Values<string>().ToArray()).ToArray(),
                    operations.Select(o => (string)o["output"]).ToArray(), ownedOperations.ToArray(), processed.Pixels, row =>
                    { File.AppendAllText(events, row.ToString(Newtonsoft.Json.Formatting.None) + "\n"); report["last_event"] = row; Save(); }, token, retire);
                report["result_shape"] = result.shape.ToString(); report["result_backend"] = result.backendType.ToString();
                report["host_native_lifetime_peak_bytes"] = MediaExecutionMemory.Peak();
                report["intermediate_readback"] = false; report["diagnostic_success"] = true; report["phase"] = "complete"; Save();
            }
            catch (Exception error) { report["error_type"] = error.GetType().FullName; report["error"] = error.ToString(); throw; }
            finally
            {
                foreach (var operation in ownedOperations) operation.Dispose();
                if (texture != null) UnityEngine.Object.DestroyImmediate(texture); Save();
            }
        }
    }
}
