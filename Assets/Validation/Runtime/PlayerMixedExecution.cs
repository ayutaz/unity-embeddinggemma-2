using System;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;
using EmbeddingGemma;
using Newtonsoft.Json.Linq;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.Profiling;

namespace EmbeddingGemma.Validation
{
    /// <summary>Explicit all-role media and mixed text/image/audio Windows Player execution.</summary>
    public static class PlayerMixedExecution
    {
        const BindingFlags Private = BindingFlags.Instance | BindingFlags.NonPublic;
        static readonly string[] Modules = { "image-square", "image-landscape", "image-portrait", "audio-1119", "audio-839" };
        public static bool IsRequested(string[] args) => args != null && args.Any(a => a != null && a.StartsWith("--embeddinggemma-mixed-", StringComparison.Ordinal));
        static JObject AuditText(TextEmbedder text)
        {
            var worker = typeof(TextEmbedder).GetField("worker", Private)?.GetValue(text) as Worker;
            return PlayerGpuExecutionAudit.Inspect(worker);
        }
        static string Sha(byte[] bytes)
        { using var hash = SHA256.Create(); return BitConverter.ToString(hash.ComputeHash(bytes)).Replace("-", "").ToLowerInvariant(); }
        static bool TextReleased(TextEmbedder text) => typeof(TextEmbedder).GetField("worker", Private)?.GetValue(text) == null
            && typeof(TextEmbedder).GetField("workerOwner", Private)?.GetValue(text) == null;

        public static async Task<bool> Run(PlayerBuildInfo build, string[] args)
        {
            // Invalid arguments and existing outputs fail before any input/model/cache mutation.
            var config = PlayerMixedConfiguration.Parse(args); Directory.CreateDirectory(Path.GetDirectoryName(config.Output));
            using var output = new FileStream(config.Output, FileMode.CreateNew, FileAccess.Write, FileShare.Read);
            var envelope = new JObject { ["schema_version"] = 1, ["success"] = false, ["gpu_verified"] = false,
                ["scope"] = "windows_release_il2cpp_high_all_media_roles_and_complete_mixed_index",
                ["full_multimodal_acceptance"] = false, ["independent_cold_warm_accepted"] = false, ["all_platforms_accepted"] = false,
                ["final_performance_accepted"] = false, ["run_id"] = config.RunId, ["process_phase"] = config.Phase,
                ["precision"] = config.Precision, ["requested_bundle_descriptor_sha256"] = config.DescriptorSha256,
                ["started_utc"] = DateTime.UtcNow.ToString("O"), ["is_editor"] = Application.isEditor, ["batch_mode"] = Application.isBatchMode,
                ["unity"] = Application.unityVersion, ["graphics_api"] = SystemInfo.graphicsDeviceType.ToString(), ["gpu"] = SystemInfo.graphicsDeviceName,
                ["system_memory_mb"] = SystemInfo.systemMemorySize, ["gpu_capacity_mb"] = SystemInfo.graphicsMemorySize,
                ["build"] = build == null ? null : JObject.FromObject(build), ["modules"] = new JObject(), ["cache_modules"] = new JArray(),
                ["text_worker_released"] = false, ["memory_scope"] = "Whole application stage samples; not a continuous peak, VRAM or SDK-exclusive allocation",
                ["timing_scope"] = "Wall time including model preparation, scheduling, fences, final vector download and receipt work; per-inference captures exclude receipt serialization" };
            byte[] Save()
            {
                envelope["managed_heap_bytes"] = GC.GetTotalMemory(false); envelope["unity_allocated_bytes"] = Profiler.GetTotalAllocatedMemoryLong();
                var bytes = new UTF8Encoding(false).GetBytes(envelope.ToString());
                output.Position = 0; output.Write(bytes, 0, bytes.Length); output.SetLength(bytes.Length); output.Flush(true); return bytes;
            }
            void Phase(string phase) { envelope["phase"] = phase; Save(); }
            void CacheEvidence(string module, GpuMediaBundleLease lease)
            {
                ((JArray)envelope["cache_modules"]).Add(PlayerMixedCachePolicy.AcquisitionEvidence(module, lease, config.Phase == "warm"));
            }
            TextEmbedder text = null; GpuMediaBundleLease textLease = null; var completed = false;
            try
            {
                Phase("runtime_preflight");
                if (Application.isEditor || Application.platform != RuntimePlatform.WindowsPlayer || !SystemInfo.supportsComputeShaders
                    || build == null || build.target != "StandaloneWindows64" || build.scriptingBackend != "IL2CPP" || build.stripping != "High"
                    || Debug.isDebugBuild || build.unityVersion != Application.unityVersion)
                    throw new NotSupportedException("An actual Windows Release IL2CPP High Player and compute GPU are required.");
                build.Validate(); Application.runInBackground = true;
                envelope["cache_context"] = PlayerMixedCachePolicy.Validate(config, build);
                var preparedCache = PlayerMixedCachePolicy.PreparedCacheDirectory(config.Cache);
                using (var descriptor = File.OpenRead(Path.Combine(config.Bundle, "bundle.json")))
                {
                    var digest = PlayerFileHash.Compute(descriptor, out _); envelope["bundle_descriptor_sha256"] = digest;
                    if (digest != config.DescriptorSha256) throw new InvalidDataException("The independent model descriptor pin differs.");
                }
                var sourceBundle = GpuMediaBundle.Open(config.Bundle);
                foreach (var name in Modules.Concat(new[] { "text" })) sourceBundle.Module(name);
                Phase("verify_original_text_reference"); var original = PlayerMixedSearchFixture.LoadText(config.Fixtures);
                envelope["text_reference_sha256"] = original["reference_sha256"].DeepClone();
                var documents = (JArray)original["documents"]; var mediaVectors = new JObject();
                foreach (var name in Modules)
                {
                    Phase("prepare_" + name);
                    using var lease = await GpuMediaBundleCache.AcquireAsync(config.Bundle, name, preparedCache);
                    CacheEvidence(name, lease);
                    using var fixture = PlayerMediaFixture.Load(config.Fixtures, name, lease.Bundle.Module(name));
                    ((JObject)envelope["modules"])[name] = new JObject { ["success"] = false };
                    var result = await PlayerMediaProtocol.Run(fixture.Cases, config.Precision,
                        () => PlayerPreparedMediaModule.Load(lease.Bundle, name), report => { ((JObject)envelope["modules"])[name] = report; Save(); });
                    ((JObject)envelope["modules"])[name] = result;
                    ((JObject)envelope["modules"])[name]["original_fixtures"] = fixture.Evidence;
                    if (!(bool)result["success"] || !(bool)result["gpu_verified"] || (string)result["validation_mode"] != "real_prepared_module")
                        throw new InvalidOperationException("Every fixed media module must complete with real GPU execution and release.");
                    PlayerMixedSearchFixture.AppendMedia(documents, mediaVectors, name, fixture.Cases, result, config.Precision);
                    await Task.Yield();
                }
                var modules = ((JObject)envelope["modules"]).Properties().Select(p => (JObject)p.Value).ToArray();
                var baseCount = modules.Sum(r => ((JArray)r["cases"]).Count); var warmCount = modules.Sum(r => ((JArray)r["warm"]).Count);
                if (baseCount != 36 || warmCount != 45) throw new InvalidOperationException("All 36 base and 45 warm media-role vectors are required.");
                envelope["base_media_vectors"] = baseCount; envelope["warm_media_vectors"] = warmCount;
                Phase("prepare_text"); textLease = await GpuMediaBundleCache.AcquireAsync(config.Bundle, "text", preparedCache); CacheEvidence("text", textLease);
                var metadata = textLease.Bundle.Module("text");
                var model = ModelLoader.Load(textLease.Bundle.VerifyFile((string)metadata["model"]));
                text = new TextEmbedder(model, File.ReadAllText(textLease.Bundle.VerifyFile((string)metadata["tokenizer"])), BackendType.GPUCompute);
                envelope["text_gpu_execution_before"] = AuditText(text); Phase("mixed_index");
                envelope["mixed_index"] = PlayerMixedSearchProtocol.Run(documents, (JArray)original["queries"], mediaVectors, config.Precision, text);
                envelope["text_gpu_execution_after"] = AuditText(text); completed = (bool)envelope["mixed_index"]["success"];
            }
            catch (Exception error)
            {
                envelope["failed_phase"] = envelope["phase"]?.DeepClone(); envelope["error_type"] = error.GetType().Name;
                if (!Application.isEditor) Debug.LogException(error); // Raw diagnostics remain private; outgoing envelopes contain types only.
            }
            finally
            {
                try
                {
                    text?.Dispose(); envelope["text_worker_released"] = text != null && TextReleased(text);
                }
                catch (Exception error) { completed = false; envelope["release_error_type"] = error.GetType().Name; }
                finally
                {
                    try { textLease?.Dispose(); }
                    catch (Exception error) { completed = false; envelope["lease_release_error_type"] = error.GetType().Name; }
                }
                envelope["success"] = completed && (bool)envelope["text_worker_released"];
                envelope["gpu_verified"] = (bool)envelope["success"] && !Application.isEditor;
                envelope["phase"] = (bool)envelope["success"] ? "complete" : "failed";
                envelope["completed_utc"] = DateTime.UtcNow.ToString("O"); var bytes = Save();
                if ((bool)envelope["success"] && config.Phase == "cold")
                {
                    try
                    {
                        var qualifier = new JObject { ["schema_version"] = 1, ["success"] = true, ["gpu_verified"] = true, ["phase"] = "cold",
                            ["source_commit"] = build.codeCommit, ["precision"] = config.Precision, ["bundle_descriptor_sha256"] = config.DescriptorSha256,
                            ["run_id"] = config.RunId, ["cold_report_sha256"] = Sha(bytes), ["base_media_vectors"] = 36, ["warm_media_vectors"] = 45,
                            ["mixed_documents"] = 18, ["mixed_queries"] = 4, ["mixed_rank_and_score_pairs"] = 72 };
                        using var stream = new FileStream(Path.Combine(config.Cache, PlayerMixedCachePolicy.QualificationName), FileMode.CreateNew, FileAccess.Write, FileShare.Read);
                        var raw = new UTF8Encoding(false).GetBytes(qualifier.ToString()); stream.Write(raw, 0, raw.Length); stream.Flush(true);
                    }
                    catch (Exception error)
                    {
                        envelope["success"] = false; envelope["gpu_verified"] = false; envelope["phase"] = "failed";
                        envelope["failed_phase"] = "write_cold_qualification"; envelope["error_type"] = error.GetType().Name; Save();
                    }
                }
            }
            return (bool)envelope["success"] && (bool)envelope["gpu_verified"];
        }
    }
}
