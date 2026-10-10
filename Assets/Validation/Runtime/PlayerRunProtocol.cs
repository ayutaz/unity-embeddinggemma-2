using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.Profiling;

namespace EmbeddingGemma.Validation
{
    [Serializable] public sealed class PlayerBuildInfo
    {
        public string codeCommit, unityVersion, sentisVersion, target, scriptingBackend, stripping;
        public JObject sourceSha256;
        public void Validate()
        {
            if (codeCommit == null || !Regex.IsMatch(codeCommit, "\\A[a-f0-9]{40}\\z") || unityVersion != "6000.3.16f1" || sentisVersion != "2.6.1" ||
                !new[] { "StandaloneWindows64", "StandaloneOSX", "Android", "iOS" }.Contains(target) ||
                !new[] { "Mono2x", "IL2CPP" }.Contains(scriptingBackend) ||
                !new[] { "Disabled", "Minimal", "Low", "Medium", "High" }.Contains(stripping) || sourceSha256 == null || !sourceSha256.HasValues ||
                sourceSha256.Properties().Any(property => property.Value.Type != JTokenType.String || !Regex.IsMatch((string)property.Value, "\\A[a-f0-9]{64}\\z")))
                throw new ArgumentException("Pinned build provenance, backend and source hashes are required.");
        }
    }
    public sealed class PlayerRunConfiguration
    {
        public string Bundle, Output, RunId;
        public static PlayerRunConfiguration Parse(string[] args, string defaultBundle, string defaultOutput)
        {
            var values = new Dictionary<string, string>();
            args ??= Array.Empty<string>();
            for (var i = 0; i < args.Length; i++)
            {
                var option = args[i];
                if (!option.StartsWith("--embeddinggemma-", StringComparison.Ordinal)) continue;
                if (!new[] { "--embeddinggemma-bundle", "--embeddinggemma-output", "--embeddinggemma-run-id" }.Contains(option) ||
                    values.ContainsKey(option) || i + 1 >= args.Length || string.IsNullOrWhiteSpace(args[i + 1]) || args[i + 1].StartsWith("--", StringComparison.Ordinal))
                    throw new ArgumentException("Invalid or duplicate Player argument: " + option);
                values[option] = args[++i];
            }
            var bundle = values.TryGetValue("--embeddinggemma-bundle", out var path) ? path : defaultBundle;
            var output = values.TryGetValue("--embeddinggemma-output", out path) ? path : defaultOutput;
            var runId = values.TryGetValue("--embeddinggemma-run-id", out var id) ? id : Guid.NewGuid().ToString("N");
            if (string.IsNullOrWhiteSpace(bundle) || string.IsNullOrWhiteSpace(output) || !Regex.IsMatch(runId, "\\A[a-f0-9]{32}\\z"))
                throw new ArgumentException("Bundle, output and a lowercase 32-digit run ID are required.");
            return new PlayerRunConfiguration { Bundle = bundle.Contains("://") ? bundle : Path.GetFullPath(bundle), Output = Path.GetFullPath(output), RunId = runId };
        }
    }
    [Serializable] public sealed class PlayerMemory
    {
        public string stage;
        public long managedHeapBytes;
        public long? unityAllocatedBytes, processWorkingSetBytes;
    }
    [Serializable] public sealed class PlayerRunReport
    {
        public int schema_version = 1, tokenRows;
        public bool success, isEditor, gpuVerified;
        public string runId, startedUtc, completedUtc, phase, error, validationMode, unity, os, gpu, graphicsApi;
        public string timingScope = "Synchronous main-thread API including prompts, tokenization, schedule and final vector readback. One untimed query warmup then three timed query inferences.";
        public string memoryScope = "Whole application stage samples, not a continuously observed peak or model-exclusive memory. GPU memory usage is unknown; null counters mean unavailable.";
        public PlayerBuildInfo build;
        public JObject bundle;
        public double auditMilliseconds, tokenizerMilliseconds;
        public List<PlayerCondition> conditions = new();
        public List<PlayerMemory> memory = new();
    }
    public sealed class PlayerReportFile
    {
        readonly string path;
        public PlayerReportFile(string path)
        {
            this.path = Path.GetFullPath(path);
            Directory.CreateDirectory(Path.GetDirectoryName(this.path));
            // CreateNew never truncates a previous run, including one left by another process.
            using var stream = new FileStream(this.path, FileMode.CreateNew, FileAccess.Write, FileShare.Read);
            var initial = Encoding.UTF8.GetBytes("{\"success\":false,\"phase\":\"not_started\"}\n"); stream.Write(initial, 0, initial.Length);
        }
        public void Save(PlayerRunReport report)
        {
            var pending = path + ".pending";
            try
            {
                File.WriteAllText(pending, JsonConvert.SerializeObject(report, Formatting.Indented) + "\n", new UTF8Encoding(false));
                // Windows readers/virus scanners can briefly deny delete sharing. Keep the
                // previous complete JSON until replacement succeeds; never truncate it.
                for (var attempt = 0; ; attempt++)
                {
                    try { File.Replace(pending, path, null); break; }
                    catch (IOException) when (attempt < 20) { System.Threading.Thread.Sleep(50); }
                }
            }
            finally
            {
                try { if (File.Exists(pending)) File.Delete(pending); }
                catch (IOException) { /* Do not mask the original replacement failure. */ }
                catch (UnauthorizedAccessException) { /* Retain the original error. */ }
            }
        }
    }
    public static class PlayerRunProtocol
    {
        public static PlayerRunReport Run(PlayerRunConfiguration config, PlayerBuildInfo build, Action<PlayerRunReport> save,
            Func<string, PlayerBundle> load = null, Func<string, (int[] ids, int[] mask)> encode = null,
            Func<PlayerBundle, string, BackendType, IPlayerEmbedder> create = null)
        {
            var report = new PlayerRunReport { runId = config.RunId, startedUtc = DateTime.UtcNow.ToString("O"), phase = "starting", build = build,
                isEditor = Application.isEditor, unity = Application.unityVersion, os = SystemInfo.operatingSystem,
                gpu = SystemInfo.graphicsDeviceName, graphicsApi = SystemInfo.graphicsDeviceType.ToString(),
                validationMode = load != null || encode != null || create != null ? "injected_contract" : "real_model" };
            try
            {
                build.Validate(); if (build.unityVersion != Application.unityVersion) throw new ArgumentException("Running Unity version differs from build provenance.");
                save(report); report.memory.Add(CaptureMemory("before_audit")); report.phase = "bundle_audit"; save(report);
                var timer = Stopwatch.StartNew(); var bundle = (load ?? PlayerBundleLoader.Load)(config.Bundle); timer.Stop();
                report.auditMilliseconds = timer.Elapsed.TotalMilliseconds; report.bundle = bundle.Receipt;
                report.phase = "token_audit"; save(report); timer.Restart();
                if (encode == null)
                {
                    var tokenizer = new TextTokenizer(bundle.TokenizerJson);
                    encode = text => { var encoded = tokenizer.Encode(text); return (encoded.GetIds().ToArray(), encoded.GetAttentionMask().ToArray()); };
                }
                report.tokenRows = PlayerValidation.CheckAllTokens(bundle.Reference, encode); timer.Stop(); report.tokenizerMilliseconds = timer.Elapsed.TotalMilliseconds;
                report.memory.Add(CaptureMemory("after_token_audit"));
                foreach (var precision in new[] { "fp32", "float16" }) foreach (var backend in new[] { BackendType.CPU, BackendType.GPUCompute })
                {
                    report.phase = precision + ":" + backend; save(report); report.memory.Add(CaptureMemory("before_" + report.phase));
                    double modelLoad = 0, workerCreation = 0;
                    var condition = PlayerValidation.RunCondition(bundle.Reference, precision, backend, () =>
                    {
                        IPlayerEmbedder provider;
                        if (create != null) provider = create(bundle, precision, backend);
                        else
                        {
                            var clock = Stopwatch.StartNew(); var model = TextModelFile.Load(Path.Combine(bundle.Directory, "model-" + precision + ".sentis"));
                            clock.Stop(); modelLoad = clock.Elapsed.TotalMilliseconds; clock.Restart();
                            provider = new SentisProvider(new TextEmbedder(model, bundle.TokenizerJson, backend)); clock.Stop(); workerCreation = clock.Elapsed.TotalMilliseconds;
                        }
                        report.memory.Add(CaptureMemory("worker_created_" + report.phase)); return provider;
                    });
                    condition.modelLoadMilliseconds = modelLoad; condition.workerCreationMilliseconds = workerCreation; report.conditions.Add(condition);
                    report.memory.Add(CaptureMemory("released_" + report.phase)); save(report);
                }
                report.success = report.tokenRows == 25 && PlayerValidation.AllConditionsPassed(report.conditions.ToArray());
                report.gpuVerified = report.success && report.validationMode == "real_model";
                if (!report.success) report.error = "One or more required conditions failed; see condition errors.";
                report.phase = report.success ? "completed" : "failed";
            }
            catch (Exception exception) { report.success = false; report.gpuVerified = false; report.phase = "failed"; report.error = exception.GetType().Name + ": " + exception.Message; }
            finally
            {
                report.completedUtc = DateTime.UtcNow.ToString("O");
                try { save(report); }
                catch (Exception exception) { report.success = false; report.gpuVerified = false; report.error = (report.error ?? "") + " Result write: " + exception.Message; }
            }
            return report;
        }

        static PlayerMemory CaptureMemory(string stage)
        {
            var memory = new PlayerMemory { stage = stage, managedHeapBytes = GC.GetTotalMemory(false) };
            var allocated = Profiler.GetTotalAllocatedMemoryLong(); if (allocated > 0) memory.unityAllocatedBytes = allocated;
            try { using var process = Process.GetCurrentProcess(); if (process.WorkingSet64 > 0) memory.processWorkingSetBytes = process.WorkingSet64; }
            catch (Exception) { /* Platform counter unavailable, retain null. */ }
            return memory;
        }
        sealed class SentisProvider : IPlayerEmbedder
        {
            readonly TextEmbedder inner;
            public SentisProvider(TextEmbedder inner) { this.inner = inner; }
            public BackendType Backend => inner.Backend;
            public float[] EmbedRaw(string text) => inner.EmbedRaw(text);
            public float[] EmbedQuery(string text) => inner.EmbedQuery(text);
            public float[] EmbedDocument(string text, string title = null) => inner.EmbedDocument(text, title);
            public void Dispose() => inner.Dispose();
        }
    }
}
