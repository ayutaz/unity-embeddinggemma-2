using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Newtonsoft.Json.Linq;
using UnityEngine;

namespace EmbeddingGemma.Validation
{
    public interface IPlayerSampleView
    {
        bool Ready { get; }
        string Error { get; }
        string ActualBackend { get; }
        bool FontAlive { get; }
        int FontInstanceId { get; }
        JObject CacheSnapshot { get; }
        IEnumerator Prepare(string bundle, string cacheRoot);
        PlayerRanking[] SearchRanking(string query);
        bool RejectBlank();
        bool RejectMissingModel();
        bool ReleaseAndCheckNativeOwner();
        void DestroyView();
    }
    public static class PlayerSampleExecution
    {
        public static IEnumerator Run(PlayerRunConfiguration config, PlayerBuildInfo build, Action<JObject> save,
            Action<JObject> completed, Func<IPlayerSampleView> create = null, Func<string, PlayerBundle> load = null,
            Func<string, JObject> capture = null, Func<bool> isBatchMode = null)
        {
            var injected = create != null || load != null || capture != null || isBatchMode != null;
            var report = new JObject {
                ["schema_version"] = 1, ["success"] = false, ["gpu_verified"] = false,
                ["validation_mode"] = injected ? "injected_contract" : "real_model",
                ["scope"] = "Imported TextSearch sample, Float16 weights, GPUCompute, fixed six documents and four queries. Programmatic sample API; no native keyboard or mouse validation.",
                ["run_id"] = config.RunId, ["started_utc"] = DateTime.UtcNow.ToString("O"), ["phase"] = "starting",
                ["is_editor"] = Application.isEditor, ["unity"] = Application.unityVersion,
                ["os"] = SystemInfo.operatingSystem, ["gpu"] = SystemInfo.graphicsDeviceName,
                ["graphics_api"] = SystemInfo.graphicsDeviceType.ToString(), ["build"] = JObject.FromObject(build),
                ["queries"] = new JArray(), ["screenshots"] = new JArray(), ["console"] = new JArray(),
                ["worker_released"] = false, ["font_destroyed"] = false
            };
            IPlayerSampleView view = null;
            var released = false; var destroyed = false;
            var stack = new Stack<IEnumerator>();
            void Log(string message, string trace, LogType kind)
            {
                if (kind != LogType.Log) ((JArray)report["console"]).Add(new JObject { ["type"] = kind.ToString(), ["message"] = message, ["stack"] = trace });
            }
            void Failure(Exception exception)
            {
                report["success"] = false; report["gpu_verified"] = false; report["phase"] = "failed";
                report["error"] = ((string)report["error"] ?? "") + exception.GetType().Name + ": " + exception.Message + "\n";
            }
            Application.logMessageReceived += Log;
            IEnumerator Core()
            {
                build.Validate();
                if (build.unityVersion != Application.unityVersion) throw new ArgumentException("Running Editor version differs from build provenance.");
                report["phase"] = "runtime_preflight";
                report["batch_mode"] = (isBatchMode ?? (() => Application.isBatchMode))();
                if ((bool)report["batch_mode"])
                    throw new NotSupportedException("Sample GUI validation requires interactive rendering; batch mode is not supported.");
                view = (create ?? (() => PlayerSampleView.FromScene()))();
                report["phase"] = "gui_preflight"; save(report);
                var clock = Stopwatch.StartNew();
                yield return new WaitForEndOfFrame();
                var guiFrames = 0;
                while (!view.FontAlive && guiFrames < 120) { guiFrames++; yield return new WaitForEndOfFrame(); }
                report["gui_wait_frames"] = guiFrames;
                report["render_observation"] = new JObject { ["width"] = Screen.width, ["height"] = Screen.height,
                    ["focused"] = Application.isFocused, ["batch_mode"] = Application.isBatchMode,
                    ["splash_finished"] = UnityEngine.Rendering.SplashScreen.isFinished };
                if (view is PlayerSampleView realView) report["sample_observation"] = realView.Describe();
                if (!view.FontAlive || view.FontInstanceId == 0)
                {
                    var path = Path.Combine(Path.GetDirectoryName(config.Output), config.RunId + "-gui-not-ready.png");
                    report["diagnostic_screenshot"] = (capture ?? Capture)(path);
                    throw new InvalidOperationException("The sample GUI native font is missing after waiting 120 render frames.");
                }
                report["font_instance_id"] = view.FontInstanceId;
                clock.Stop(); report["gui_preflight_milliseconds"] = clock.Elapsed.TotalMilliseconds;
                report["phase"] = "bundle_audit"; save(report); clock.Restart();
                var bundle = (load ?? PlayerBundleLoader.Load)(config.Bundle); clock.Stop();
                report["bundle"] = bundle.Receipt; report["hash_backends"] = bundle.HashBackends; report["audit_milliseconds"] = clock.Elapsed.TotalMilliseconds;
                if (bundle.Reference?.Queries?.Length != 4) throw new ArgumentException("Four pinned sample queries are required.");
                report["phase"] = "preparation"; save(report); clock.Restart();
                yield return view.Prepare(config.Bundle, Path.Combine(Path.GetDirectoryName(config.Output), "sample-model-cache"));
                clock.Stop(); report["preparation_milliseconds"] = clock.Elapsed.TotalMilliseconds;
                report["preparation_timing_scope"] = "File URL staging and full cache audit, model deserialization, Worker creation and six document embeddings together; one measurement.";
                if (!view.Ready) throw new InvalidOperationException("Sample is not ready: " + view.Error);
                report["actual_backend"] = view.ActualBackend;
                if (view.ActualBackend != "GPUCompute") throw new InvalidOperationException("An actual GPUCompute Worker is required.");
                report["cache"] = view.CacheSnapshot;
                if (!view.FontAlive) throw new InvalidOperationException("The sample lost its GUI font during model preparation.");
                for (var i = 0; i < bundle.Reference.Queries.Length; i++)
                {
                    var query = bundle.Reference.Queries[i]; report["phase"] = "query:" + query.id; save(report);
                    clock.Restart(); var ranks = view.SearchRanking(query.text); clock.Stop();
                    PlayerSampleQueryValidation.Check(query, ranks);
                    ((JArray)report["queries"]).Add(new JObject { ["id"] = query.id, ["text"] = query.text, ["ranking"] = JArray.FromObject(ranks), ["milliseconds"] = clock.Elapsed.TotalMilliseconds,
                        ["maximum_score_error"] = ranks.Select((rank, index) => Math.Abs(rank.score - query.ranking[index].score)).Max() });
                    if (i < 2)
                    {
                        yield return new WaitForEndOfFrame();
                        var path = Path.Combine(Path.GetDirectoryName(config.Output), config.RunId + "-" + query.id + ".png");
                        var screenshot = (capture ?? Capture)(path);
                        if (screenshot == null) throw new InvalidOperationException("Screenshot capture returned no result.");
                        ((JArray)report["screenshots"]).Add(screenshot);
                    }
                }
                if (!view.RejectBlank()) throw new InvalidOperationException("The sample accepted a blank query.");
                report["blank_rejected"] = true;
                released = view.ReleaseAndCheckNativeOwner(); report["worker_released"] = released;
                if (!released) throw new InvalidOperationException("Worker or model cache lease was not released.");
                if (!view.RejectMissingModel()) throw new InvalidOperationException("A missing model was not rejected without retaining a Worker.");
                report["missing_model_rejected"] = true;
                if (!view.FontAlive) throw new InvalidOperationException("The native GUI font was destroyed before the view.");
                view.DestroyView(); destroyed = true; yield return null;
                report["font_destroyed"] = !view.FontAlive;
                if (view.FontAlive) throw new InvalidOperationException("The sample view retained its native font after destruction.");
                if (((JArray)report["console"]).Any(log => (string)log["type"] == "Error" || (string)log["type"] == "Exception" || (string)log["type"] == "Assert"))
                    throw new InvalidOperationException("Unexpected Console errors were recorded.");
                report["success"] = true; report["gpu_verified"] = !injected && !Application.isEditor; report["phase"] = "completed";
            }
            stack.Push(Core());
            try
            {
                while (stack.Count > 0)
                {
                    bool moved; object current = null;
                    try
                    {
                        moved = stack.Peek().MoveNext();
                        if (moved) current = stack.Peek().Current;
                        else (stack.Pop() as IDisposable)?.Dispose();
                    }
                    catch (Exception exception) { Failure(exception); break; }
                    if (moved) { if (current is IEnumerator child) stack.Push(child); else yield return current; }
                }
            }
            finally
            {
                while (stack.Count > 0)
                    try { (stack.Pop() as IDisposable)?.Dispose(); } catch (Exception exception) { Failure(exception); }
                try
                {
                    if (view != null && !destroyed)
                    {
                        try { if (!released) report["worker_released"] = view.ReleaseAndCheckNativeOwner(); }
                        finally { view.DestroyView(); }
                    }
                }
                catch (Exception exception) { Failure(exception); }
                Application.logMessageReceived -= Log;
                report["completed_utc"] = DateTime.UtcNow.ToString("O");
                try { save(report); } catch (Exception exception) { Failure(exception); }
                completed(report);
            }
        }
        static JObject Capture(string path)
        {
            var texture = ScreenCapture.CaptureScreenshotAsTexture();
            if (texture == null) throw new InvalidOperationException("Player screenshot texture is unavailable.");
            try
            {
                var bytes = texture.EncodeToPNG();
                using (var stream = new FileStream(path, FileMode.CreateNew, FileAccess.Write, FileShare.Read)) stream.Write(bytes, 0, bytes.Length);
                using var read = File.OpenRead(path); var sha = PlayerFileHash.Compute(read, out var backend);
                return new JObject { ["name"] = Path.GetFileName(path), ["width"] = texture.width, ["height"] = texture.height, ["bytes"] = bytes.Length, ["sha256"] = sha, ["hash_backend"] = backend };
            }
            finally { UnityEngine.Object.Destroy(texture); }
        }
    }
}
