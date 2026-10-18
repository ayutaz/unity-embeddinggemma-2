using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json.Linq;

namespace EmbeddingGemma.Validation
{
    public sealed class PlayerMediaCase
    {
        public string Id;
        public TextRole Role = TextRole.Document;
        public float[] Expected;
        public object Input;
    }

    public interface IPlayerMediaModule
    {
        Task<float[]> EmbedAsync(PlayerMediaCase input, CancellationToken token);
        JObject InspectGpuExecution();
        Task CloseAsync();
        bool IsReleased { get; }
    }

    /// <summary>One fixed media module per process; complete multimodal acceptance is a separate gate.</summary>
    public static class PlayerMediaProtocol
    {
        public static Task<JObject> Run(IReadOnlyList<PlayerMediaCase> cases, string precision,
            Func<Task<IPlayerMediaModule>> load, Action<JObject> save)
        {
            if (cases == null || cases.Count == 0 || cases.Any(row => row == null || string.IsNullOrEmpty(row.Id) || !Enum.IsDefined(typeof(TextRole), row.Role)) ||
                cases.Select(row => (row.Id, row.Role)).Distinct().Count() != cases.Count ||
                precision != "fp32" && precision != "float16" || load == null || save == null)
                throw new ArgumentException("Unique fixed cases, a known precision and execution callbacks are required.");
            foreach (var row in cases) PlayerValidation.CheckEmbedding(row.Expected, row.Expected, precision);
            return Execute(cases, precision, load, save);
        }

        static async Task<JObject> Execute(IReadOnlyList<PlayerMediaCase> cases, string precision,
            Func<Task<IPlayerMediaModule>> load, Action<JObject> save)
        {
            var report = new JObject { ["schema_version"] = 1, ["success"] = false, ["gpu_verified"] = false,
                ["full_multimodal_acceptance"] = false, ["scope"] = "fixed_media_module_player",
                ["roles"] = new JArray(cases.Select(row => row.Role.ToString().ToLowerInvariant()).Distinct()),
                ["pre_cancel_verified"] = false, ["pending_call_cancel_verified"] = false, ["session_released"] = false,
                ["precision"] = precision, ["phase"] = "loading", ["cases"] = new JArray(), ["warm"] = new JArray(),
                ["timing_scope"] = "Main-thread wall time including preprocessing, scheduling, GPU fences and final vector readback; not GPU-only time",
                ["first_inference_scope"] = "First fixed case before cancellation probes; external shader/driver cache state is not certified",
                ["cancellation_scope"] = "Pending API call cancelled at its first asynchronous yield; cancellation during a submitted GPU kernel is not certified",
                ["memory_scope"] = "Managed heap samples only; process peak and VRAM require independent measurement",
                ["started_utc"] = DateTime.UtcNow.ToString("O") };
            IPlayerMediaModule module = null; var completed = false;
            void Phase(string name) { report["phase"] = name; report["managed_heap_bytes"] = GC.GetTotalMemory(false); save(report); }
            async Task Capture(PlayerMediaCase row, JArray group)
            {
                var timer = Stopwatch.StartNew(); var actual = await module.EmbedAsync(row, CancellationToken.None); timer.Stop();
                var capture = new JObject { ["id"] = row.Id, ["role"] = row.Role.ToString().ToLowerInvariant(), ["milliseconds"] = timer.Elapsed.TotalMilliseconds,
                    ["embedding"] = actual == null ? null : new JArray(actual) };
                group.Add(capture); save(report);
                capture["cosine"] = PlayerValidation.CheckEmbedding(actual, row.Expected, precision); save(report);
            }
            try
            {
                Phase("loading"); var timer = Stopwatch.StartNew(); module = await load(); timer.Stop();
                if (module == null) throw new InvalidOperationException("Module loader returned no session.");
                report["load_milliseconds"] = timer.Elapsed.TotalMilliseconds;
                report["validation_mode"] = module is PlayerPreparedMediaModule ? "real_prepared_module" : "injected_contract";
                Phase("gpu_audit_before"); report["gpu_execution_before"] = module.InspectGpuExecution();
                foreach (var row in cases) { Phase("case_" + row.Id); await Capture(row, (JArray)report["cases"]); }
                report["first_inference_milliseconds"] = report["cases"][0]["milliseconds"]?.DeepClone();
                Phase("pre_cancel");
                using (var cancellation = new CancellationTokenSource())
                {
                    cancellation.Cancel();
                    try { await module.EmbedAsync(cases[0], cancellation.Token); }
                    catch (OperationCanceledException) { report["pre_cancel_verified"] = true; }
                    if (!(bool)report["pre_cancel_verified"]) throw new InvalidOperationException("Pre-cancelled inference did not cancel.");
                }
                Phase("pending_call_cancel");
                using (var cancellation = new CancellationTokenSource())
                {
                    var pending = module.EmbedAsync(cases[0], cancellation.Token);
                    var wasPending = !pending.IsCompleted; cancellation.Cancel();
                    try { await pending; }
                    catch (OperationCanceledException) { report["pending_call_cancel_verified"] = wasPending; }
                    if (!(bool)report["pending_call_cancel_verified"]) throw new InvalidOperationException("A genuinely pending inference must cancel.");
                }
                for (var repetition = 0; repetition < 3; repetition++)
                    foreach (var row in cases.GroupBy(row => row.Role).Select(group => group.First()))
                    { Phase("warm_" + repetition + "_" + row.Role.ToString().ToLowerInvariant()); await Capture(row, (JArray)report["warm"]); }
                Phase("gpu_audit_after"); report["gpu_execution_after"] = module.InspectGpuExecution(); completed = true;
            }
            catch (Exception error)
            {
                // Raw exceptions/logs stay private: paths from a driver or file API must not enter a shareable receipt.
                report["failed_phase"] = report["phase"]?.DeepClone(); report["error_type"] = error.GetType().Name;
            }
            finally
            {
                if (module != null)
                {
                    var timer = Stopwatch.StartNew();
                    try { Phase("closing"); }
                    finally
                    {
                        try { await module.CloseAsync(); report["session_released"] = module.IsReleased; }
                        catch (Exception error) { report["release_error_type"] = error.GetType().Name; }
                        finally { timer.Stop(); report["release_milliseconds"] = timer.Elapsed.TotalMilliseconds; }
                    }
                }
                report["success"] = completed && (bool)report["session_released"];
                report["gpu_verified"] = (bool)report["success"] && module is PlayerPreparedMediaModule && !UnityEngine.Application.isEditor;
                report["phase"] = (bool)report["success"] ? "complete" : "failed";
                report["completed_utc"] = DateTime.UtcNow.ToString("O"); save(report);
            }
            return report;
        }
    }
}
