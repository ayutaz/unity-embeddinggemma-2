using System;
using System.Collections;
using System.IO;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using UnityEngine;

namespace EmbeddingGemma.Validation
{
    /// <summary>Opt-in device fixture materialization and the unchanged original mixed numerical/GPU protocol.</summary>
    public sealed class PlayerDeviceMixedRunner : MonoBehaviour
    {
        PlayerBuildInfo build;
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        static void Bootstrap()
        {
            if (Application.isEditor) return;
            var marker = Resources.Load<TextAsset>("EmbeddingGemmaDeviceMixedValidationBuild");
            if (marker == null) return;
            var owner = new GameObject("EmbeddingGemma Device Mixed Validation"); DontDestroyOnLoad(owner);
            owner.AddComponent<PlayerDeviceMixedRunner>().build = JsonConvert.DeserializeObject<PlayerBuildInfo>(marker.text);
        }
        IEnumerator Start()
        {
            var success = false;
            yield return Execute(build, Application.persistentDataPath, Application.streamingAssetsPath, new JObject(), value => success = value);
            // iOS may ignore Quit. External process/lifecycle evidence remains a separate requirement.
            Application.Quit(success ? 0 : 1);
        }
        static byte[] ReadBounded(string path)
        {
            using var stream = File.OpenRead(path);
            if (stream.Length <= 0 || stream.Length > 256 * 1024) throw new InvalidDataException("Device input exceeds its bounded allowance.");
            var bytes = new byte[(int)stream.Length]; var offset = 0;
            while (offset < bytes.Length)
            {
                var count = stream.Read(bytes, offset, bytes.Length - offset);
                if (count == 0) throw new EndOfStreamException("Device input was truncated."); offset += count;
            }
            return bytes;
        }
        public static IEnumerator Execute(PlayerBuildInfo build, string persistentDataPath, string streamingAssetsPath, JObject report, Action<bool> completed)
        {
            if (report == null || report.HasValues || completed == null) throw new ArgumentException("A new report and completion observer are required.");
            report["scope"] = "device_mixed_fixture_ingress_and_original_gpu_protocol";
            report["success"] = false; report["gpu_verified"] = false; report["device_acceptance_completed"] = false;
            report["all_platforms_accepted"] = false; report["final_performance_accepted"] = false;
            report["phase"] = "preflight"; report["started_utc"] = DateTime.UtcNow.ToString("O");
            report["is_editor"] = Application.isEditor; report["platform"] = Application.platform.ToString();
            FileStream output = null; IEnumerator staging = null; var success = false;
            bool Attempt(Action action)
            {
                try { action(); return true; }
                catch (Exception error) { report["error_type"] = error.GetType().Name; return false; }
            }
            bool Save() => output == null || Attempt(() =>
            {
                var bytes = new UTF8Encoding(false).GetBytes(report.ToString()); output.Position = 0;
                output.Write(bytes, 0, bytes.Length); output.SetLength(bytes.Length); output.Flush(true);
            });
            try
            {
                if (!Attempt(() => PlayerMixedRuntimePolicy.Validate(build, Application.platform, SystemInfo.graphicsDeviceType,
                    Application.isEditor, Debug.isDebugBuild, SystemInfo.supportsComputeShaders, Application.unityVersion))) yield break;
                PlayerDeviceMixedConfiguration config = null; PlayerMixedConfiguration mixed = null; byte[] manifest = null;
                if (!Attempt(() =>
                {
                    var path = Path.Combine(persistentDataPath, PlayerDeviceMixedConfiguration.DirectoryName, "run.json");
                    config = PlayerDeviceMixedConfiguration.Parse(new UTF8Encoding(false, true).GetString(ReadBounded(path)), persistentDataPath);
                    mixed = PlayerMixedConfiguration.Parse(config.MixedArguments());
                    if (File.Exists(mixed.Output) || File.Exists(config.StartupReport)) throw new IOException("Earlier run reports cannot be overwritten.");
                    manifest = ReadBounded(config.FixtureManifest);
                    Directory.CreateDirectory(Path.GetDirectoryName(config.StartupReport));
                    output = new FileStream(config.StartupReport, FileMode.CreateNew, FileAccess.Write, FileShare.Read);
                    report["run_id"] = config.RunId; report["precision"] = config.Precision; report["process_phase"] = config.Phase;
                    report["mixed_report"] = "reports/" + config.RunId + ".json";
                    report["build"] = JObject.FromObject(build);
                })) yield break;
                report["phase"] = "stage_original_fixtures"; report["fixture_staging"] = new JObject(); if (!Save()) yield break;
                staging = PlayerFixtureStager.Stage(PlayerDeviceMixedConfiguration.FixtureSource(streamingAssetsPath),
                    mixed.Fixtures, manifest, config.FixtureManifestSha256, (JObject)report["fixture_staging"]);
                while (true)
                {
                    var moved = false; object current = null;
                    if (!Attempt(() => { moved = staging.MoveNext(); if (moved) current = staging.Current; })) yield break;
                    if (!Save()) yield break;
                    if (!moved) break; yield return current;
                }
                var owned = staging; staging = null; if (!Attempt(() => (owned as IDisposable)?.Dispose())) yield break;
                if (!(bool)report["fixture_staging"]["success"]) yield break;
                report["phase"] = "original_mixed_gpu_execution"; if (!Save()) yield break;
                Task<bool> execution = PlayerMixedExecution.Run(build, config.MixedArguments());
                while (!execution.IsCompleted) yield return null;
                if (execution.Status != TaskStatus.RanToCompletion)
                {
                    report["error_type"] = execution.IsCanceled ? "OperationCanceledException" : execution.Exception?.GetBaseException().GetType().Name;
                    yield break;
                }
                success = execution.Result;
            }
            finally
            {
                if (staging != null && !Attempt(() => (staging as IDisposable)?.Dispose())) success = false;
                report["success"] = success; report["gpu_verified"] = success;
                report["phase"] = success ? "complete" : report["error_type"] == null ? "cancelled_or_failed" : "failed";
                report["completed_utc"] = DateTime.UtcNow.ToString("O");
                if (!Save()) { success = false; report["success"] = false; report["gpu_verified"] = false; }
                if (output != null && !Attempt(() => output.Dispose())) success = false;
                completed(success);
            }
        }
    }
}
