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
    /// <summary>Opt-in fixed image/audio input runner; ordinary and text validation Players have no media marker.</summary>
    public sealed class PlayerMediaRunner : MonoBehaviour
    {
        PlayerBuildInfo build;
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        static void Bootstrap()
        {
            if (Application.isEditor) return;
            var marker = Resources.Load<TextAsset>("EmbeddingGemmaMediaValidationBuild");
            if (marker == null) return;
            var owner = new GameObject("EmbeddingGemma Media Player Validation"); DontDestroyOnLoad(owner);
            owner.AddComponent<PlayerMediaRunner>().build = JsonConvert.DeserializeObject<PlayerBuildInfo>(marker.text);
        }
        IEnumerator Start()
        {
            yield return null;
            var task = Execute();
            while (!task.IsCompleted) yield return null;
            var success = task.Status == TaskStatus.RanToCompletion && task.Result;
            if (task.IsFaulted) Debug.LogError("Media Player validation could not save a complete result.");
            Application.Quit(success ? 0 : 1);
        }
        async Task<bool> Execute()
        {
            var config = PlayerMediaConfiguration.Parse(Environment.GetCommandLineArgs());
            Directory.CreateDirectory(Path.GetDirectoryName(config.Output));
            // Hold an exclusive new handle for the whole run. Never truncate an earlier process's evidence.
            using var output = new FileStream(config.Output, FileMode.CreateNew, FileAccess.Write, FileShare.Read);
            var envelope = new JObject { ["schema_version"] = 1, ["success"] = false, ["gpu_verified"] = false,
                ["full_multimodal_acceptance"] = false, ["model_generation_qualified"] = false,
                ["scope"] = "fixed_document_media_module_player", ["run_id"] = config.RunId, ["module"] = config.Module,
                ["precision"] = config.Precision, ["requested_bundle_descriptor_sha256"] = config.DescriptorSha256,
                ["started_utc"] = DateTime.UtcNow.ToString("O"), ["build"] = JObject.FromObject(build),
                ["unity"] = Application.unityVersion, ["graphics_api"] = SystemInfo.graphicsDeviceType.ToString(), ["gpu"] = SystemInfo.graphicsDeviceName };
            void Save()
            {
                var bytes = new UTF8Encoding(false).GetBytes(envelope.ToString());
                output.Position = 0; output.Write(bytes, 0, bytes.Length); output.SetLength(bytes.Length); output.Flush(true);
            }
            PlayerMediaFixture fixtures = null;
            try
            {
                envelope["phase"] = "preflight"; Save(); build.Validate();
                if (Application.isEditor || Application.platform != RuntimePlatform.WindowsPlayer ||
                    !SystemInfo.supportsComputeShaders || build.target != "StandaloneWindows64" || build.scriptingBackend != "IL2CPP" ||
                    build.stripping != "High" || Debug.isDebugBuild || Application.unityVersion != build.unityVersion)
                    throw new NotSupportedException("An actual Windows Release IL2CPP High Player and compute GPU are required.");
                using (var descriptor = File.OpenRead(Path.Combine(config.Bundle, "bundle.json")))
                {
                    var sha = PlayerFileHash.Compute(descriptor, out _); envelope["bundle_descriptor_sha256"] = sha;
                    if (sha != config.DescriptorSha256) throw new InvalidDataException("Prepared descriptor differs from the explicitly selected candidate.");
                }
                var bundle = GpuMediaBundle.Open(config.Bundle);
                envelope["phase"] = "verify_original_fixtures"; Save();
                fixtures = PlayerMediaFixture.Load(config.Fixtures, config.Module, bundle.Module(config.Module));
                envelope["fixtures"] = fixtures.Evidence;
                envelope["execution"] = await PlayerMediaProtocol.Run(fixtures.Cases, config.Precision,
                    () => PlayerPreparedMediaModule.Load(bundle, config.Module), report => { envelope["execution"] = report; Save(); });
                envelope["success"] = (bool)envelope["execution"]["success"];
                envelope["gpu_verified"] = (bool)envelope["execution"]["gpu_verified"];
            }
            catch (Exception error)
            {
                envelope["success"] = false; envelope["gpu_verified"] = false;
                envelope["error_type"] = error.GetType().Name;
                // This raw log is private evidence and is not an upload payload.
                Debug.LogException(error);
            }
            finally
            {
                try { fixtures?.Dispose(); }
                catch (Exception error) { envelope["success"] = false; envelope["gpu_verified"] = false; envelope["fixture_release_error_type"] = error.GetType().Name; }
                envelope["phase"] = (bool)envelope["success"] ? "complete" : "failed";
                envelope["completed_utc"] = DateTime.UtcNow.ToString("O"); Save();
            }
            return (bool)envelope["success"] && (bool)envelope["gpu_verified"];
        }
    }
}
