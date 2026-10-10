using System;
using System.Collections;
using System.IO;
using Newtonsoft.Json;
using UnityEngine;

namespace EmbeddingGemma.Validation
{
    /// <summary>Opt-in validation Player bootstrap; no marker means ordinary Players are unaffected.</summary>
    public sealed class PlayerValidationRunner : MonoBehaviour
    {
        PlayerBuildInfo build;
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        static void Bootstrap()
        {
            if (Application.isEditor) return;
            var marker = Resources.Load<TextAsset>("EmbeddingGemmaValidationBuild");
            if (marker == null) return;
            var owner = new GameObject("EmbeddingGemma Player Validation"); DontDestroyOnLoad(owner);
            var runner = owner.AddComponent<PlayerValidationRunner>();
            runner.build = JsonConvert.DeserializeObject<PlayerBuildInfo>(marker.text);
        }
        IEnumerator Start()
        {
            yield return null;
            PlayerRunConfiguration config = null;
            PlayerReportFile writer = null;
            try
            {
                var defaultOutput = Path.Combine(Application.persistentDataPath, "EmbeddingGemmaValidation", Guid.NewGuid().ToString("N"), "results.json");
                config = PlayerRunConfiguration.Parse(Environment.GetCommandLineArgs(), Application.streamingAssetsPath + "/EmbeddingGemmaValidation", defaultOutput);
                writer = new PlayerReportFile(config.Output);
            }
            catch (Exception exception) { Debug.LogException(exception); }
            if (writer == null) { Application.Quit(1); yield break; }
            PlayerRunReport result = null;
            var staging = Path.Combine(Application.persistentDataPath, "EmbeddingGemmaValidation", "BundleCache");
            yield return PlayerValidationExecution.Run(config, build, staging, writer.Save, report => result = report);
            if (result != null) Debug.Log("EmbeddingGemma validation " + result.phase + ": " + config.Output);
            Application.Quit(result != null && result.success ? 0 : 1);
        }
    }
}
