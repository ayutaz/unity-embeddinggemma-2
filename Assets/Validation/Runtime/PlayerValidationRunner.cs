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
            var exitCode = 1;
            try
            {
                var defaultOutput = Path.Combine(Application.persistentDataPath, "EmbeddingGemmaValidation", Guid.NewGuid().ToString("N"), "results.json");
                var config = PlayerRunConfiguration.Parse(Environment.GetCommandLineArgs(), Application.streamingAssetsPath + "/EmbeddingGemmaValidation", defaultOutput);
                var writer = new PlayerReportFile(config.Output);
                var result = PlayerRunProtocol.Run(config, build, writer.Save);
                Debug.Log("EmbeddingGemma validation " + result.phase + ": " + config.Output);
                exitCode = result.success ? 0 : 1;
            }
            catch (Exception exception) { Debug.LogException(exception); }
            Application.Quit(exitCode);
        }
    }
}
