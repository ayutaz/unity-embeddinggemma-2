using System;
using System.Collections;
using System.IO;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using UnityEngine;

namespace EmbeddingGemma.Validation
{
    public sealed class PlayerSampleRunner : MonoBehaviour
    {
        PlayerBuildInfo build;
        public static bool ShouldBootstrap(bool editor, bool sampleMarker) => !editor && sampleMarker;
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        static void Bootstrap()
        {
            var marker = Resources.Load<TextAsset>("EmbeddingGemmaSampleValidationBuild");
            if (!ShouldBootstrap(Application.isEditor, marker != null)) return;
            Application.runInBackground = true;
            var owner = new GameObject("EmbeddingGemma Sample Player Validation"); DontDestroyOnLoad(owner);
            owner.AddComponent<PlayerSampleRunner>().build = JsonConvert.DeserializeObject<PlayerBuildInfo>(marker.text);
        }
        IEnumerator Start()
        {
            yield return null;
            PlayerRunConfiguration config = null; PlayerReportFile writer = null;
            try
            {
                var output = Path.Combine(Application.persistentDataPath, "EmbeddingGemmaSampleValidation", Guid.NewGuid().ToString("N"), "results.json");
                config = PlayerRunConfiguration.Parse(Environment.GetCommandLineArgs(), Path.Combine(Application.streamingAssetsPath, "EmbeddingGemmaTextSearch"), output);
                writer = new PlayerReportFile(config.Output);
            }
            catch (Exception exception) { Debug.LogException(exception); }
            if (writer == null) { Application.Quit(1); yield break; }
            JObject result = null;
            yield return PlayerSampleExecution.Run(config, build, writer.Save, report => result = report);
            // Let deferred view/font destruction complete before requesting engine shutdown.
            yield return null;
            Debug.Log("EmbeddingGemma sample validation: " + config.Output);
            Application.Quit(result != null && (bool)result["success"] ? 0 : 1);
        }
    }
}
