using System;
using System.Collections;
using Unity.InferenceEngine;
using UnityEngine;

namespace EmbeddingGemma.Validation
{
    public static class PlayerValidationExecution
    {
        public static IEnumerator Run(PlayerRunConfiguration config, PlayerBuildInfo build, string destination,
            Action<PlayerRunReport> save, Action<PlayerRunReport> complete,
            Func<string, string, IEnumerator> transfer = null, Func<string, PlayerBundle> load = null,
            Func<string, (int[] ids, int[] mask)> encode = null,
            Func<PlayerBundle, string, BackendType, IPlayerEmbedder> create = null)
        {
            var stage = new PlayerBundleStage { source = config.Bundle };
            var result = new PlayerRunReport { runId = config.RunId, startedUtc = DateTime.UtcNow.ToString("O"), phase = "bundle_staging", build = build,
                staging = stage, isEditor = Application.isEditor, unity = Application.unityVersion, os = SystemInfo.operatingSystem,
                gpu = SystemInfo.graphicsDeviceName, graphicsApi = SystemInfo.graphicsDeviceType.ToString(),
                validationMode = transfer != null || load != null || encode != null || create != null ? "injected_contract" : "real_model" };
            try
            {
                build.Validate();
                if (build.unityVersion != Application.unityVersion) throw new ArgumentException("Running Unity version differs from build provenance.");
                save(result);
            }
            catch (Exception exception) { result.error = exception.GetType().Name + ": " + exception.Message; }
            if (result.error == null)
            {
                yield return PlayerBundleStager.Stage(config.Bundle, destination, stage, transfer);
                if (!stage.transferCompleted) result.error = stage.error ?? "Bundle staging did not complete.";
            }
            if (result.error != null)
            {
                result.phase = "failed"; result.completedUtc = DateTime.UtcNow.ToString("O");
                try { save(result); }
                catch (Exception exception) { result.error += " Result write: " + exception.Message; }
            }
            else
            {
                var resolved = new PlayerRunConfiguration { Bundle = stage.directory, Output = config.Output, RunId = config.RunId };
                var started = result.startedUtc;
                result = PlayerRunProtocol.Run(resolved, build, report => { report.startedUtc = started; report.staging = stage; save(report); }, load, encode, create);
            }
            complete(result);
        }
    }
}
