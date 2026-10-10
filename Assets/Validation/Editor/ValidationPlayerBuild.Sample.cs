using System;
using System.IO;
using Newtonsoft.Json.Linq;
using UnityEditor;
using UnityEditor.Build;

namespace EmbeddingGemma.Validation.Editor
{
    public static partial class ValidationPlayerBuild
    {
        public const string SampleScenePath = "Assets/EmbeddingGemmaTextSearch/TextSearch.unity";
        public const string SampleMarkerPath = "Assets/Validation/Generated/Resources/EmbeddingGemmaSampleValidationBuild.json";
        public static BuildPlayerOptions WindowsSampleOptions(string output, string codeCommit)
        {
            var options = WindowsOptions(output, codeCommit, development: false);
            if (!File.Exists(SampleScenePath)) throw new IOException("Import the TextSearch sample before building its Player.");
            options.scenes = new[] { SampleScenePath };
            return options;
        }
        public static PlayerBuildInfo BuildWindowsSample(string output, string codeCommit, Action<BuildPlayerOptions> execute = null)
            => BuildTargetPlayer(WindowsSampleOptions(output, codeCommit), codeCommit, NamedBuildTarget.Standalone, execute,
                releaseIl2Cpp: true, details: () => new JObject { ["validation_kind"] = "sample", ["sample_scene"] = SampleScenePath }, sampleScene: true);
    }
}
