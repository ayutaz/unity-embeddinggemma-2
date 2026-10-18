using System;
using Newtonsoft.Json.Linq;
using UnityEditor;
using UnityEditor.Build;

namespace EmbeddingGemma.Validation.Editor
{
    public static partial class ValidationPlayerBuild
    {
        public const string MediaMarkerPath = "Assets/Validation/Generated/Resources/EmbeddingGemmaMediaValidationBuild.json";
        public static PlayerBuildInfo BuildWindowsMedia(string output, string codeCommit, Action<BuildPlayerOptions> execute = null)
            => BuildTargetPlayer(WindowsOptions(output, codeCommit, development: false), codeCommit, NamedBuildTarget.Standalone, execute,
                releaseIl2Cpp: true, markerOverride: MediaMarkerPath, details: () => new JObject { ["validation_kind"] = "fixed_document_media_module" });
    }
}
