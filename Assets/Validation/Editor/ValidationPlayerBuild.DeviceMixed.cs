using System;
using UnityEditor;
using UnityEditor.Build;

namespace EmbeddingGemma.Validation.Editor
{
    public static partial class ValidationPlayerBuild
    {
        public const string DeviceMixedMarkerPath = "Assets/Validation/Generated/Resources/EmbeddingGemmaDeviceMixedValidationBuild.json";
        public static PlayerBuildInfo BuildAndroidMixed(string output, string codeCommit, Action<BuildPlayerOptions> execute = null)
        {
            var options = AndroidOptions(output, codeCommit, development: false); EnsureBuildState(BuildTarget.Android);
            var profile = new AndroidProfile();
            return BuildTargetPlayer(options, codeCommit, NamedBuildTarget.Android, execute, configure: profile.Apply, restore: profile.Restore,
                markerOverride: DeviceMixedMarkerPath, details: () => { var data = profile.Describe(); data["validation_kind"] = "device_all_media_roles_and_mixed_index"; return data; });
        }
        public static PlayerBuildInfo BuildIOSMixed(string output, string codeCommit, Action<BuildPlayerOptions> execute = null)
        {
            var options = IOSOptions(output, codeCommit); EnsureBuildState(BuildTarget.iOS);
            var profile = new IOSProfile();
            return BuildTargetPlayer(options, codeCommit, NamedBuildTarget.iOS, execute, configure: profile.Apply, restore: profile.Restore,
                markerOverride: DeviceMixedMarkerPath, details: () => { var data = IOSProfileSettings(); data["validation_kind"] = "device_all_media_roles_and_mixed_index"; return data; });
        }
    }
}
