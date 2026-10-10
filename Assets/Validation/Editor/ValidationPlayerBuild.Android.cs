using System;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using Newtonsoft.Json.Linq;
using UnityEditor;
using UnityEditor.Build;
using UnityEngine.Rendering;

namespace EmbeddingGemma.Validation.Editor
{
    public static partial class ValidationPlayerBuild
    {
        public static BuildPlayerOptions AndroidOptions(string output, string codeCommit, bool development = true)
        {
            if (codeCommit == null || !Regex.IsMatch(codeCommit, "\\A[a-f0-9]{40}\\z") || string.IsNullOrWhiteSpace(output) ||
                !output.EndsWith(".apk", StringComparison.OrdinalIgnoreCase)) throw new ArgumentException("A fixed source commit and .apk output are required.");
            output = Path.GetFullPath(output);
            var directory = Path.GetDirectoryName(output);
            if (Directory.Exists(directory) && Directory.EnumerateFileSystemEntries(directory).Any()) throw new IOException("Build output directory must be empty.");
            return new BuildPlayerOptions { scenes = new[] { ScenePath }, locationPathName = output, target = BuildTarget.Android,
                targetGroup = BuildTargetGroup.Android, options = development ? BuildOptions.Development : BuildOptions.None };
        }
        public static void WithAndroidProfile(Action execute)
        {
            if (execute == null) throw new ArgumentNullException(nameof(execute));
            var profile = new AndroidProfile();
            try { profile.Apply(); execute(); }
            finally { profile.Restore(); }
        }
        public static PlayerBuildInfo BuildAndroid(string output, string codeCommit, Action<BuildPlayerOptions> execute = null, bool development = true)
        {
            var options = AndroidOptions(output, codeCommit, development);
            EnsureBuildState(BuildTarget.Android);
            var profile = new AndroidProfile();
            return BuildTargetPlayer(options, codeCommit, NamedBuildTarget.Android, execute,
                configure: profile.Apply, restore: profile.Restore, details: profile.Describe);
        }

        sealed class AndroidProfile
        {
            readonly Action restore;
            public AndroidProfile()
            {
                var target = NamedBuildTarget.Android;
                var backend = PlayerSettings.GetScriptingBackend(target); var stripping = PlayerSettings.GetManagedStrippingLevel(target);
                var compiler = PlayerSettings.GetIl2CppCompilerConfiguration(target); var architecture = PlayerSettings.Android.targetArchitectures;
                var identifier = PlayerSettings.GetApplicationIdentifier(target); var min = PlayerSettings.Android.minSdkVersion; var sdk = PlayerSettings.Android.targetSdkVersion;
                var custom = PlayerSettings.Android.useCustomKeystore; var expansion = PlayerSettings.Android.splitApplicationBinary; var perCpu = PlayerSettings.Android.buildApkPerCpuArchitecture;
                var bundle = EditorUserBuildSettings.buildAppBundle; var export = EditorUserBuildSettings.exportAsGoogleAndroidProject;
                var automatic = PlayerSettings.GetUseDefaultGraphicsAPIs(BuildTarget.Android);
                GraphicsDeviceType[] apis;
                // With automatic selection, GetGraphicsAPIs returns the default list, hiding the saved explicit list.
                try { PlayerSettings.SetUseDefaultGraphicsAPIs(BuildTarget.Android, false); apis = PlayerSettings.GetGraphicsAPIs(BuildTarget.Android); }
                finally { PlayerSettings.SetUseDefaultGraphicsAPIs(BuildTarget.Android, automatic); }
                restore = () =>
                {
                    PlayerSettings.SetScriptingBackend(target, backend); PlayerSettings.SetManagedStrippingLevel(target, stripping); PlayerSettings.SetIl2CppCompilerConfiguration(target, compiler);
                    PlayerSettings.Android.targetArchitectures = architecture; PlayerSettings.SetApplicationIdentifier(target, identifier); PlayerSettings.Android.minSdkVersion = min;
                    PlayerSettings.Android.targetSdkVersion = sdk; PlayerSettings.Android.useCustomKeystore = custom; PlayerSettings.Android.splitApplicationBinary = expansion;
                    PlayerSettings.Android.buildApkPerCpuArchitecture = perCpu; EditorUserBuildSettings.buildAppBundle = bundle; EditorUserBuildSettings.exportAsGoogleAndroidProject = export;
                    PlayerSettings.SetGraphicsAPIs(BuildTarget.Android, apis); PlayerSettings.SetUseDefaultGraphicsAPIs(BuildTarget.Android, automatic);
                };
            }
            public void Apply()
            {
                var target = NamedBuildTarget.Android;
                PlayerSettings.SetScriptingBackend(target, ScriptingImplementation.IL2CPP); PlayerSettings.SetManagedStrippingLevel(target, ManagedStrippingLevel.High);
                PlayerSettings.SetIl2CppCompilerConfiguration(target, Il2CppCompilerConfiguration.Release); PlayerSettings.Android.targetArchitectures = AndroidArchitecture.ARM64;
                PlayerSettings.SetApplicationIdentifier(target, "com.ayutaz.embeddinggemma.validation"); PlayerSettings.Android.minSdkVersion = AndroidSdkVersions.AndroidApiLevel26;
                PlayerSettings.Android.targetSdkVersion = AndroidSdkVersions.AndroidApiLevelAuto; PlayerSettings.Android.useCustomKeystore = false;
                PlayerSettings.Android.splitApplicationBinary = false; PlayerSettings.Android.buildApkPerCpuArchitecture = false;
                EditorUserBuildSettings.buildAppBundle = false; EditorUserBuildSettings.exportAsGoogleAndroidProject = false;
                PlayerSettings.SetUseDefaultGraphicsAPIs(BuildTarget.Android, false);
                PlayerSettings.SetGraphicsAPIs(BuildTarget.Android, new[] { GraphicsDeviceType.Vulkan, GraphicsDeviceType.OpenGLES3 });
            }
            public void Restore() => restore();
            public JObject Describe() => new JObject
            {
                ["architectures"] = PlayerSettings.Android.targetArchitectures.ToString(),
                ["application_identifier"] = PlayerSettings.GetApplicationIdentifier(NamedBuildTarget.Android),
                ["min_sdk"] = (int)PlayerSettings.Android.minSdkVersion, ["target_sdk"] = PlayerSettings.Android.targetSdkVersion.ToString(),
                ["custom_keystore"] = PlayerSettings.Android.useCustomKeystore, ["apk_expansion"] = PlayerSettings.Android.splitApplicationBinary,
                ["apk_per_cpu"] = PlayerSettings.Android.buildApkPerCpuArchitecture, ["app_bundle"] = EditorUserBuildSettings.buildAppBundle,
                ["export_project"] = EditorUserBuildSettings.exportAsGoogleAndroidProject,
                ["automatic_graphics_api"] = PlayerSettings.GetUseDefaultGraphicsAPIs(BuildTarget.Android),
                ["graphics_apis"] = new JArray(PlayerSettings.GetGraphicsAPIs(BuildTarget.Android).Select(api => api.ToString()))
            };
        }
    }
}
