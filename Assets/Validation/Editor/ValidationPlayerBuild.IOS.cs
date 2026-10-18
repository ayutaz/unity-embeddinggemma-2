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
        public static BuildPlayerOptions IOSOptions(string output, string codeCommit)
        {
            if (codeCommit == null || !Regex.IsMatch(codeCommit, "\\A[a-f0-9]{40}\\z") || string.IsNullOrWhiteSpace(output))
                throw new ArgumentException("A pinned source commit and new Xcode export directory are required.");
            output = Path.GetFullPath(output);
            if (output == Path.GetPathRoot(output)) throw new ArgumentException("A new child export directory is required.");
            output = output.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
            if (output.EndsWith(".ipa", StringComparison.OrdinalIgnoreCase) || output.EndsWith(".xcodeproj", StringComparison.OrdinalIgnoreCase))
                throw new ArgumentException("Select an export directory, not an IPA or Xcode project file.");
            var directory = Path.GetDirectoryName(output);
            if (File.Exists(output) || Directory.Exists(directory) && Directory.EnumerateFileSystemEntries(directory).Any())
                throw new IOException("Build output parent must be empty; existing exports and receipts are never overwritten.");
            return new BuildPlayerOptions { scenes = new[] { ScenePath }, locationPathName = output, target = BuildTarget.iOS,
                targetGroup = BuildTargetGroup.iOS, options = BuildOptions.None };
        }

        public static void WithIOSProfile(Action execute)
        {
            if (execute == null) throw new ArgumentNullException(nameof(execute));
            var profile = new IOSProfile();
            try { profile.Apply(); execute(); }
            finally { profile.Restore(); }
        }

        public static PlayerBuildInfo BuildIOS(string output, string codeCommit, Action<BuildPlayerOptions> execute = null)
        {
            var options = IOSOptions(output, codeCommit);
            // Reject a missing/incorrect target before creating output, scenes or profiles.
            EnsureBuildState(BuildTarget.iOS);
            var profile = new IOSProfile();
            return BuildTargetPlayer(options, codeCommit, NamedBuildTarget.iOS, execute,
                configure: profile.Apply, restore: profile.Restore, details: IOSProfileSettings);
        }

        // Deliberate allowlist: signing identities, provisioning values and local export paths stay private.
        public static JObject IOSProfileSettings() => new JObject
        {
            ["application_identifier"] = PlayerSettings.GetApplicationIdentifier(NamedBuildTarget.iOS),
            ["sdk"] = PlayerSettings.iOS.sdkVersion.ToString(), ["device_family"] = PlayerSettings.iOS.targetDevice.ToString(),
            ["deployment_target"] = PlayerSettings.iOS.targetOSVersionString,
            ["automatic_graphics_api"] = PlayerSettings.GetUseDefaultGraphicsAPIs(BuildTarget.iOS),
            ["graphics_apis"] = new JArray(PlayerSettings.GetGraphicsAPIs(BuildTarget.iOS).Select(api => api.ToString())),
            ["signing_scope"] = "Private Xcode signing; identities and provisioning values omitted"
        };

        sealed class IOSProfile
        {
            readonly Action restore;
            public IOSProfile()
            {
                var target = NamedBuildTarget.iOS;
                var backend = PlayerSettings.GetScriptingBackend(target); var stripping = PlayerSettings.GetManagedStrippingLevel(target);
                var compiler = PlayerSettings.GetIl2CppCompilerConfiguration(target); var identifier = PlayerSettings.GetApplicationIdentifier(target);
                var sdk = PlayerSettings.iOS.sdkVersion; var device = PlayerSettings.iOS.targetDevice; var minimum = PlayerSettings.iOS.targetOSVersionString;
                var automatic = PlayerSettings.GetUseDefaultGraphicsAPIs(BuildTarget.iOS);
                GraphicsDeviceType[] apis;
                try { PlayerSettings.SetUseDefaultGraphicsAPIs(BuildTarget.iOS, false); apis = PlayerSettings.GetGraphicsAPIs(BuildTarget.iOS); }
                finally { PlayerSettings.SetUseDefaultGraphicsAPIs(BuildTarget.iOS, automatic); }
                restore = () =>
                {
                    PlayerSettings.SetScriptingBackend(target, backend); PlayerSettings.SetManagedStrippingLevel(target, stripping);
                    PlayerSettings.SetIl2CppCompilerConfiguration(target, compiler); PlayerSettings.SetApplicationIdentifier(target, identifier);
                    PlayerSettings.iOS.sdkVersion = sdk; PlayerSettings.iOS.targetDevice = device; PlayerSettings.iOS.targetOSVersionString = minimum;
                    PlayerSettings.SetGraphicsAPIs(BuildTarget.iOS, apis); PlayerSettings.SetUseDefaultGraphicsAPIs(BuildTarget.iOS, automatic);
                };
            }
            public void Apply()
            {
                var target = NamedBuildTarget.iOS;
                PlayerSettings.SetScriptingBackend(target, ScriptingImplementation.IL2CPP); PlayerSettings.SetManagedStrippingLevel(target, ManagedStrippingLevel.High);
                PlayerSettings.SetIl2CppCompilerConfiguration(target, Il2CppCompilerConfiguration.Release);
                PlayerSettings.SetApplicationIdentifier(target, "com.ayutaz.embeddinggemma.validation");
                PlayerSettings.iOS.sdkVersion = iOSSdkVersion.DeviceSDK; PlayerSettings.iOS.targetDevice = iOSTargetDevice.iPhoneAndiPad;
                // Probe build configuration, not a measured minimum supported OS/device claim.
                PlayerSettings.iOS.targetOSVersionString = "15.0";
                PlayerSettings.SetUseDefaultGraphicsAPIs(BuildTarget.iOS, false);
                PlayerSettings.SetGraphicsAPIs(BuildTarget.iOS, new[] { GraphicsDeviceType.Metal });
                // Do not read, copy, change or publish the caller's signing/provisioning values.
            }
            public void Restore() => restore();
        }
    }
}
