using System;
using System.IO;
using EmbeddingGemma.Validation.Editor;
using NUnit.Framework;
using UnityEditor;
using UnityEditor.Build;
using UnityEditor.SceneManagement;
using UnityEngine.Rendering;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class ValidationAndroidBuildTests
    {
        string directory;
        [SetUp] public void Setup() { directory = Path.Combine(Path.GetTempPath(), "embeddinggemma-android-build-" + Guid.NewGuid().ToString("N")); Directory.CreateDirectory(directory); }
        [TearDown] public void Cleanup() { Directory.Delete(directory, true); }
        [TestCase(true)] [TestCase(false)]
        public void PlansAnApkWithOneValidationScene(bool development)
        {
            var path = Path.Combine(directory, "Validation.apk");
            var options = ValidationPlayerBuild.AndroidOptions(path, new string('a', 40), development);
            Assert.That(options.target, Is.EqualTo(BuildTarget.Android)); Assert.That(options.targetGroup, Is.EqualTo(BuildTargetGroup.Android));
            Assert.That(options.options, Is.EqualTo(development ? BuildOptions.Development : BuildOptions.None));
            Assert.That(options.scenes, Is.EqualTo(new[] { ValidationPlayerBuild.ScenePath })); Assert.That(options.locationPathName, Is.EqualTo(path));
        }
        [TestCase("commit")] [TestCase("extension")]
        public void RejectsUnpinnedOrNonApkOutput(string invalid)
            => Assert.Throws<ArgumentException>(() => ValidationPlayerBuild.AndroidOptions(Path.Combine(directory, invalid == "extension" ? "Validation.aab" : "Validation.apk"), invalid == "commit" ? "main" : new string('a', 40)));
        [Test] public void DoesNotOverwriteAnExistingBuild()
        {
            var path = Path.Combine(directory, "existing.txt"); File.WriteAllText(path, "keep");
            Assert.Throws<IOException>(() => ValidationPlayerBuild.AndroidOptions(Path.Combine(directory, "Validation.apk"), new string('a', 40)));
            Assert.That(File.ReadAllText(path), Is.EqualTo("keep"));
        }
        [Test] public void RejectsARequiredProfileCallbackBeforeChangingSettings()
            => Assert.Throws<ArgumentNullException>(() => ValidationPlayerBuild.WithAndroidProfile(null));
        [TestCase(false, false)] [TestCase(true, false)] [TestCase(false, true)] [TestCase(true, true)]
        public void AppliesArm64Il2CppAndRestoresEverySettingEvenWhenBuildFails(bool fail, bool automaticBefore)
        {
            var target = NamedBuildTarget.Android;
            var backend = PlayerSettings.GetScriptingBackend(target); var stripping = PlayerSettings.GetManagedStrippingLevel(target);
            var compiler = PlayerSettings.GetIl2CppCompilerConfiguration(target); var architecture = PlayerSettings.Android.targetArchitectures;
            var identifier = PlayerSettings.GetApplicationIdentifier(target); var min = PlayerSettings.Android.minSdkVersion; var sdk = PlayerSettings.Android.targetSdkVersion;
            var custom = PlayerSettings.Android.useCustomKeystore; var expansion = PlayerSettings.Android.splitApplicationBinary; var perCpu = PlayerSettings.Android.buildApkPerCpuArchitecture;
            var bundle = EditorUserBuildSettings.buildAppBundle; var export = EditorUserBuildSettings.exportAsGoogleAndroidProject;
            var automatic = PlayerSettings.GetUseDefaultGraphicsAPIs(BuildTarget.Android);
            PlayerSettings.SetUseDefaultGraphicsAPIs(BuildTarget.Android, false);
            var apis = PlayerSettings.GetGraphicsAPIs(BuildTarget.Android);
            PlayerSettings.SetUseDefaultGraphicsAPIs(BuildTarget.Android, automatic);
            try
            {
                PlayerSettings.SetScriptingBackend(target, ScriptingImplementation.Mono2x); PlayerSettings.SetManagedStrippingLevel(target, ManagedStrippingLevel.Low);
                PlayerSettings.SetIl2CppCompilerConfiguration(target, Il2CppCompilerConfiguration.Debug); PlayerSettings.Android.targetArchitectures = AndroidArchitecture.ARMv7;
                PlayerSettings.SetApplicationIdentifier(target, "com.example.validation.before"); PlayerSettings.Android.minSdkVersion = AndroidSdkVersions.AndroidApiLevel28;
                PlayerSettings.Android.targetSdkVersion = AndroidSdkVersions.AndroidApiLevel35; PlayerSettings.Android.useCustomKeystore = true;
                PlayerSettings.Android.splitApplicationBinary = true; PlayerSettings.Android.buildApkPerCpuArchitecture = true;
                EditorUserBuildSettings.buildAppBundle = true; EditorUserBuildSettings.exportAsGoogleAndroidProject = true;
                PlayerSettings.SetUseDefaultGraphicsAPIs(BuildTarget.Android, false); PlayerSettings.SetGraphicsAPIs(BuildTarget.Android, new[] { GraphicsDeviceType.OpenGLES3 });
                PlayerSettings.SetUseDefaultGraphicsAPIs(BuildTarget.Android, automaticBefore);
                var calls = 0;
                Action execute = () =>
                {
                    calls++;
                    Assert.That(PlayerSettings.GetScriptingBackend(target), Is.EqualTo(ScriptingImplementation.IL2CPP));
                    Assert.That(PlayerSettings.GetManagedStrippingLevel(target), Is.EqualTo(ManagedStrippingLevel.High));
                    Assert.That(PlayerSettings.GetIl2CppCompilerConfiguration(target), Is.EqualTo(Il2CppCompilerConfiguration.Release));
                    Assert.That(PlayerSettings.Android.targetArchitectures, Is.EqualTo(AndroidArchitecture.ARM64));
                    Assert.That(PlayerSettings.GetApplicationIdentifier(target), Is.EqualTo("com.ayutaz.embeddinggemma.validation"));
                    Assert.That(PlayerSettings.Android.minSdkVersion, Is.EqualTo(AndroidSdkVersions.AndroidApiLevel26));
                    Assert.That(PlayerSettings.Android.targetSdkVersion, Is.EqualTo(AndroidSdkVersions.AndroidApiLevelAuto));
                    Assert.That(PlayerSettings.Android.useCustomKeystore, Is.False);
                    Assert.That(PlayerSettings.Android.splitApplicationBinary, Is.False); Assert.That(PlayerSettings.Android.buildApkPerCpuArchitecture, Is.False);
                    Assert.That(EditorUserBuildSettings.buildAppBundle, Is.False); Assert.That(EditorUserBuildSettings.exportAsGoogleAndroidProject, Is.False);
                    Assert.That(PlayerSettings.GetUseDefaultGraphicsAPIs(BuildTarget.Android), Is.False);
                    Assert.That(PlayerSettings.GetGraphicsAPIs(BuildTarget.Android), Is.EqualTo(new[] { GraphicsDeviceType.Vulkan, GraphicsDeviceType.OpenGLES3 }));
                    if (fail) throw new InvalidOperationException("injected Android failure");
                };
                if (fail) Assert.That(Assert.Throws<InvalidOperationException>(() => ValidationPlayerBuild.WithAndroidProfile(execute)).Message, Is.EqualTo("injected Android failure"));
                else ValidationPlayerBuild.WithAndroidProfile(execute);
                Assert.That(calls, Is.EqualTo(1));
                Assert.That(PlayerSettings.GetScriptingBackend(target), Is.EqualTo(ScriptingImplementation.Mono2x));
                Assert.That(PlayerSettings.GetManagedStrippingLevel(target), Is.EqualTo(ManagedStrippingLevel.Low));
                Assert.That(PlayerSettings.GetIl2CppCompilerConfiguration(target), Is.EqualTo(Il2CppCompilerConfiguration.Debug));
                Assert.That(PlayerSettings.Android.targetArchitectures, Is.EqualTo(AndroidArchitecture.ARMv7));
                Assert.That(PlayerSettings.GetApplicationIdentifier(target), Is.EqualTo("com.example.validation.before"));
                Assert.That(PlayerSettings.Android.minSdkVersion, Is.EqualTo(AndroidSdkVersions.AndroidApiLevel28));
                Assert.That(PlayerSettings.Android.targetSdkVersion, Is.EqualTo(AndroidSdkVersions.AndroidApiLevel35));
                Assert.That(PlayerSettings.Android.useCustomKeystore, Is.True); Assert.That(PlayerSettings.Android.splitApplicationBinary, Is.True);
                Assert.That(PlayerSettings.Android.buildApkPerCpuArchitecture, Is.True); Assert.That(EditorUserBuildSettings.buildAppBundle, Is.True);
                Assert.That(EditorUserBuildSettings.exportAsGoogleAndroidProject, Is.True); Assert.That(PlayerSettings.GetUseDefaultGraphicsAPIs(BuildTarget.Android), Is.EqualTo(automaticBefore));
                PlayerSettings.SetUseDefaultGraphicsAPIs(BuildTarget.Android, false);
                Assert.That(PlayerSettings.GetGraphicsAPIs(BuildTarget.Android), Is.EqualTo(new[] { GraphicsDeviceType.OpenGLES3 }));
            }
            finally
            {
                PlayerSettings.SetScriptingBackend(target, backend); PlayerSettings.SetManagedStrippingLevel(target, stripping); PlayerSettings.SetIl2CppCompilerConfiguration(target, compiler);
                PlayerSettings.Android.targetArchitectures = architecture; PlayerSettings.SetApplicationIdentifier(target, identifier); PlayerSettings.Android.minSdkVersion = min;
                PlayerSettings.Android.targetSdkVersion = sdk; PlayerSettings.Android.useCustomKeystore = custom; PlayerSettings.Android.splitApplicationBinary = expansion;
                PlayerSettings.Android.buildApkPerCpuArchitecture = perCpu; EditorUserBuildSettings.buildAppBundle = bundle; EditorUserBuildSettings.exportAsGoogleAndroidProject = export;
                PlayerSettings.SetGraphicsAPIs(BuildTarget.Android, apis); PlayerSettings.SetUseDefaultGraphicsAPIs(BuildTarget.Android, automatic);
            }
        }
        [Test] public void WrongActivePlatformCannotGenerateOrModifyAssets()
        {
            Assert.That(EditorUserBuildSettings.activeBuildTarget, Is.Not.EqualTo(BuildTarget.Android), "Run this rejection contract before switching the Editor to Android.");
            var before = EditorSceneManager.GetSceneManagerSetup();
            var calls = 0;
            Assert.Throws<InvalidOperationException>(() => ValidationPlayerBuild.BuildAndroid(Path.Combine(directory, "Validation.apk"), new string('a', 40), _ => calls++));
            Assert.That(calls, Is.Zero); Assert.That(File.Exists(Path.Combine(directory, "build.json")), Is.False);
            Assert.That(File.Exists(ValidationPlayerBuild.MarkerPath), Is.False); Assert.That(File.Exists(ValidationPlayerBuild.ScenePath), Is.False);
            Assert.That(EditorSceneManager.GetSceneManagerSetup().Length, Is.EqualTo(before.Length));
        }
        [TestCase(false)] [TestCase(true)]
        public void BuildRecordsAndroidProvenanceAndRestoresTheSceneAndSettings(bool fail)
        {
            Assert.That(EditorUserBuildSettings.activeBuildTarget, Is.EqualTo(BuildTarget.Android), "Run these integration contracts after the explicit Android target switch.");
            var target = NamedBuildTarget.Android;
            var backend = PlayerSettings.GetScriptingBackend(target); var stripping = PlayerSettings.GetManagedStrippingLevel(target);
            var compiler = PlayerSettings.GetIl2CppCompilerConfiguration(target); var architecture = PlayerSettings.Android.targetArchitectures;
            var identifier = PlayerSettings.GetApplicationIdentifier(target); var scene = UnityEngine.SceneManagement.SceneManager.GetActiveScene().path;
            var calls = 0;
            Action<BuildPlayerOptions> execute = options =>
            {
                calls++; Assert.That(options.target, Is.EqualTo(BuildTarget.Android)); Assert.That(options.options, Is.EqualTo(BuildOptions.Development));
                Assert.That(PlayerSettings.Android.targetArchitectures, Is.EqualTo(AndroidArchitecture.ARM64));
                Assert.That(PlayerSettings.GetScriptingBackend(target), Is.EqualTo(ScriptingImplementation.IL2CPP));
                Assert.That(File.Exists(ValidationPlayerBuild.MarkerPath), Is.True);
                Assert.That(UnityEngine.SceneManagement.SceneManager.GetActiveScene().path, Is.EqualTo(ValidationPlayerBuild.ScenePath));
                if (fail) throw new InvalidOperationException("injected Android build failure");
            };
            if (fail) Assert.That(Assert.Throws<InvalidOperationException>(() => ValidationPlayerBuild.BuildAndroid(Path.Combine(directory, "Validation.apk"), new string('a', 40), execute)).Message, Is.EqualTo("injected Android build failure"));
            else
            {
                var info = ValidationPlayerBuild.BuildAndroid(Path.Combine(directory, "Validation.apk"), new string('a', 40), execute);
                Assert.That(info.target, Is.EqualTo("Android")); Assert.That(info.scriptingBackend, Is.EqualTo("IL2CPP")); Assert.That(info.stripping, Is.EqualTo("High"));
                Assert.That(info.sourceSha256.HasValues, Is.True);
            }
            Assert.That(calls, Is.EqualTo(1));
            Assert.That(UnityEngine.SceneManagement.SceneManager.GetActiveScene().path, Is.EqualTo(scene));
            Assert.That(File.Exists(ValidationPlayerBuild.MarkerPath), Is.False); Assert.That(File.Exists(ValidationPlayerBuild.ScenePath), Is.False);
            Assert.That(PlayerSettings.GetScriptingBackend(target), Is.EqualTo(backend)); Assert.That(PlayerSettings.GetManagedStrippingLevel(target), Is.EqualTo(stripping));
            Assert.That(PlayerSettings.GetIl2CppCompilerConfiguration(target), Is.EqualTo(compiler)); Assert.That(PlayerSettings.Android.targetArchitectures, Is.EqualTo(architecture));
            Assert.That(PlayerSettings.GetApplicationIdentifier(target), Is.EqualTo(identifier));
            var receipt = Newtonsoft.Json.Linq.JObject.Parse(File.ReadAllText(Path.Combine(directory, "build.json")));
            Assert.That((bool)receipt["success"], Is.False, "Injected builds never prove APK generation."); Assert.That((bool)receipt["injected_build"], Is.True);
            Assert.That((string)receipt["build_info"]["target"], Is.EqualTo("Android")); Assert.That((string)receipt["il2cpp_compiler"], Is.EqualTo("Release"));
            Assert.That((string)receipt["target_settings"]["architectures"], Is.EqualTo("ARM64")); Assert.That((bool)receipt["target_settings"]["app_bundle"], Is.False);
            Assert.That(receipt["completed_utc"], Is.Not.Null);
        }
    }
}
