using System;
using System.IO;
using System.Linq;
using EmbeddingGemma.Validation.Editor;
using NUnit.Framework;
using UnityEditor;
using UnityEditor.Build;
using UnityEditor.SceneManagement;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class ValidationPlayerBuildTests
    {
        string directory;
        [SetUp] public void Setup() { directory = Path.Combine(Path.GetTempPath(), "embeddinggemma-build-" + Guid.NewGuid().ToString("N")); Directory.CreateDirectory(directory); }
        [TearDown] public void Cleanup() { Directory.Delete(directory, true); }
        [Test] public void PlansAWindowsDevelopmentBuildWithOneValidationScene()
        {
            var options = ValidationPlayerBuild.WindowsOptions(Path.Combine(directory, "Validation.exe"), new string('a', 40));
            Assert.That(options.target, Is.EqualTo(BuildTarget.StandaloneWindows64)); Assert.That(options.options.HasFlag(BuildOptions.Development), Is.True);
            Assert.That(options.scenes, Is.EqualTo(new[] { ValidationPlayerBuild.ScenePath })); Assert.That(options.locationPathName, Is.EqualTo(Path.Combine(directory, "Validation.exe")));
        }
        [Test] public void PlansAReleaseBuildWithoutDevelopmentOptions()
        {
            var options = ValidationPlayerBuild.WindowsOptions(Path.Combine(directory, "Validation.exe"), new string('a', 40), development: false);
            Assert.That(options.options, Is.EqualTo(BuildOptions.None));
            Assert.That(options.scenes, Is.EqualTo(new[] { ValidationPlayerBuild.ScenePath }));
        }
        [TestCase(false)] [TestCase(true)]
        public void ReleaseBuildSelectsIl2CppHighAndRestoresAllSettings(bool fail)
        {
            var target = NamedBuildTarget.Standalone;
            var backend = PlayerSettings.GetScriptingBackend(target);
            var stripping = PlayerSettings.GetManagedStrippingLevel(target);
            var compiler = PlayerSettings.GetIl2CppCompilerConfiguration(target);
            var calls = 0;
            try
            {
                // Deliberately differ from the requested profile so restoration is observable.
                PlayerSettings.SetScriptingBackend(target, ScriptingImplementation.Mono2x);
                PlayerSettings.SetManagedStrippingLevel(target, ManagedStrippingLevel.Low);
                PlayerSettings.SetIl2CppCompilerConfiguration(target, Il2CppCompilerConfiguration.Debug);
                Action<BuildPlayerOptions> execute = options =>
                {
                    calls++;
                    Assert.That(options.options, Is.EqualTo(BuildOptions.None));
                    Assert.That(PlayerSettings.GetScriptingBackend(target), Is.EqualTo(ScriptingImplementation.IL2CPP));
                    Assert.That(PlayerSettings.GetManagedStrippingLevel(target), Is.EqualTo(ManagedStrippingLevel.High));
                    Assert.That(PlayerSettings.GetIl2CppCompilerConfiguration(target), Is.EqualTo(Il2CppCompilerConfiguration.Release));
                    if (fail) throw new InvalidOperationException("injected release failure");
                };
                if (fail)
                    Assert.That(Assert.Throws<InvalidOperationException>(() => ValidationPlayerBuild.BuildWindows(Path.Combine(directory, "Validation.exe"), new string('a', 40), execute, releaseIl2Cpp: true)).Message,
                        Is.EqualTo("injected release failure"));
                else
                {
                    var info = ValidationPlayerBuild.BuildWindows(Path.Combine(directory, "Validation.exe"), new string('a', 40), execute, releaseIl2Cpp: true);
                    Assert.That(info.scriptingBackend, Is.EqualTo("IL2CPP")); Assert.That(info.stripping, Is.EqualTo("High"));
                }
                Assert.That(calls, Is.EqualTo(1));
                Assert.That(PlayerSettings.GetScriptingBackend(target), Is.EqualTo(ScriptingImplementation.Mono2x));
                Assert.That(PlayerSettings.GetManagedStrippingLevel(target), Is.EqualTo(ManagedStrippingLevel.Low));
                Assert.That(PlayerSettings.GetIl2CppCompilerConfiguration(target), Is.EqualTo(Il2CppCompilerConfiguration.Debug));
                var receipt = Newtonsoft.Json.Linq.JObject.Parse(File.ReadAllText(Path.Combine(directory, "build.json")));
                Assert.That((bool)receipt["success"], Is.False, "An injected build is never a real Player success.");
                Assert.That((string)receipt["build_options"], Is.EqualTo("None"));
                Assert.That((string)receipt["il2cpp_compiler"], Is.EqualTo("Release"));
            }
            finally
            {
                PlayerSettings.SetScriptingBackend(target, backend);
                PlayerSettings.SetManagedStrippingLevel(target, stripping);
                PlayerSettings.SetIl2CppCompilerConfiguration(target, compiler);
            }
        }
        [TestCase("commit")] [TestCase("extension")]
        public void InvalidBuildInputsCannotGenerateAPlayer(string failure)
            => Assert.Throws<ArgumentException>(() => ValidationPlayerBuild.WindowsOptions(Path.Combine(directory, failure == "extension" ? "Validation.apk" : "Validation.exe"), failure == "commit" ? "main" : new string('a', 40)));
        [Test] public void ExistingBuildFilesCannotBeOverwritten()
        {
            File.WriteAllText(Path.Combine(directory, "existing.txt"), "keep");
            Assert.Throws<IOException>(() => ValidationPlayerBuild.WindowsOptions(Path.Combine(directory, "Validation.exe"), new string('a', 40)));
            Assert.That(File.ReadAllText(Path.Combine(directory, "existing.txt")), Is.EqualTo("keep"));
        }
        [Test] public void FailedBuildRestoresTheSceneAndRemovesTemporaryMarker()
        {
            var before = EditorSceneManager.GetSceneManagerSetup().Select(scene => scene.path).ToArray();
            var error = Assert.Throws<InvalidOperationException>(() => ValidationPlayerBuild.BuildWindows(Path.Combine(directory, "Validation.exe"), new string('a', 40), _ => throw new InvalidOperationException("injected build failure")));
            Assert.That(error.Message, Is.EqualTo("injected build failure"));
            Assert.That(EditorSceneManager.GetSceneManagerSetup().Select(scene => scene.path), Is.EqualTo(before));
            Assert.That(File.Exists(ValidationPlayerBuild.MarkerPath), Is.False); Assert.That(File.Exists(ValidationPlayerBuild.ScenePath), Is.False);
        }
    }
}
