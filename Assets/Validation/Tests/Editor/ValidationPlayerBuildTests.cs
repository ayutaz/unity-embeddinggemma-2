using System;
using System.IO;
using System.Linq;
using EmbeddingGemma.Validation.Editor;
using NUnit.Framework;
using UnityEditor;
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
