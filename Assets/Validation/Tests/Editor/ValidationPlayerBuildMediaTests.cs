using System;
using System.IO;
using EmbeddingGemma.Validation.Editor;
using NUnit.Framework;
using UnityEditor;
using UnityEditor.Build;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class ValidationPlayerBuildMediaTests
    {
        [TestCase(false)] [TestCase(true)] public void MediaPlayerUsesUniqueMarkerReleaseProfileAndRestoresSettings(bool fail)
        {
            var root = Path.Combine(Path.GetTempPath(), "eg-media-build-" + Guid.NewGuid().ToString("N"));
            var target = NamedBuildTarget.Standalone; var backend = PlayerSettings.GetScriptingBackend(target);
            var stripping = PlayerSettings.GetManagedStrippingLevel(target); var compiler = PlayerSettings.GetIl2CppCompilerConfiguration(target);
            var calls = 0;
            Action<BuildPlayerOptions> build = options =>
            {
                calls++; Assert.That(options.target, Is.EqualTo(BuildTarget.StandaloneWindows64));
                Assert.That(options.options, Is.EqualTo(BuildOptions.None));
                Assert.That(File.Exists(ValidationPlayerBuild.MediaMarkerPath), Is.True);
                Assert.That(File.Exists(ValidationPlayerBuild.MarkerPath), Is.False);
                Assert.That(File.Exists(ValidationPlayerBuild.SampleMarkerPath), Is.False);
                Assert.That(PlayerSettings.GetScriptingBackend(target), Is.EqualTo(ScriptingImplementation.IL2CPP));
                Assert.That(PlayerSettings.GetManagedStrippingLevel(target), Is.EqualTo(ManagedStrippingLevel.High));
                Assert.That(PlayerSettings.GetIl2CppCompilerConfiguration(target), Is.EqualTo(Il2CppCompilerConfiguration.Release));
                if (fail) throw new InvalidOperationException("Synthetic failure.");
            };
            try
            {
                if (fail) Assert.Throws<InvalidOperationException>(() => ValidationPlayerBuild.BuildWindowsMedia(Path.Combine(root, "Media.exe"), new string('a', 40), build));
                else ValidationPlayerBuild.BuildWindowsMedia(Path.Combine(root, "Media.exe"), new string('a', 40), build);
                Assert.That(calls, Is.EqualTo(1));
                Assert.That(File.Exists(ValidationPlayerBuild.MediaMarkerPath), Is.False);
                Assert.That(File.Exists(ValidationPlayerBuild.ScenePath), Is.False);
                Assert.That(PlayerSettings.GetScriptingBackend(target), Is.EqualTo(backend));
                Assert.That(PlayerSettings.GetManagedStrippingLevel(target), Is.EqualTo(stripping));
                Assert.That(PlayerSettings.GetIl2CppCompilerConfiguration(target), Is.EqualTo(compiler));
            }
            finally { if (Directory.Exists(root)) Directory.Delete(root, true); }
        }
    }
}
