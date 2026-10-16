using System;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using Newtonsoft.Json.Linq;
using UnityEditor;
using UnityEditor.Build;
using UnityEditor.Build.Reporting;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.SceneManagement;

namespace EmbeddingGemma.Validation.Editor
{
    public static class GpuConstantProbeBuild
    {
        public const string MarkerPath = "Assets/Validation/Generated/Resources/GpuConstantProbeBuild.json";
        public const string ScenePath = "Assets/Validation/Generated/GpuConstantProbe.unity";

        public static void Build(string output, string sourceCommit, Action<BuildPlayerOptions> execute = null)
        {
            if (sourceCommit == null || !Regex.IsMatch(sourceCommit, "\\A[a-f0-9]{40}\\z") ||
                string.IsNullOrWhiteSpace(output) || !output.EndsWith(".exe", StringComparison.OrdinalIgnoreCase))
                throw new ArgumentException("A fixed source commit and .exe output are required.");
            if (EditorApplication.isPlaying || EditorUserBuildSettings.activeBuildTarget != BuildTarget.StandaloneWindows64)
                throw new InvalidOperationException("Stop Play Mode and select Windows64 before building.");
            for (var index = 0; index < SceneManager.sceneCount; index++)
                if (SceneManager.GetSceneAt(index).isDirty) throw new InvalidOperationException("Save scene edits before building.");
            if (File.Exists(MarkerPath) || File.Exists(ScenePath) ||
                File.Exists("Assets/Validation/Generated/Resources/EmbeddingGemmaValidationBuild.json"))
                throw new IOException("Temporary validation assets already exist; inspect them before building.");
            output = Path.GetFullPath(output);
            var directory = Path.GetDirectoryName(output);
            if (Directory.Exists(directory) && Directory.EnumerateFileSystemEntries(directory).Any())
                throw new IOException("Build output directory must be empty.");
            var target = NamedBuildTarget.Standalone;
            var backend = PlayerSettings.GetScriptingBackend(target);
            var stripping = PlayerSettings.GetManagedStrippingLevel(target);
            var compiler = PlayerSettings.GetIl2CppCompilerConfiguration(target);
            var automatic = PlayerSettings.GetUseDefaultGraphicsAPIs(BuildTarget.StandaloneWindows64);
            var graphics = PlayerSettings.GetGraphicsAPIs(BuildTarget.StandaloneWindows64);
            var scenes = EditorSceneManager.GetSceneManagerSetup();
            Directory.CreateDirectory(directory);
            var receipt = new JObject
            {
                ["success"] = false, ["injected_build"] = execute != null, ["source_commit"] = sourceCommit,
                ["unity"] = Application.unityVersion, ["sentis"] = "2.6.1", ["started_utc"] = DateTime.UtcNow.ToString("O"),
                ["source_sha256"] = SourceHashes()
            };
            void Save() => File.WriteAllText(Path.Combine(directory, "build.json"), receipt.ToString(), new UTF8Encoding(false));
            Save();
            try
            {
                PlayerSettings.SetScriptingBackend(target, ScriptingImplementation.IL2CPP);
                PlayerSettings.SetManagedStrippingLevel(target, ManagedStrippingLevel.High);
                PlayerSettings.SetIl2CppCompilerConfiguration(target, Il2CppCompilerConfiguration.Release);
                PlayerSettings.SetUseDefaultGraphicsAPIs(BuildTarget.StandaloneWindows64, false);
                PlayerSettings.SetGraphicsAPIs(BuildTarget.StandaloneWindows64, new[] { GraphicsDeviceType.Direct3D12 });
                receipt["scripting_backend"] = "IL2CPP"; receipt["stripping"] = "High";
                receipt["compiler"] = "Release"; receipt["graphics_api"] = "Direct3D12";
                Directory.CreateDirectory(Path.GetDirectoryName(MarkerPath));
                File.WriteAllText(MarkerPath, receipt.ToString(), new UTF8Encoding(false));
                AssetDatabase.ImportAsset(MarkerPath, ImportAssetOptions.ForceSynchronousImport);
                var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
                if (!EditorSceneManager.SaveScene(scene, ScenePath)) throw new IOException("Could not save probe scene.");
                var options = new BuildPlayerOptions { scenes = new[] { ScenePath }, locationPathName = output,
                    target = BuildTarget.StandaloneWindows64, options = BuildOptions.None };
                if (execute != null) execute(options);
                else
                {
                    var report = BuildPipeline.BuildPlayer(options);
                    receipt["build_result"] = report.summary.result.ToString();
                    receipt["build_seconds"] = report.summary.totalTime.TotalSeconds;
                    receipt["build_bytes"] = report.summary.totalSize;
                    receipt["errors"] = report.summary.totalErrors; receipt["warnings"] = report.summary.totalWarnings;
                    if (report.summary.result != BuildResult.Succeeded) throw new InvalidOperationException("Probe build failed: " + report.summary.result);
                    receipt["success"] = true;
                }
            }
            catch (Exception error) { receipt["error"] = error.Message; throw; }
            finally
            {
                try
                {
                    if (scenes.Length == 0) EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
                    else EditorSceneManager.RestoreSceneManagerSetup(scenes);
                }
                catch (Exception error) { receipt["success"] = false; receipt["restore_error"] = error.Message; throw; }
                finally
                {
                    try { AssetDatabase.DeleteAsset(ScenePath); AssetDatabase.DeleteAsset(MarkerPath); }
                    finally
                    {
                        try
                        {
                            PlayerSettings.SetScriptingBackend(target, backend);
                            PlayerSettings.SetManagedStrippingLevel(target, stripping);
                            PlayerSettings.SetIl2CppCompilerConfiguration(target, compiler);
                            PlayerSettings.SetGraphicsAPIs(BuildTarget.StandaloneWindows64, graphics);
                            PlayerSettings.SetUseDefaultGraphicsAPIs(BuildTarget.StandaloneWindows64, automatic);
                        }
                        catch (Exception error) { receipt["success"] = false; receipt["settings_restore_error"] = error.Message; throw; }
                        finally { receipt["completed_utc"] = DateTime.UtcNow.ToString("O"); Save(); }
                    }
                }
            }
        }

        static JObject SourceHashes()
        {
            var package = UnityEditor.PackageManager.PackageInfo.FindForAssetPath("Packages/com.ayutaz.embeddinggemma/Runtime/GpuConstantWorker.cs");
            if (package == null) throw new InvalidOperationException("The Runtime package is unresolved.");
            var hashes = new JObject();
            void Add(string file, string key)
            {
                using var hash = SHA256.Create();
                hashes[key] = BitConverter.ToString(hash.ComputeHash(Encoding.UTF8.GetBytes(File.ReadAllText(file).Replace("\r\n", "\n"))))
                    .Replace("-", "").ToLowerInvariant();
            }
            Add(Path.Combine(package.resolvedPath, "Runtime/GpuConstantWorker.cs"), "runtime/GpuConstantWorker.cs");
            Add(Path.Combine(package.resolvedPath, "Runtime/link.xml"), "runtime/link.xml");
            foreach (var file in Directory.GetFiles("Assets/Validation/Runtime", "GpuConstantProbe*.cs").OrderBy(path => path, StringComparer.Ordinal))
                Add(file, "probe/" + Path.GetFileName(file));
            return hashes;
        }
    }
}
