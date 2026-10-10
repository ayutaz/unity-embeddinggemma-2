using System;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using UnityEditor;
using UnityEditor.Build;
using UnityEditor.Build.Reporting;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace EmbeddingGemma.Validation.Editor
{
    public static partial class ValidationPlayerBuild
    {
        public const string ScenePath = "Assets/Validation/Generated/Validation.unity";
        public const string MarkerPath = "Assets/Validation/Generated/Resources/EmbeddingGemmaValidationBuild.json";
        public static BuildPlayerOptions WindowsOptions(string output, string codeCommit, bool development = true)
        {
            if (codeCommit == null || !Regex.IsMatch(codeCommit, "\\A[a-f0-9]{40}\\z") || string.IsNullOrWhiteSpace(output) ||
                !output.EndsWith(".exe", StringComparison.OrdinalIgnoreCase)) throw new ArgumentException("A fixed source commit and .exe output are required.");
            output = Path.GetFullPath(output);
            var directory = Path.GetDirectoryName(output);
            if (Directory.Exists(directory) && Directory.EnumerateFileSystemEntries(directory).Any()) throw new IOException("Build output directory must be empty.");
            return new BuildPlayerOptions { scenes = new[] { ScenePath }, locationPathName = output,
                target = BuildTarget.StandaloneWindows64, options = development ? BuildOptions.Development : BuildOptions.None };
        }
        public static PlayerBuildInfo BuildWindows(string output, string codeCommit, Action<BuildPlayerOptions> execute = null, bool releaseIl2Cpp = false)
            => BuildTargetPlayer(WindowsOptions(output, codeCommit, development: !releaseIl2Cpp), codeCommit, NamedBuildTarget.Standalone, execute, releaseIl2Cpp);

        static void EnsureBuildState(BuildTarget target)
        {
            if (EditorApplication.isPlaying || EditorUserBuildSettings.activeBuildTarget != target)
                throw new InvalidOperationException("Stop Play Mode and select " + target + " before building.");
            for (var i = 0; i < SceneManager.sceneCount; i++) if (SceneManager.GetSceneAt(i).isDirty) throw new InvalidOperationException("Save scene edits before building.");
            if (File.Exists(ScenePath) || File.Exists(MarkerPath) || File.Exists(SampleMarkerPath)) throw new IOException("Temporary validation assets already exist; inspect them before building.");
        }

        static PlayerBuildInfo BuildTargetPlayer(BuildPlayerOptions options, string codeCommit, NamedBuildTarget target,
            Action<BuildPlayerOptions> execute, bool releaseIl2Cpp = false, Action configure = null, Action restore = null, Func<JObject> details = null, bool sampleScene = false)
        {
            EnsureBuildState(options.target);
            var previous = EditorSceneManager.GetSceneManagerSetup();
            var previousBackend = PlayerSettings.GetScriptingBackend(target);
            var previousStripping = PlayerSettings.GetManagedStrippingLevel(target);
            var previousCompiler = PlayerSettings.GetIl2CppCompilerConfiguration(target);
            var directory = Path.GetDirectoryName(options.locationPathName);
            var markerPath = sampleScene ? SampleMarkerPath : MarkerPath;
            Directory.CreateDirectory(directory);
            var receipt = new JObject { ["success"] = false, ["injected_build"] = execute != null, ["source_commit"] = codeCommit, ["started_utc"] = DateTime.UtcNow.ToString("O") };
            var receiptPath = Path.Combine(directory, "build.json");
            void Save() => File.WriteAllText(receiptPath, receipt.ToString(), new UTF8Encoding(false));
            Save();
            try
            {
                configure?.Invoke();
                if (releaseIl2Cpp)
                {
                    PlayerSettings.SetScriptingBackend(target, ScriptingImplementation.IL2CPP);
                    PlayerSettings.SetManagedStrippingLevel(target, ManagedStrippingLevel.High);
                    PlayerSettings.SetIl2CppCompilerConfiguration(target, Il2CppCompilerConfiguration.Release);
                }
                receipt["build_options"] = options.options.ToString();
                receipt["il2cpp_compiler"] = PlayerSettings.GetIl2CppCompilerConfiguration(target).ToString();
                if (details != null) receipt["target_settings"] = details();
                var info = new PlayerBuildInfo { codeCommit = codeCommit, unityVersion = Application.unityVersion, sentisVersion = "2.6.1", target = options.target.ToString(),
                    scriptingBackend = PlayerSettings.GetScriptingBackend(target).ToString(),
                    stripping = PlayerSettings.GetManagedStrippingLevel(target).ToString(), sourceSha256 = SourceHashes(sampleScene) };
                info.Validate(); receipt["build_info"] = JObject.FromObject(info); Save();
                Directory.CreateDirectory(Path.GetDirectoryName(markerPath));
                File.WriteAllText(markerPath, JsonConvert.SerializeObject(info, Formatting.Indented), new UTF8Encoding(false));
                AssetDatabase.ImportAsset(markerPath, ImportAssetOptions.ForceSynchronousImport);
                if (!sampleScene)
                {
                    var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
                    if (!EditorSceneManager.SaveScene(scene, ScenePath)) throw new IOException("Could not save validation scene.");
                }
                if (execute != null) execute(options);
                else
                {
                    var report = BuildPipeline.BuildPlayer(options);
                    receipt["build_result"] = report.summary.result.ToString();
                    receipt["build_seconds"] = report.summary.totalTime.TotalSeconds; receipt["build_bytes"] = report.summary.totalSize;
                    receipt["errors"] = report.summary.totalErrors; receipt["warnings"] = report.summary.totalWarnings;
                    if (report.summary.result != BuildResult.Succeeded) throw new InvalidOperationException("Player build failed: " + report.summary.result);
                }
                receipt["success"] = execute == null; return info;
            }
            catch (Exception exception) { receipt["error"] = exception.Message; throw; }
            finally
            {
                try
                {
                    // Test Runner can start with an unsaved empty scene and no saved setup.
                    if (previous.Length == 0) EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
                    else EditorSceneManager.RestoreSceneManagerSetup(previous);
                }
                catch (Exception exception) { receipt["success"] = false; receipt["restore_error"] = exception.Message; throw; }
                finally
                {
                    try { if (!sampleScene) AssetDatabase.DeleteAsset(ScenePath); AssetDatabase.DeleteAsset(markerPath); }
                    finally
                    {
                        try
                        {
                            restore?.Invoke();
                            if (releaseIl2Cpp)
                            {
                                PlayerSettings.SetScriptingBackend(target, previousBackend);
                                PlayerSettings.SetManagedStrippingLevel(target, previousStripping);
                                PlayerSettings.SetIl2CppCompilerConfiguration(target, previousCompiler);
                            }
                        }
                        catch (Exception exception) { receipt["success"] = false; receipt["settings_restore_error"] = exception.Message; throw; }
                        finally { receipt["completed_utc"] = DateTime.UtcNow.ToString("O"); Save(); }
                    }
                }
            }
        }

        static JObject SourceHashes(bool includeSample = false)
        {
            var hashes = new JObject();
            void AddFile(string file, string key)
            {
                var text = File.ReadAllText(file).Replace("\r\n", "\n"); using var hash = SHA256.Create();
                hashes[key] = BitConverter.ToString(hash.ComputeHash(Encoding.UTF8.GetBytes(text))).Replace("-", "").ToLowerInvariant();
            }
            void Add(string root, string prefix)
            {
                foreach (var file in Directory.GetFiles(root, "*.cs", SearchOption.AllDirectories).OrderBy(path => path, StringComparer.Ordinal))
                {
                    AddFile(file, prefix + Path.GetRelativePath(root, file).Replace('\\', '/'));
                }
            }
            Add(Path.GetFullPath("Assets/Validation/Runtime"), "Assets/Validation/Runtime/");
            AddFile("Assets/Validation/Runtime/link.xml", "Assets/Validation/Runtime/link.xml");
            var package = UnityEditor.PackageManager.PackageInfo.FindForAssetPath("Packages/com.ayutaz.embeddinggemma/Runtime/TextEmbedder.cs");
            if (package == null) throw new InvalidOperationException("The Runtime package is unresolved.");
            Add(Path.Combine(package.resolvedPath, "Runtime"), "Packages/com.ayutaz.embeddinggemma/Runtime/");
            if (includeSample)
            {
                Add(Path.GetFullPath("Assets/EmbeddingGemmaTextSearch"), "Assets/EmbeddingGemmaTextSearch/");
                AddFile(SampleScenePath, SampleScenePath);
                const string corpus = "Assets/EmbeddingGemmaTextSearch/Resources/EmbeddingGemmaTextSearch/corpus.json";
                AddFile(corpus, corpus);
                const string assembly = "Assets/EmbeddingGemmaTextSearch/EmbeddingGemma.TextSearch.Sample.asmdef";
                AddFile(assembly, assembly);
            }
            return hashes;
        }
    }
}
