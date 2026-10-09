using System;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using Newtonsoft.Json.Linq;
using Unity.InferenceEngine;
using UnityEditor;
using UnityEngine;

namespace EmbeddingGemma.Samples
{
    /// <summary>Explicit Editor conversion of an audited CI export; models are never downloaded here.</summary>
    public static class TextSearchModelPreparation
    {
        [Serializable] public sealed class PreparedFile { public string name, sha256; public long bytes; }
        [Serializable] public sealed class Report
        {
            public bool success;
            public string error, sourceCommit, searchSourceCommit, unityVersion, completedUtc;
            public PreparedFile[] files;
        }

        public static Report Prepare(Model model, string tokenizerPath, string destination, string sourceCommit = "local", string searchSourceCommit = null)
        {
            destination = Path.GetFullPath(destination);
            Directory.CreateDirectory(destination);
            var report = new Report { success = false, sourceCommit = sourceCommit, searchSourceCommit = searchSourceCommit ?? sourceCommit, unityVersion = Application.unityVersion };
            var reportPath = Path.Combine(destination, "preparation.json");
            File.WriteAllText(reportPath, JsonUtility.ToJson(report, true));
            try
            {
                if (model == null) throw new ArgumentNullException(nameof(model));
                if (!File.Exists(tokenizerPath)) throw new FileNotFoundException("tokenizer.json is required.", tokenizerPath);
                var tokenizer = File.ReadAllText(tokenizerPath);
                try { _ = new TextTokenizer(tokenizer); }
                catch (Exception exception) { throw new ArgumentException("Invalid tokenizer.json", nameof(tokenizerPath), exception); }
                TextModelFile.Save(model, Path.Combine(destination, "model-fp32.sentis"));
                TextModelFile.Save(model, Path.Combine(destination, "model-float16.sentis"), float16: true);
                File.WriteAllText(Path.Combine(destination, "tokenizer.json"), tokenizer);
                report.files = new[] { "model-fp32.sentis", "model-float16.sentis", "tokenizer.json" }
                    .Select(name => new PreparedFile { name = name, sha256 = Digest(Path.Combine(destination, name)), bytes = new FileInfo(Path.Combine(destination, name)).Length }).ToArray();
                report.success = true;
                return report;
            }
            catch (Exception exception) { report.error = exception.Message; throw; }
            finally
            {
                report.completedUtc = DateTime.UtcNow.ToString("O");
                File.WriteAllText(reportPath, JsonUtility.ToJson(report, true));
            }
        }

        static string Digest(string path)
        {
            using var stream = File.OpenRead(path);
            using var hash = SHA256.Create();
            return BitConverter.ToString(hash.ComputeHash(stream)).Replace("-", "").ToLowerInvariant();
        }

        [MenuItem("Tools/EmbeddingGemma/Prepare Text Search Models")]
        public static void PrepareStagedModels()
        {
            var project = Path.GetFullPath(Path.Combine(Application.dataPath, ".."));
            var source = Path.Combine(project, "artifacts/m1");
            var audit = JObject.Parse(File.ReadAllText(Path.Combine(project, "artifacts/m1-stage.json")));
            if ((bool?)audit["success"] != true || (bool?)audit["search_reference_staged"] != true)
                throw new InvalidOperationException("Run stage --search against a successful CI artifact first.");
            if ((string)audit["model_revision"] != "914f7f89142e33e77833254d9c9b90c3cef7303b")
                throw new InvalidOperationException("The staged model revision is not the pinned revision.");
            const string assetPath = "Assets/M1Generated/model.pt2";
            foreach (var file in new[] { "model.pt2", "tokenizer.json", "search-reference.json" })
            {
                var path = file == "model.pt2" ? Path.Combine(project, assetPath) : Path.Combine(source, file);
                if (Digest(path) != (string)audit["sha256"]?[file])
                    throw new InvalidOperationException("Staged file hash changed: " + file);
            }
            AssetDatabase.ImportAsset(assetPath, ImportAssetOptions.ForceSynchronousImport);
            var asset = AssetDatabase.LoadAssetAtPath<ModelAsset>(assetPath);
            if (asset == null) throw new InvalidOperationException("The .pt2 model did not import.");
            var destination = Path.Combine(Application.streamingAssetsPath, "EmbeddingGemmaTextSearch");
            var report = Prepare(ModelLoader.Load(asset), Path.Combine(source, "tokenizer.json"), destination,
                                 (string)audit["source_commit"], (string)audit["search_source_commit"]);
            AssetDatabase.Refresh();
            Debug.Log("Text Search models prepared: " + destination + "\n" + JsonUtility.ToJson(report, true));
        }
    }
}
