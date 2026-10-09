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
            public string error, sourceCommit, searchSourceCommit, unityVersion, completedUtc, modelSha256;
            public PreparedFile[] files;
        }

        public static Report Prepare(Model model, string tokenizerPath, string destination, string sourceCommit = "local", string searchSourceCommit = null)
            => PrepareCore(() => model, tokenizerPath, destination, sourceCommit, searchSourceCommit, null);

        static Report PrepareCore(Func<Model> loadModel, string tokenizerPath, string destination,
            string sourceCommit, string searchSourceCommit, string modelSha256)
        {
            destination = Path.GetFullPath(destination);
            Directory.CreateDirectory(destination);
            var report = new Report { success = false, sourceCommit = sourceCommit, searchSourceCommit = searchSourceCommit ?? sourceCommit,
                unityVersion = Application.unityVersion, modelSha256 = modelSha256 };
            var reportPath = Path.Combine(destination, "preparation.json");
            File.WriteAllText(reportPath, JsonUtility.ToJson(report, true));
            try
            {
                if (!File.Exists(tokenizerPath)) throw new FileNotFoundException("tokenizer.json is required.", tokenizerPath);
                var tokenizer = File.ReadAllText(tokenizerPath);
                try { _ = new TextTokenizer(tokenizer); }
                catch (Exception exception) { throw new ArgumentException("Invalid tokenizer.json", nameof(tokenizerPath), exception); }
                var model = loadModel();
                if (model == null) throw new ArgumentNullException(nameof(model));
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

        public static Report PrepareCached(Func<Model> loadModel, string tokenizerPath, string destination,
            string sourceCommit, string searchSourceCommit, string modelSha256)
        {
            // Hash the files, but do not deserialize gigabytes of weights on a cache hit.
            var cached = VerifiedCache(tokenizerPath, destination, sourceCommit, searchSourceCommit, modelSha256);
            return cached ?? PrepareCore(loadModel, tokenizerPath, destination, sourceCommit, searchSourceCommit, modelSha256);
        }

        static Report VerifiedCache(string tokenizerPath, string destination,
            string sourceCommit, string searchSourceCommit, string modelSha256)
        {
            try
            {
                var report = JsonUtility.FromJson<Report>(File.ReadAllText(Path.Combine(destination, "preparation.json")));
                if (report == null || !report.success || string.IsNullOrEmpty(modelSha256) || report.modelSha256 != modelSha256 ||
                    report.sourceCommit != sourceCommit || report.searchSourceCommit != searchSourceCommit ||
                    report.unityVersion != Application.unityVersion || report.files == null ||
                    !report.files.Select(file => file?.name).SequenceEqual(new[] { "model-fp32.sentis", "model-float16.sentis", "tokenizer.json" }))
                    return null;
                foreach (var file in report.files)
                {
                    var path = Path.Combine(destination, file.name);
                    if (!File.Exists(path) || new FileInfo(path).Length != file.bytes || Digest(path) != file.sha256) return null;
                }
                if (Digest(tokenizerPath) != report.files[2].sha256) return null;
                return report;
            }
            catch (IOException) { return null; }
            catch (ArgumentException) { return null; }
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
            var destination = Path.Combine(Application.streamingAssetsPath, "EmbeddingGemmaTextSearch");
            Model LoadStagedModel()
            {
                var asset = AssetDatabase.LoadAssetAtPath<ModelAsset>(assetPath);
                if (asset == null)
                {
                    AssetDatabase.ImportAsset(assetPath, ImportAssetOptions.ForceSynchronousImport);
                    asset = AssetDatabase.LoadAssetAtPath<ModelAsset>(assetPath);
                }
                if (asset == null) throw new InvalidOperationException("The .pt2 model did not import.");
                try { return ModelLoader.Load(asset); }
                finally { Resources.UnloadAsset(asset); }
            }
            var report = PrepareCached(LoadStagedModel, Path.Combine(source, "tokenizer.json"), destination,
                (string)audit["source_commit"], (string)audit["search_source_commit"], (string)audit["sha256"]?["model.pt2"]);
            // Runtime loads these files directly; it needs neither ModelAsset imports nor a full Refresh.
            Debug.Log("Text Search models verified: " + destination + "\n" + JsonUtility.ToJson(report, true));
        }
    }
}
