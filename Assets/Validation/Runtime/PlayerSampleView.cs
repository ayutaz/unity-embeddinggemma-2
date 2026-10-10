using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using Newtonsoft.Json.Linq;
using Unity.InferenceEngine;
using UnityEngine;

namespace EmbeddingGemma.Validation
{
    public sealed class PlayerSampleView : IPlayerSampleView
    {
        // The validation project need not import the optional sample to compile.
        public const string SampleTypeName = "EmbeddingGemma.Samples.TextSearchSample,EmbeddingGemma.TextSearch.Sample";
        const BindingFlags Private = BindingFlags.Instance | BindingFlags.NonPublic;
        readonly Component sample;
        readonly Type type;
        Font ownedFont;
        bool destroyRequested;
        public static PlayerSampleView FromScene()
        {
            var sampleType = Type.GetType(SampleTypeName) ?? throw new InvalidOperationException("The imported sample assembly is absent.");
            return new PlayerSampleView(UnityEngine.Object.FindFirstObjectByType(sampleType) as Component);
        }
        public PlayerSampleView(Component sample)
        {
            this.sample = sample != null ? sample : throw new ArgumentNullException(nameof(sample));
            type = sample.GetType();
            if (type.FullName != "EmbeddingGemma.Samples.TextSearchSample") throw new ArgumentException("The imported TextSearch sample is required.");
        }
        object Property(string name) => type.GetProperty(name).GetValue(sample);
        object Field(string name) => type.GetField(name, Private).GetValue(sample);
        object Call(string name, params object[] args)
        {
            try { return type.GetMethod(name).Invoke(sample, args); }
            catch (TargetInvocationException exception) { throw exception.InnerException ?? exception; }
        }
        public bool Ready => (bool)Property("IsReady");
        public string Error => (string)Property("Error");
        public bool FontAlive => destroyRequested ? ownedFont != null : (Font)Field("font") != null;
        public int FontInstanceId => FontAlive ? ((Font)Field("font")).GetInstanceID() : 0;
        public JObject Describe()
        {
            var cameras = new JArray();
            foreach (var camera in UnityEngine.Object.FindObjectsByType<Camera>(FindObjectsInactive.Include, FindObjectsSortMode.None))
                cameras.Add(new JObject { ["name"] = camera.name, ["enabled"] = camera.enabled, ["active"] = camera.gameObject.activeInHierarchy,
                    ["target_display"] = camera.targetDisplay, ["pixel_width"] = camera.pixelWidth, ["pixel_height"] = camera.pixelHeight,
                    ["rendering_path"] = camera.renderingPath.ToString(), ["actual_rendering_path"] = camera.actualRenderingPath.ToString(),
                    ["rect"] = camera.rect.ToString(), ["background"] = camera.backgroundColor.ToString(), ["target_texture"] = camera.targetTexture != null });
            return new JObject { ["active"] = sample.gameObject.activeInHierarchy,
                ["enabled"] = ((MonoBehaviour)sample).enabled, ["use_guilayout"] = ((MonoBehaviour)sample).useGUILayout,
                ["ready"] = Ready, ["font_alive"] = FontAlive, ["model_path"] = (string)type.GetField("ModelPath").GetValue(sample),
                ["frame"] = Time.frameCount, ["scene"] = sample.gameObject.scene.path, ["cameras"] = cameras,
                ["active_camera_count"] = Camera.allCamerasCount };
        }
        public IReadOnlyList<SearchHit> Results => (IReadOnlyList<SearchHit>)Property("Results");
        public string ActualBackend
        {
            get
            {
                var session = Field("session") as TextSearchSession;
                var embedder = session == null ? null : typeof(TextSearchSession).GetField("embedder", Private).GetValue(session) as TextEmbedder;
                var worker = embedder == null ? null : typeof(TextEmbedder).GetField("worker", Private).GetValue(embedder) as Worker;
                return worker?.backendType.ToString();
            }
        }
        public bool Search(string query) => (bool)Call("Search", query);
        public void Release() => Call("ReleaseDocuments");
        public IEnumerator Prepare(string bundle, string cacheRoot)
        {
            var source = new Uri(Path.GetFullPath(bundle) + Path.DirectorySeparatorChar).AbsoluteUri;
            type.GetField("ModelPath").SetValue(sample, source + "model-float16.sentis");
            type.GetField("TokenizerPath").SetValue(sample, source + "tokenizer.json");
            type.GetField("Backend").SetValue(sample, BackendType.GPUCompute);
            return (IEnumerator)Call("PrepareDocumentsFromStreamingAssets", cacheRoot, null, null);
        }
        public PlayerRanking[] SearchRanking(string query)
        {
            if (!Search(query)) throw new InvalidOperationException(Error);
            return Results.Select(hit => new PlayerRanking { document_id = hit.Document.Id, score = hit.Score }).ToArray();
        }
        public bool RejectBlank() => !Search("") && Results.Count == 0 && !string.IsNullOrEmpty(Error);
        public bool RejectMissingModel()
        {
            var field = type.GetField("ModelPath"); var original = field.GetValue(sample);
            try
            {
                field.SetValue(sample, Path.Combine(Path.GetTempPath(), "embeddinggemma-missing-" + Guid.NewGuid().ToString("N") + ".sentis"));
                return !(bool)Call("PrepareDocuments", new object[] { null }) && !Ready && ActualBackend == null && Results.Count == 0 && !string.IsNullOrEmpty(Error);
            }
            finally { field.SetValue(sample, original); }
        }
        public JObject CacheSnapshot
        {
            get
            {
                var stage = Field("modelStage"); if (stage == null) return null;
                var snapshot = new JObject();
                foreach (var name in new[] { "Success", "CacheReused", "TransferredFiles", "Milliseconds", "AuditMilliseconds", "HashBackends", "CacheRejectedError" })
                {
                    var value = stage.GetType().GetField(name).GetValue(stage);
                    snapshot[name] = value == null ? JValue.CreateNull() : JToken.FromObject(value);
                }
                return snapshot;
            }
        }
        public bool ReleaseAndCheckNativeOwner()
        {
            var session = Field("session") as TextSearchSession;
            var provider = session == null ? null : typeof(TextSearchSession).GetField("embedder", Private).GetValue(session);
            var stage = Field("modelStage");
            Release();
            // A synthetic provider cannot establish native Worker disposal.
            return (provider == null || provider is TextEmbedder && typeof(TextEmbedder).GetField("worker", Private).GetValue(provider) == null) &&
                (stage == null || stage.GetType().GetField("Lease", Private).GetValue(stage) == null) &&
                Field("session") == null && Field("modelStage") == null && !Ready && ActualBackend == null && Results.Count == 0;
        }
        public void DestroyView()
        {
            ownedFont = (Font)Field("font"); destroyRequested = true;
            UnityEngine.Object.Destroy(sample.gameObject);
        }
    }
    public static class PlayerSampleQueryValidation
    {
        public static void Check(PlayerRow expected, IReadOnlyList<PlayerRanking> actual)
        {
            var ranks = expected?.ranking;
            if (ranks == null || ranks.Length != 6 || ranks.Any(rank => rank == null || string.IsNullOrWhiteSpace(rank.document_id) || !Finite(rank.score)) ||
                ranks.Select(rank => rank.document_id).Distinct(StringComparer.Ordinal).Count() != 6 || actual == null || actual.Count != 6)
                throw new ArgumentException("A complete unique six-document reference and result are required.");
            for (var i = 0; i < ranks.Length; i++)
                if (actual[i] == null || actual[i].document_id != ranks[i].document_id || !Finite(actual[i].score) || Math.Abs(actual[i].score - ranks[i].score) > 0.02)
                    throw new ArgumentException("Sample rank or score differs at position " + i);
        }
        static bool Finite(double value) => !double.IsNaN(value) && !double.IsInfinity(value);
    }
}
