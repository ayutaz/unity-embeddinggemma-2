using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Unity.InferenceEngine;
using UnityEngine;

namespace EmbeddingGemma.Samples
{
    /// <summary>Import the sample, prepare models explicitly, then open TextSearch.unity.</summary>
    public sealed class TextSearchSample : MonoBehaviour
    {
        [Serializable] sealed class CorpusFile { public DocumentRow[] documents; }
        [Serializable] sealed class DocumentRow { public string id, text, title; }

        public TextAsset Corpus;
        public string ModelPath;
        public string TokenizerPath;
        public BackendType Backend = BackendType.GPUCompute;
        public string Query = "猫を健康に育てるには";
        public bool IsReady => session?.IsReady == true;
        public bool IsPreparing { get; private set; }
        public string Status { get; private set; } = "モデルと文書を準備してください。";
        public string Error { get; private set; } = "";
        public IReadOnlyList<SearchHit> Results => results;

        TextSearchSession session;
        SearchHit[] results = Array.Empty<SearchHit>();
        Vector2 scroll;
        Font font;
        TextSearchGuiStyles styles;
        IEnumerator preparationOperation;
        Coroutine preparationCoroutine;
        TextSearchModelStage modelStage;
        int preparationVersion;

        void OnEnable()
        {
            var directory = TextSearchModelCache.Combine(Application.streamingAssetsPath, "EmbeddingGemmaTextSearch");
            if (string.IsNullOrWhiteSpace(ModelPath)) ModelPath = TextSearchModelCache.Combine(directory, "model-fp32.sentis");
            if (string.IsNullOrWhiteSpace(TokenizerPath)) TokenizerPath = TextSearchModelCache.Combine(directory, "tokenizer.json");
        }

        public IEnumerator PrepareDocumentsFromStreamingAssets(string cacheRoot = null, Func<string, string, IEnumerator> transfer = null,
            Func<string, string, ITextEmbedder> providerFactory = null)
        {
            ReleaseDocuments();
            var version = ++preparationVersion;
            var stage = new TextSearchModelStage(); modelStage = stage;
            var operation = TextSearchModelCache.Resolve(ModelPath, TokenizerPath,
                cacheRoot ?? Path.Combine(Application.persistentDataPath, "EmbeddingGemmaTextSearch", "ModelCache"), stage, transfer);
            preparationOperation = operation; IsPreparing = true; Status = "モデルを展開・確認しています。";
            try
            {
                while (version == preparationVersion && operation.MoveNext()) yield return operation.Current;
                if (version != preparationVersion) yield break;
                if (!stage.Success)
                {
                    Error = "準備に失敗: " + stage.Error; Status = "モデルを準備できませんでした。";
                    stage.Dispose(); modelStage = null; yield break;
                }
                if (!PrepareCore(() => providerFactory != null ? providerFactory(stage.ModelPath, stage.TokenizerPath)
                    : LoadProvider(stage.ModelPath, stage.TokenizerPath))) Status = "モデルを準備できませんでした。";
            }
            finally
            {
                (operation as IDisposable)?.Dispose();
                if (version == preparationVersion) { preparationOperation = null; preparationCoroutine = null; IsPreparing = false; }
            }
        }
        public void BeginPreparation()
        {
            if (IsPreparing) return;
            if (ModelPath?.Contains("://") == true || TokenizerPath?.Contains("://") == true)
                preparationCoroutine = StartCoroutine(PrepareDocumentsFromStreamingAssets());
            else PrepareDocuments();
        }

        public bool PrepareDocuments(Func<ITextEmbedder> providerFactory = null)
        {
            ReleaseDocuments();
            return PrepareCore(providerFactory ?? (() => LoadProvider(ModelPath, TokenizerPath)));
        }
        ITextEmbedder LoadProvider(string model, string tokenizer)
        {
            if (!File.Exists(model)) throw new FileNotFoundException("準備した.sentisモデルのパスを指定してください。", model);
            if (!File.Exists(tokenizer)) throw new FileNotFoundException("tokenizer.jsonのパスを指定してください。", tokenizer);
            return new TextEmbedder(TextModelFile.Load(model), File.ReadAllText(tokenizer), Backend);
        }
        bool PrepareCore(Func<ITextEmbedder> providerFactory)
        {
            try
            {
                if (Backend != BackendType.CPU && Backend != BackendType.GPUCompute)
                    throw new ArgumentException("CPUかGPUComputeを選択してください。");
                var asset = Corpus != null ? Corpus : Resources.Load<TextAsset>("EmbeddingGemmaTextSearch/corpus");
                if (asset == null) throw new InvalidOperationException("文書データがありません。TextSearchサンプルをインポートしてください。");
                var data = JsonUtility.FromJson<CorpusFile>(asset.text);
                if (data?.documents == null || data.documents.Length == 0)
                    throw new ArgumentException("文書データが空か不正です。");
                var documents = data.documents.Select(row => new SearchDocument(row.id, row.text, row.title)).ToArray();
                session = new TextSearchSession();
                session.Prepare(documents, () =>
                {
                    var provider = providerFactory();
                    if (provider == null) throw new InvalidOperationException("推論器を作成できませんでした。");
                    if (provider.Backend != Backend)
                    {
                        provider.Dispose();
                        throw new InvalidOperationException("指定backendと実際の推論器が一致しません。");
                    }
                    return provider;
                });
                Status = $"文書 {session.DocumentCount} 件を準備済み / {Backend}";
                return true;
            }
            catch (Exception exception)
            {
                DisposeWorkerAndCache();
                Error = "準備に失敗: " + exception.Message;
                return false;
            }
        }

        public bool Search(string query)
        {
            Query = query;
            results = Array.Empty<SearchHit>();
            Error = "";
            try
            {
                if (!IsReady) throw new InvalidOperationException("先にモデルと文書を準備してください。");
                results = session.Search(query);
                Status = $"検索結果 {results.Length} 件 / {Backend}";
                return true;
            }
            catch (Exception exception)
            {
                Status = "検索できませんでした。";
                Error = "検索に失敗: " + exception.Message;
                return false;
            }
        }

        public void ReleaseDocuments()
        {
            preparationVersion++;
            if (preparationCoroutine != null) { StopCoroutine(preparationCoroutine); preparationCoroutine = null; }
            (preparationOperation as IDisposable)?.Dispose(); preparationOperation = null; IsPreparing = false;
            DisposeWorkerAndCache();
            results = Array.Empty<SearchHit>();
            Error = "";
            Status = "モデルと文書を準備してください。";
        }
        void DisposeWorkerAndCache()
        {
            try { session?.Dispose(); }
            finally { session = null; modelStage?.Dispose(); modelStage = null; }
        }

        void OnDisable() => ReleaseDocuments();
        void OnDestroy()
        {
            ReleaseDocuments();
            styles?.Dispose(); styles = null;
            TextSearchGuiFont.Release(font); font = null;
        }

        void OnGUI()
        {
            if (font == null) font = TextSearchGuiFont.Acquire();
            styles ??= new TextSearchGuiStyles(GUI.skin, font);
            try
            {
                GUILayout.BeginArea(new Rect(16, 16, Math.Max(320, Screen.width - 32), Math.Max(240, Screen.height - 32)), GUI.skin.box);
                scroll = GUILayout.BeginScrollView(scroll);
                GUILayout.Label("EmbeddingGemma 2 — テキスト検索", styles.Heading);
                GUILayout.Label("日本語 / 英語の固定6文書をcosineで比較します。同点は文書ID順です。", styles.Label);
                GUILayout.Label("モデルは別途準備してください。自動ダウンロードは行いません。", styles.Label);
                GUI.enabled = !IsReady && !IsPreparing;
                GUILayout.Label(".sentisモデル", styles.Label); ModelPath = GUILayout.TextField(ModelPath ?? "", styles.TextField);
                GUILayout.Label("tokenizer.json", styles.Label); TokenizerPath = GUILayout.TextField(TokenizerPath ?? "", styles.TextField);
                Backend = GUILayout.Toolbar(Backend == BackendType.CPU ? 0 : 1, new[] { "CPU", "GPUCompute" }, styles.Button) == 0 ? BackendType.CPU : BackendType.GPUCompute;
                GUI.enabled = true;
                GUILayout.BeginHorizontal();
                GUI.enabled = !IsPreparing;
                if (GUILayout.Button("モデルと文書を準備", styles.Button)) BeginPreparation();
                GUI.enabled = true;
                if (GUILayout.Button("解放 / モデルを変更", styles.Button)) ReleaseDocuments();
                GUILayout.EndHorizontal();
                GUILayout.Space(12);
                GUILayout.Label("検索文", styles.Label); Query = GUILayout.TextField(Query ?? "", styles.TextField);
                GUI.enabled = IsReady;
                if (GUILayout.Button("検索", styles.Button)) Search(Query);
                GUI.enabled = true;
                GUILayout.Label(Status, styles.Label);
                if (!string.IsNullOrEmpty(Error)) GUILayout.Label(Error, styles.Error);
                for (var i = 0; i < results.Length; i++)
                {
                    var hit = results[i];
                    GUILayout.Space(8);
                    GUILayout.Label($"{i + 1}. {hit.Document.Title ?? hit.Document.Id}   cosine {hit.Score:F4}   [{hit.Document.Id}]", styles.Label);
                    GUILayout.Label(hit.Document.Text, styles.Label);
                }
                GUILayout.EndScrollView(); GUILayout.EndArea();
            }
            finally { GUI.enabled = true; }
        }
    }
}
