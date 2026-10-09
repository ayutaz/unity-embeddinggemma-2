using System;
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
        public string Status { get; private set; } = "モデルと文書を準備してください。";
        public string Error { get; private set; } = "";
        public IReadOnlyList<SearchHit> Results => results;

        TextSearchSession session;
        SearchHit[] results = Array.Empty<SearchHit>();
        Vector2 scroll;
        Font font;

        void OnEnable()
        {
            var directory = Path.Combine(Application.streamingAssetsPath, "EmbeddingGemmaTextSearch");
            if (string.IsNullOrWhiteSpace(ModelPath)) ModelPath = Path.Combine(directory, "model-fp32.sentis");
            if (string.IsNullOrWhiteSpace(TokenizerPath)) TokenizerPath = Path.Combine(directory, "tokenizer.json");
        }

        public bool PrepareDocuments(Func<ITextEmbedder> providerFactory = null)
        {
            ReleaseDocuments();
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
                    ITextEmbedder provider;
                    if (providerFactory != null) provider = providerFactory();
                    else
                    {
                        if (!File.Exists(ModelPath)) throw new FileNotFoundException("準備した.sentisモデルのパスを指定してください。", ModelPath);
                        if (!File.Exists(TokenizerPath)) throw new FileNotFoundException("tokenizer.jsonのパスを指定してください。", TokenizerPath);
                        provider = new TextEmbedder(TextModelFile.Load(ModelPath), File.ReadAllText(TokenizerPath), Backend);
                    }
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
                ReleaseDocuments();
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
            catch (Exception exception) { Error = "検索に失敗: " + exception.Message; return false; }
        }

        public void ReleaseDocuments()
        {
            session?.Dispose(); session = null;
            results = Array.Empty<SearchHit>();
            Error = "";
            Status = "モデルと文書を準備してください。";
        }

        void OnDisable() => ReleaseDocuments();
        void OnDestroy()
        {
            ReleaseDocuments();
            if (font != null) Destroy(font);
        }

        void OnGUI()
        {
            if (font == null) font = Font.CreateDynamicFontFromOSFont(new[] { "Yu Gothic UI", "Noto Sans CJK JP", "Hiragino Sans", "Arial" }, 16);
            var oldFont = GUI.skin.font;
            GUI.skin.font = font;
            try
            {
                GUILayout.BeginArea(new Rect(16, 16, Math.Max(320, Screen.width - 32), Math.Max(240, Screen.height - 32)), GUI.skin.box);
                scroll = GUILayout.BeginScrollView(scroll);
                GUILayout.Label("EmbeddingGemma 2 — テキスト検索", new GUIStyle(GUI.skin.label) { fontSize = 22 });
                GUILayout.Label("日本語 / 英語の固定6文書をcosineで比較します。同点は文書ID順です。");
                GUILayout.Label("モデルは別途準備してください。自動ダウンロードは行いません。");
                GUI.enabled = !IsReady;
                GUILayout.Label(".sentisモデル"); ModelPath = GUILayout.TextField(ModelPath ?? "");
                GUILayout.Label("tokenizer.json"); TokenizerPath = GUILayout.TextField(TokenizerPath ?? "");
                Backend = GUILayout.Toolbar(Backend == BackendType.CPU ? 0 : 1, new[] { "CPU", "GPUCompute" }) == 0 ? BackendType.CPU : BackendType.GPUCompute;
                GUI.enabled = true;
                GUILayout.BeginHorizontal();
                if (GUILayout.Button("モデルと文書を準備")) PrepareDocuments();
                if (GUILayout.Button("解放 / モデルを変更")) ReleaseDocuments();
                GUILayout.EndHorizontal();
                GUILayout.Space(12);
                GUILayout.Label("検索文"); Query = GUILayout.TextField(Query ?? "");
                if (GUILayout.Button("検索")) Search(Query);
                GUILayout.Label(Status);
                if (!string.IsNullOrEmpty(Error)) GUILayout.Label(Error, new GUIStyle(GUI.skin.label) { normal = { textColor = Color.red }, wordWrap = true });
                for (var i = 0; i < results.Length; i++)
                {
                    var hit = results[i];
                    GUILayout.Space(8);
                    GUILayout.Label($"{i + 1}. {hit.Document.Title ?? hit.Document.Id}   cosine {hit.Score:F4}   [{hit.Document.Id}]");
                    GUILayout.Label(hit.Document.Text, new GUIStyle(GUI.skin.label) { wordWrap = true });
                }
                GUILayout.EndScrollView(); GUILayout.EndArea();
            }
            finally { GUI.enabled = true; GUI.skin.font = oldFont; }
        }
    }
}
