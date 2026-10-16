using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

namespace EmbeddingGemma.Samples
{
    /// <summary>Importable, main-thread GPU media registration and text search. Paths and labels are display metadata only.</summary>
    public sealed class MultimodalSearchSample : MonoBehaviour
    {
        public string SourceDirectory = "";
        public string CacheDirectory = "";
        public string ModuleName = "image-square";
        public string MediaPath = "";
        public string Query = "猫の画像";
        public bool IsBusy { get; private set; }
        public int DocumentCount => documents.Count;
        public string Status { get; private set; } = "モデルの準備先と入力ファイルを指定してください。";
        public string LastError { get; private set; }
        public int LastTransferredFiles { get; private set; }
        public int LastReusedObjects { get; private set; }
        public SearchHit[] Hits { get; private set; } = Array.Empty<SearchHit>();
        readonly List<SearchDocument> documents = new List<SearchDocument>();
        readonly List<float[]> vectors = new List<float[]>();
        GpuMediaBundleLease lease;
        GpuMediaSession session;
        CancellationTokenSource operationCancellation;
        Task activeOperation, closeTask;
        bool closing;
        Font font; MediaSearchGuiStyles styles; Vector2 scroll;
        static readonly string[] Modules = { "image-square", "image-landscape", "image-portrait", "audio-1119", "audio-839" };

        async void OnEnable()
        {
            if (closeTask != null) await closeTask;
            closing = false;
        }
        void OnDisable() { _ = CloseAsync(); }
        void OnDestroy() { styles?.Dispose(); styles = null; MediaSearchGuiFont.Release(font); font = null; }
        public void Cancel() => operationCancellation?.Cancel();

        public Task AddAsync(CancellationToken cancellationToken = default) => StartOperation(async token =>
        {
            var path = Path.GetFullPath(MediaPath); var module = ModuleName;
            if (!File.Exists(path)) throw new FileNotFoundException("入力ファイルが見つかりません。", path);
            float[] vector;
            if (module.StartsWith("image-", StringComparison.Ordinal))
            {
                var texture = new Texture2D(2, 2, TextureFormat.RGBA32, false, true);
                try
                {
                    if (!ImageConversion.LoadImage(texture, File.ReadAllBytes(path), true)) throw new InvalidDataException("PNG / JPEG画像を読み込めません。");
                    await SelectModule(module, token); Status = "画像をGPUで登録しています…";
                    vector = await session.EmbedImageAsync(texture, cancellationToken: token);
                }
                finally { if (Application.isPlaying) Destroy(texture); else DestroyImmediate(texture); }
            }
            else if (module.StartsWith("audio-", StringComparison.Ordinal))
            {
                var pcm = MediaSampleInput.ReadPcm(path);
                if (module != "audio-" + pcm.ValidFrameCount) throw new ArgumentException("音声の長さに対応する固定profileを選択してください。通常音声は1119、短い検証音声は839です。");
                await SelectModule(module, token); Status = "音声をGPUで登録しています…";
                vector = await session.EmbedAudioAsync(pcm, cancellationToken: token);
            }
            else throw new ArgumentException("画像または音声moduleを選択してください。");
            token.ThrowIfCancellationRequested();
            documents.Add(new SearchDocument("media-" + documents.Count, path, Path.GetFileName(path))); vectors.Add(vector);
            Hits = Array.Empty<SearchHit>(); Status = documents.Count + "件を登録しました。テキストで検索できます。";
        }, cancellationToken);

        public Task SearchAsync(CancellationToken cancellationToken = default) => StartOperation(async token =>
        {
            if (documents.Count == 0) throw new InvalidOperationException("画像または音声を先に登録してください。");
            if (string.IsNullOrWhiteSpace(Query)) throw new ArgumentException("検索文を入力してください。");
            var query = Query; await SelectModule("text", token); Status = "テキストをGPUで検索しています…";
            token.ThrowIfCancellationRequested();
            var index = new MultimodalSearchIndex(documents, vectors, session); var hits = index.Search(query, documents.Count);
            token.ThrowIfCancellationRequested(); Hits = hits; Status = hits.Length + "件の検索結果";
        }, cancellationToken);

        Task StartOperation(Func<CancellationToken, Task> operation, CancellationToken token)
        {
            if (IsBusy || closing) return Task.FromException(new InvalidOperationException("進行中の処理が完了するまでお待ちください。"));
            IsBusy = true; LastError = null; operationCancellation = CancellationTokenSource.CreateLinkedTokenSource(token);
            activeOperation = Execute(operation, operationCancellation); return activeOperation;
        }
        async Task Execute(Func<CancellationToken, Task> operation, CancellationTokenSource cancellation)
        {
            try { await Task.Yield(); cancellation.Token.ThrowIfCancellationRequested(); await operation(cancellation.Token); }
            catch (OperationCanceledException) { Status = "キャンセルしました。再実行できます。"; throw; }
            catch (Exception error) { LastError = error.Message; Status = "処理に失敗しました。入力を確認して再実行してください。"; throw; }
            finally { operationCancellation = null; cancellation.Dispose(); IsBusy = false; }
        }
        async Task SelectModule(string name, CancellationToken token)
        {
            if (session != null && session.ModuleName == name && !session.IsClosed) return;
            Status = "前のモデルを解放しています…"; await CloseModule(); token.ThrowIfCancellationRequested();
            Status = "モデルのキャッシュを照合しています…";
            var source = string.IsNullOrWhiteSpace(SourceDirectory) ? Path.Combine(Application.streamingAssetsPath, "EmbeddingGemmaMedia") : SourceDirectory;
            var cache = string.IsNullOrWhiteSpace(CacheDirectory) ? Path.Combine(Application.persistentDataPath, "EmbeddingGemmaMediaCache") : CacheDirectory;
            lease = await GpuMediaBundleCache.AcquireAsync(source, name, cache, token);
            LastTransferredFiles = lease.TransferredFiles; LastReusedObjects = lease.ReusedObjects;
            try { Status = "モデルをGPUへ読み込んでいます…"; session = await lease.Bundle.LoadAsync(name, token); }
            catch { lease.Dispose(); lease = null; throw; }
        }
        async Task CloseModule()
        {
            var previous = session; session = null;
            try { if (previous != null) await previous.CloseAsync(); }
            catch (Exception error) { LastError = error.Message; }
            finally { lease?.Dispose(); lease = null; }
        }
        public Task CloseAsync()
        {
            if (closing) return closeTask ?? Task.CompletedTask;
            closing = true; Cancel(); closeTask = CloseCore(); return closeTask;
        }
        async Task CloseCore()
        {
            if (activeOperation != null) try { await activeOperation; } catch (Exception error) { if (!(error is OperationCanceledException)) LastError = error.Message; }
            await CloseModule();
        }
        async void GuiAction(Func<Task> action) { try { await action(); } catch (Exception) { /* Execute retained the failure for display. */ } }

        void OnGUI()
        {
            if (font == null) font = MediaSearchGuiFont.Acquire();
            styles ??= new MediaSearchGuiStyles(GUI.skin, font);
            GUILayout.BeginArea(new Rect(20, 20, Mathf.Max(240, Screen.width - 40), Mathf.Max(240, Screen.height - 40)));
            scroll = GUILayout.BeginScrollView(scroll);
            GUILayout.Label("画像・音声のGPU検索", styles.Heading);
            GUILayout.Label("画像または音声を登録してから、テキストで検索します。", styles.Label);
            var enabled = GUI.enabled; GUI.enabled = !IsBusy && !closing;
            GUILayout.Label("準備済みモデルのフォルダー", styles.Label); SourceDirectory = GUILayout.TextField(SourceDirectory, styles.TextField);
            GUILayout.Label("入力ファイル（PNG / JPEG、mono 16 kHz WAV / .f32）", styles.Label); MediaPath = GUILayout.TextField(MediaPath, styles.TextField);
            var selected = Math.Max(0, Array.IndexOf(Modules, ModuleName)); selected = GUILayout.SelectionGrid(selected, Modules, 3, styles.Button); ModuleName = Modules[selected];
            if (GUILayout.Button("画像・音声を登録", styles.Button)) GuiAction(() => AddAsync());
            GUILayout.Label("検索文", styles.Label); Query = GUILayout.TextField(Query, styles.TextField);
            if (GUILayout.Button("検索", styles.Button)) GuiAction(() => SearchAsync());
            GUI.enabled = enabled && IsBusy; if (GUILayout.Button("キャンセル", styles.Button)) Cancel(); GUI.enabled = enabled;
            GUILayout.Label(Status, styles.Label); if (!string.IsNullOrEmpty(LastError)) GUILayout.Label(LastError, styles.Error);
            GUILayout.Label("登録数: " + documents.Count, styles.Label);
            foreach (var hit in Hits) GUILayout.Label(hit.Score.ToString("F4") + "  " + hit.Document.Title + "\n" + hit.Document.Text, styles.Label);
            GUILayout.EndScrollView(); GUILayout.EndArea();
        }
    }
}
