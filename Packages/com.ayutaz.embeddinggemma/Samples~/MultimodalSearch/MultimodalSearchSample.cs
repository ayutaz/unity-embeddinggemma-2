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
        public bool IsRecording => recording?.IsRecording ?? false;
        public AudioPcmInput CapturedPcm { get; private set; }
        public AudioClip RecordingClip => recordingClip;
        public int PreviewCount => previews.Count + (recordingClip != null ? 1 : 0);
        MediaPreviewStore previews = new MediaPreviewStore();
        MediaCaptureSession recording;
        IMediaCaptureDevice captureDevice;
        AudioClip recordingClip;
        AudioSource playback;
        int recordingPreset = 1;
        bool showRecording;

        void CaptureAction(Action action)
        {
            LastError = null;
            try { action(); }
            catch (Exception error) { LastError = error.Message; Status = "録音を完了できません。状態を確認して再実行してください。"; throw; }
        }
        public void SetCaptureDevice(IMediaCaptureDevice device)
        {
            if (IsBusy || IsRecording || closing) throw new InvalidOperationException("進行中の処理が完了するまで入力デバイスは変更できません。");
            if (device == null) throw new ArgumentNullException(nameof(device));
            recording?.Dispose(); captureDevice = device; recording = new MediaCaptureSession(device);
        }
        public void StartRecording(int maximumSamples = AudioPcmInput.ProfileSampleCount) => CaptureAction(() =>
        {
            if (IsBusy || closing) throw new InvalidOperationException("モデル処理の終了またはリセット後に録音してください。");
            recording ??= new MediaCaptureSession(captureDevice ?? new UnityMediaCaptureDevice());
            recording.Start(maximumSamples); StopPlayback(); ClearRecording();
            Status = "録音中です。停止・破棄、または選択した最大時間で終了します。音声はメモリ内だけに保持します。";
        });
        public void StopRecording() => CaptureAction(() =>
        {
            if (!IsRecording) throw new InvalidOperationException("録音は開始されていません。");
            var pcm = recording.Stop(); var clip = MakeAudioClip(pcm, "録音プレビュー"); ClearRecording();
            CapturedPcm = pcm; recordingClip = clip;
            var supported = pcm.ValidFrameCount == 1119 || pcm.ValidFrameCount == 839;
            if (supported) ModuleName = "audio-" + pcm.ValidFrameCount;
            Status = supported ? "録音を停止しました。再生またはGPU登録を選択してください。" : "録音を停止しました。再生できます。GPU登録には8.4秒または11.2秒の固定profileが必要です。自動で長さは変更しません。";
        });
        void ClearRecording()
        { StopPlayback(); MediaPreviewStore.Release(recordingClip); recordingClip = null; CapturedPcm = null; }
        public void CancelRecording()
        {
            try { recording?.Cancel(); }
            finally { ClearRecording(); Status = "録音を破棄しました。音声の保存・送信は行っていません。"; }
        }
        static AudioClip MakeAudioClip(AudioPcmInput pcm, string title)
        {
            var clip = AudioClip.Create(title, pcm.SourceSampleCount, 1, AudioPcmInput.RequiredSamplingRate, false);
            try
            {
                if (clip == null || !clip.SetData(pcm.Waveform.Slice(0, pcm.SourceSampleCount).ToArray(), 0)) throw new InvalidOperationException("音声プレビューを準備できません。");
                return clip;
            }
            catch { MediaPreviewStore.Release(clip); throw; }
        }
        void PlayClip(AudioClip clip)
        {
            if (clip == null) throw new InvalidOperationException("再生する音声がありません。");
            if (IsRecording) throw new InvalidOperationException("録音を停止してから再生してください。");
            if (FindFirstObjectByType<AudioListener>() == null) gameObject.AddComponent<AudioListener>();
            if (playback == null)
            {
                playback = GetComponent<AudioSource>();
                if (playback == null) playback = gameObject.AddComponent<AudioSource>();
            }
            playback.playOnAwake = false; playback.loop = false; playback.spatialBlend = 0; playback.volume = .25f;
            playback.Stop(); playback.clip = clip; playback.Play();
        }
        public void PlayRecording() => CaptureAction(() => PlayClip(recordingClip));
        public void StopPlayback() { if (playback != null) { playback.Stop(); playback.clip = null; } }
        public Task AddRecordingAsync(CancellationToken cancellationToken = default) => StartOperation(async token =>
        {
            var pcm = CapturedPcm ?? throw new InvalidOperationException("先に録音を停止してください。");
            if (pcm.ValidFrameCount != 1119 && pcm.ValidFrameCount != 839) throw new ArgumentException("録音の長さに対応する固定profileがありません。8.4秒または11.2秒で録音してください。");
            await SelectModule("audio-" + pcm.ValidFrameCount, token); Status = "録音をGPUで登録しています…";
            var vector = await session.EmbedAudioAsync(pcm, cancellationToken: token); token.ThrowIfCancellationRequested();
            var id = "media-" + documents.Count; var title = "録音 " + (documents.Count + 1);
            previews.SetAudio(id, MakeAudioClip(pcm, title));
            documents.Add(new SearchDocument(id, "メモリ内の録音（ファイル保存なし）", title)); vectors.Add(vector);
            Hits = Array.Empty<SearchHit>(); Status = documents.Count + "件を登録しました。テキストで検索できます。";
        }, cancellationToken);
        public async Task ResetAsync()
        {
            await CloseAsync(); documents.Clear(); vectors.Clear(); Hits = Array.Empty<SearchHit>();
            closing = !isActiveAndEnabled; Status = "登録とモデルを解放しました。もう一度準備できます。";
        }
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
        void Update()
        {
            if (!IsRecording) return;
            try { if (recording.ReachedLimit) StopRecording(); }
            catch (Exception error) { LastError = error.Message; try { CancelRecording(); } catch { /* Retain the first device error. */ } }
        }
        void OnApplicationPause(bool paused)
        {
            if (!paused) return;
            if (IsRecording) CancelRecording();
            StopPlayback();
        }
        void OnApplicationFocus(bool focused) { if (!focused && IsRecording) CancelRecording(); }

        public Task AddAsync(CancellationToken cancellationToken = default) => StartOperation(async token =>
        {
            var path = Path.GetFullPath(MediaPath); var module = ModuleName;
            if (!File.Exists(path)) throw new FileNotFoundException("入力ファイルが見つかりません。", path);
            float[] vector; var id = "media-" + documents.Count;
            Texture previewImage = null; AudioClip previewAudio = null;
            try
            {
            if (module.StartsWith("image-", StringComparison.Ordinal))
            {
                var texture = new Texture2D(2, 2, TextureFormat.RGBA32, false, true);
                try
                {
                    if (!ImageConversion.LoadImage(texture, File.ReadAllBytes(path), true)) throw new InvalidDataException("PNG / JPEG画像を読み込めません。");
                    await SelectModule(module, token); Status = "画像をGPUで登録しています…";
                    vector = await session.EmbedImageAsync(texture, cancellationToken: token);
                    token.ThrowIfCancellationRequested(); previewImage = MediaPreviewStore.CreateThumbnail(texture);
                }
                finally { if (Application.isPlaying) Destroy(texture); else DestroyImmediate(texture); }
            }
            else if (module.StartsWith("audio-", StringComparison.Ordinal))
            {
                var pcm = MediaSampleInput.ReadPcm(path);
                if (module != "audio-" + pcm.ValidFrameCount) throw new ArgumentException("音声の長さに対応する固定profileを選択してください。通常音声は1119、短い検証音声は839です。");
                await SelectModule(module, token); Status = "音声をGPUで登録しています…";
                vector = await session.EmbedAudioAsync(pcm, cancellationToken: token);
                token.ThrowIfCancellationRequested(); previewAudio = MakeAudioClip(pcm, Path.GetFileName(path));
            }
            else throw new ArgumentException("画像または音声moduleを選択してください。");
            token.ThrowIfCancellationRequested();
            if (previewImage != null) { previews.SetImage(id, previewImage); previewImage = null; }
            if (previewAudio != null) { previews.SetAudio(id, previewAudio); previewAudio = null; }
            documents.Add(new SearchDocument(id, path, Path.GetFileName(path))); vectors.Add(vector);
            Hits = Array.Empty<SearchHit>(); Status = documents.Count + "件を登録しました。テキストで検索できます。";
            }
            finally { MediaPreviewStore.Release(previewImage); MediaPreviewStore.Release(previewAudio); }
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
            if (IsBusy || IsRecording || closing) return Task.FromException(new InvalidOperationException("進行中の処理または録音が完了するまでお待ちください。"));
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
            closing = true; Cancel();
            try { recording?.Cancel(); } catch (Exception error) { LastError = error.Message; }
            StopPlayback(); closeTask = CloseCore(); return closeTask;
        }
        async Task CloseCore()
        {
            if (activeOperation != null) try { await activeOperation; } catch (Exception error) { if (!(error is OperationCanceledException)) LastError = error.Message; }
            try { await CloseModule(); }
            finally { recording?.Dispose(); recording = null; ClearRecording(); previews.Dispose(); previews = new MediaPreviewStore(); }
        }
        async void GuiAction(Func<Task> action) { try { await action(); } catch (Exception error) { if (!(error is OperationCanceledException)) LastError = error.Message; } }
        void GuiCaptureAction(Action action) { try { action(); } catch (Exception error) { LastError = error.Message; } }

        void OnGUI()
        {
            if (font == null) font = MediaSearchGuiFont.Acquire();
            styles ??= new MediaSearchGuiStyles(GUI.skin, font);
            GUILayout.BeginArea(new Rect(20, 20, Mathf.Max(240, Screen.width - 40), Mathf.Max(240, Screen.height - 40)));
            scroll = GUILayout.BeginScrollView(scroll);
            GUILayout.Label("画像・音声のGPU検索", styles.Heading);
            GUILayout.Label("画像または音声を登録してから、テキストで検索します。", styles.Label);
            var enabled = GUI.enabled; GUI.enabled = enabled && !IsBusy && !IsRecording && !closing;
            GUILayout.Label("準備済みモデルのフォルダー", styles.Label); SourceDirectory = GUILayout.TextField(SourceDirectory, styles.TextField);
            GUILayout.Label("入力ファイル（PNG / JPEG、mono 16 kHz WAV / .f32）", styles.Label); MediaPath = GUILayout.TextField(MediaPath, styles.TextField);
            var selected = Math.Max(0, Array.IndexOf(Modules, ModuleName)); selected = GUILayout.SelectionGrid(selected, Modules, 3, styles.Button); ModuleName = Modules[selected];
            if (GUILayout.Button("画像・音声を登録", styles.Button)) GuiAction(() => AddAsync());
            showRecording = GUILayout.Toggle(showRecording, "マイク入力（任意・メモリ内だけに保持）", styles.Button);
            if (showRecording)
            {
                GUILayout.Label("最大録音時間。早く停止した音声は長さを自動変更せず、対応profileの場合だけGPU登録できます。", styles.Label);
                recordingPreset = GUILayout.SelectionGrid(recordingPreset, new[] { "8.4秒 / audio-839", "11.2秒 / audio-1119" }, 2, styles.Button);
                if (GUILayout.Button("録音を開始", styles.Button)) GuiCaptureAction(() => StartRecording(recordingPreset == 0 ? 134400 : 179200));
                GUI.enabled = enabled && IsRecording;
                if (GUILayout.Button("録音を停止", styles.Button)) GuiCaptureAction(StopRecording);
                if (GUILayout.Button("録音を破棄", styles.Button)) GuiCaptureAction(CancelRecording);
                GUI.enabled = enabled && !IsBusy && !IsRecording && !closing && recordingClip != null;
                if (GUILayout.Button("録音を再生", styles.Button)) GuiCaptureAction(PlayRecording);
                if (GUILayout.Button("録音をGPUで登録", styles.Button)) GuiAction(() => AddRecordingAsync());
            }
            GUI.enabled = enabled && !IsBusy && !IsRecording && !closing;
            GUILayout.Label("検索文", styles.Label); Query = GUILayout.TextField(Query, styles.TextField);
            if (GUILayout.Button("検索", styles.Button)) GuiAction(() => SearchAsync());
            GUI.enabled = enabled && IsBusy; if (GUILayout.Button("キャンセル", styles.Button)) Cancel(); GUI.enabled = enabled;
            if (GUILayout.Button("再生を停止", styles.Button)) StopPlayback();
            GUI.enabled = enabled && !IsBusy && !IsRecording;
            if (GUILayout.Button("モデル・登録内容を解放して最初から", styles.Button)) GuiAction(ResetAsync);
            GUI.enabled = enabled;
            GUILayout.Label(Status, styles.Label); if (!string.IsNullOrEmpty(LastError)) GUILayout.Label(LastError, styles.Error);
            GUILayout.Label("登録数: " + documents.Count, styles.Label);
            foreach (var hit in Hits)
            {
                GUILayout.BeginVertical(GUI.skin.box);
                GUILayout.Label(hit.Score.ToString("F4") + "  " + hit.Document.Title + "\n" + hit.Document.Text, styles.Label);
                var image = previews.Image(hit.Document.Id); if (image != null) GUILayout.Label(image, GUILayout.Width(image.width), GUILayout.Height(image.height));
                var audio = previews.Audio(hit.Document.Id);
                GUI.enabled = enabled && !IsRecording;
                if (audio != null && GUILayout.Button("この音声を再生", styles.Button)) GuiCaptureAction(() => PlayClip(audio));
                GUI.enabled = enabled; GUILayout.EndVertical();
            }
            GUILayout.EndScrollView(); GUILayout.EndArea();
        }
    }
}
