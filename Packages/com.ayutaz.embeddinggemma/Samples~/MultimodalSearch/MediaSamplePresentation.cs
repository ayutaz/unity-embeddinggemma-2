using System;
using System.Collections.Generic;
using UnityEngine;

namespace EmbeddingGemma.Samples
{
    public sealed class MediaPreviewStore : IDisposable
    {
        readonly Dictionary<string, UnityEngine.Object> owners = new Dictionary<string, UnityEngine.Object>(StringComparer.Ordinal);
        bool disposed;
        public static RenderTexture CreateThumbnail(Texture source, int maximumDimension = 160)
        {
            if (source == null) throw new ArgumentNullException(nameof(source));
            if (maximumDimension <= 0) throw new ArgumentOutOfRangeException(nameof(maximumDimension));
            var scale = Mathf.Min(1f, maximumDimension / (float)Mathf.Max(source.width, source.height));
            var result = new RenderTexture(Mathf.Max(1, Mathf.RoundToInt(source.width * scale)), Mathf.Max(1, Mathf.RoundToInt(source.height * scale)), 0, RenderTextureFormat.ARGB32, RenderTextureReadWrite.Linear);
            try
            {
                if (!result.Create()) throw new InvalidOperationException("GPU preview allocation failed.");
                Graphics.Blit(source, result); return result;
            }
            catch { Release(result); throw; }
        }
        public int Count => owners.Count;
        void EnsureOpen() { if (disposed) throw new ObjectDisposedException(nameof(MediaPreviewStore)); }
        void Set(string id, UnityEngine.Object value)
        {
            EnsureOpen();
            if (string.IsNullOrWhiteSpace(id)) throw new ArgumentException("A media ID is required.", nameof(id));
            if (value == null) throw new ArgumentNullException(nameof(value));
            if (owners.TryGetValue(id, out var previous) && !ReferenceEquals(previous, value)) Release(previous);
            owners[id] = value;
        }
        public void SetImage(string id, Texture image) => Set(id, image);
        public void SetAudio(string id, AudioClip audio) => Set(id, audio);
        public Texture Image(string id) { EnsureOpen(); return owners.TryGetValue(id, out var value) ? value as Texture : null; }
        public AudioClip Audio(string id) { EnsureOpen(); return owners.TryGetValue(id, out var value) ? value as AudioClip : null; }
        internal static void Release(UnityEngine.Object value)
        {
            if (value == null) return;
            if (value is RenderTexture render) render.Release();
            if (Application.isPlaying) UnityEngine.Object.Destroy(value); else UnityEngine.Object.DestroyImmediate(value);
        }
        public void Dispose()
        {
            if (disposed) return;
            disposed = true; foreach (var value in owners.Values) Release(value); owners.Clear();
        }
    }

    public interface IMediaCaptureDevice
    {
        bool PermissionGranted { get; }
        string[] Devices { get; }
        int CapturedSamples { get; }
        void Start(string device, int frequency, int maximumSamples);
        float[] Stop(int sampleCount);
        void Cancel();
    }

    public sealed class MediaCaptureSession : IDisposable
    {
        readonly IMediaCaptureDevice device;
        bool disposed;
        int maximum;
        public MediaCaptureSession(IMediaCaptureDevice device) { this.device = device ?? throw new ArgumentNullException(nameof(device)); }
        public bool IsRecording { get; private set; }
        public bool ReachedLimit => IsRecording && device.CapturedSamples >= maximum;
        public void Start(int maximumSamples = AudioPcmInput.ProfileSampleCount)
        {
            if (disposed) throw new ObjectDisposedException(nameof(MediaCaptureSession));
            if (IsRecording) throw new InvalidOperationException("A recording is already active.");
            if (maximumSamples < AudioPcmInput.MinimumSampleCount || maximumSamples > AudioPcmInput.ProfileSampleCount)
                throw new ArgumentOutOfRangeException(nameof(maximumSamples));
            if (!device.PermissionGranted) throw new UnauthorizedAccessException("マイクの利用権限がありません。");
            var devices = device.Devices;
            if (devices == null || devices.Length == 0) throw new InvalidOperationException("利用できるマイクがありません。");
            maximum = maximumSamples;
            try { device.Start(devices[0], AudioPcmInput.RequiredSamplingRate, maximum); IsRecording = true; }
            catch { try { device.Cancel(); } catch { /* Keep the original device error. */ } throw; }
        }
        public AudioPcmInput Stop()
        {
            if (disposed) throw new ObjectDisposedException(nameof(MediaCaptureSession));
            if (!IsRecording) throw new InvalidOperationException("録音は開始されていません。");
            IsRecording = false;
            try
            {
                var count = Math.Min(maximum, device.CapturedSamples);
                if (count < AudioPcmInput.MinimumSampleCount) throw new ArgumentException("録音が短すぎます。もう一度録音してください。");
                var values = device.Stop(count);
                if (values == null || values.Length != count) throw new InvalidOperationException("マイクの読み取りが途中で終了しました。");
                return new AudioPcmInput(values);
            }
            finally { device.Cancel(); }
        }
        public void Cancel()
        {
            if (!IsRecording) return;
            IsRecording = false; device.Cancel();
        }
        public void Dispose()
        {
            if (disposed) return;
            try { Cancel(); } finally { disposed = true; }
        }
    }
}
