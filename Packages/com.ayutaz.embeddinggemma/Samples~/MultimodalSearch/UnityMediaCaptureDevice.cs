using System;
using UnityEngine;

namespace EmbeddingGemma.Samples
{
    /// <summary>Injectable native boundary. Tests can exercise device health without opening a microphone.</summary>
    public interface IMicrophoneCaptureApi
    {
        bool PermissionGranted { get; }
        string[] Devices { get; }
        AudioClip Start(string device, bool loop, int lengthSeconds, int frequency);
        int GetPosition(string device);
        bool IsRecording(string device);
        void End(string device);
    }

    /// <summary>Explicit, memory-only mono microphone capture. Never requests permissions, saves or transmits audio.</summary>
    public sealed class UnityMediaCaptureDevice : IMediaCaptureDevice
    {
        AudioClip clip;
        string device;
        const double ProgressTimeoutSeconds = 2;
        bool startRequested;
        int lastPosition;
        double lastProgressAt;
        readonly IMicrophoneCaptureApi api;
        readonly Func<double> realtime;
        public UnityMediaCaptureDevice() : this(new NativeApi(), () => Time.realtimeSinceStartupAsDouble) { }
        public UnityMediaCaptureDevice(IMicrophoneCaptureApi api, Func<double> realtime)
        {
            this.api = api ?? throw new ArgumentNullException(nameof(api));
            this.realtime = realtime ?? throw new ArgumentNullException(nameof(realtime));
        }
        public bool PermissionGranted => api.PermissionGranted;
        public string[] Devices => api.Devices;
        public int CapturedSamples
        {
            get
            {
                if (clip == null) return 0;
                var position = api.GetPosition(device);
                if (position < 0 || position > clip.samples || position < lastPosition ||
                    (!api.IsRecording(device) && (position > 0 || lastPosition > 0)))
                    throw new InvalidOperationException("マイク入力が中断されました。デバイスを確認して録音を再開してください。");
                var now = realtime();
                if (position > lastPosition) lastProgressAt = now;
                else if (now - lastProgressAt >= ProgressTimeoutSeconds)
                    throw new InvalidOperationException("マイクから音声を取得できません。デバイスを確認して録音を再開してください。");
                lastPosition = position;
                return position;
            }
        }
        public void Start(string deviceName, int frequency, int maximumSamples)
        {
            if (startRequested || clip != null) throw new InvalidOperationException("Microphone capture is already active.");
            device = deviceName;
            startRequested = true;
            try
            {
                clip = api.Start(device, false, Mathf.CeilToInt(maximumSamples / (float)frequency), frequency);
                if (clip == null || clip.channels != 1 || clip.frequency != AudioPcmInput.RequiredSamplingRate)
                    throw new InvalidOperationException("mono 16kHzの録音を開始できません。対応する入力デバイスを選択してください。");
                lastPosition = 0;
                lastProgressAt = realtime();
            }
            catch
            {
                try { Cancel(); } catch { /* Preserve the original start failure. */ }
                throw;
            }
        }
        public float[] Stop(int sampleCount)
        {
            if (clip == null || sampleCount <= 0 || sampleCount > clip.samples) throw new InvalidOperationException("録音データが不足しています。");
            EndCapture();
            var samples = new float[sampleCount];
            if (!clip.GetData(samples, 0)) throw new InvalidOperationException("録音データを読み出せません。");
            return samples;
        }
        public void Cancel()
        {
            try { EndCapture(); }
            finally
            {
                device = null;
                lastPosition = 0;
                lastProgressAt = 0;
                MediaPreviewStore.Release(clip);
                clip = null;
            }
        }
        void EndCapture()
        {
            if (!startRequested) return;
            startRequested = false;
            api.End(device);
        }
        sealed class NativeApi : IMicrophoneCaptureApi
        {
            public bool PermissionGranted => Application.HasUserAuthorization(UserAuthorization.Microphone);
            public string[] Devices => Microphone.devices;
            public AudioClip Start(string device, bool loop, int lengthSeconds, int frequency) => Microphone.Start(device, loop, lengthSeconds, frequency);
            public int GetPosition(string device) => Microphone.GetPosition(device);
            public bool IsRecording(string device) => Microphone.IsRecording(device);
            public void End(string device) => Microphone.End(device);
        }
    }
}
