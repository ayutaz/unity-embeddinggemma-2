using System;
using UnityEngine;

namespace EmbeddingGemma.Samples
{
    /// <summary>Explicit, memory-only mono microphone capture. Never requests permissions, saves or transmits audio.</summary>
    public sealed class UnityMediaCaptureDevice : IMediaCaptureDevice
    {
        AudioClip clip;
        string device;
        public bool PermissionGranted => Application.HasUserAuthorization(UserAuthorization.Microphone);
        public string[] Devices => Microphone.devices;
        public int CapturedSamples
        {
            get
            {
                if (clip == null) return 0;
                var position = Microphone.GetPosition(device);
                if (position < 0) throw new InvalidOperationException("マイクから音声を取得できません。デバイスを確認してください。");
                return position;
            }
        }
        public void Start(string deviceName, int frequency, int maximumSamples)
        {
            if (clip != null) throw new InvalidOperationException("Microphone capture is already active.");
            device = deviceName;
            clip = Microphone.Start(device, false, Mathf.CeilToInt(maximumSamples / (float)frequency), frequency);
            if (clip == null || clip.channels != 1 || clip.frequency != AudioPcmInput.RequiredSamplingRate)
                throw new InvalidOperationException("mono 16kHzの録音を開始できません。対応する入力デバイスを選択してください。");
        }
        public float[] Stop(int sampleCount)
        {
            if (clip == null || sampleCount <= 0 || sampleCount > clip.samples) throw new InvalidOperationException("録音データが不足しています。");
            Microphone.End(device);
            var samples = new float[sampleCount];
            if (!clip.GetData(samples, 0)) throw new InvalidOperationException("録音データを読み出せません。");
            return samples;
        }
        public void Cancel()
        {
            try { if (device != null) Microphone.End(device); }
            finally { device = null; MediaPreviewStore.Release(clip); clip = null; }
        }
    }
}
