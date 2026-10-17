using System;
using System.Collections;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;

namespace EmbeddingGemma.Samples.Tests
{
    /// <summary>Actual Unity audio engine acceptance using generated PCM, not live microphone capture.</summary>
    public sealed class MediaPlaybackTests
    {
        sealed class GeneratedPcm : IMediaCaptureDevice
        {
            public bool PermissionGranted => true;
            public string[] Devices => new[] { "generated PCM acceptance fixture" };
            public int CapturedSamples => 179200;
            public void Start(string device, int frequency, int maximumSamples) { }
            public float[] Stop(int count) => new float[count];
            public void Cancel() { }
        }
        [UnityTest] public IEnumerator ExplicitPlaybackAdvancesAudioClockAndStopResetReleaseItsReferences()
        {
            yield return new EnterPlayMode();
            var host = new GameObject("real audio engine acceptance"); var sample = host.AddComponent<MultimodalSearchSample>();
            try
            {
                sample.SetCaptureDevice(new GeneratedPcm()); sample.StartRecording(); sample.StopRecording(); var clip = sample.RecordingClip;
                Assert.That(host.GetComponent<AudioSource>(), Is.Null, "Creating a preview must never autoplay.");
                sample.PlayRecording(); var audio = host.GetComponent<AudioSource>(); Assert.That(audio, Is.Not.Null);
                Assert.That(audio.clip, Is.SameAs(clip)); Assert.That(audio.playOnAwake, Is.False); Assert.That(audio.loop, Is.False);
                var deadline = Time.realtimeSinceStartupAsDouble + 2;
                while (audio.timeSamples == 0 && Time.realtimeSinceStartupAsDouble < deadline) yield return null;
                Assert.That(audio.isPlaying, Is.True, "Audio driver playback must run; never skip.");
                Assert.That(audio.timeSamples, Is.GreaterThan(0), "Actual DSP playback must advance; selecting a clip is insufficient.");
                sample.StopPlayback(); Assert.That(audio.isPlaying, Is.False); Assert.That(audio.clip, Is.Null); Assert.That(clip != null, Is.True);
                sample.PlayRecording(); var reset = sample.ResetAsync(); while (!reset.IsCompleted) yield return null; reset.GetAwaiter().GetResult();
                yield return null;
                Assert.That(audio.isPlaying, Is.False); Assert.That(audio.clip, Is.Null); Assert.That(clip == null, Is.True);
                Assert.That(sample.PreviewCount, Is.Zero); Assert.That(sample.CapturedPcm, Is.Null);
            }
            finally { UnityEngine.Object.Destroy(host); }
            yield return new ExitPlayMode();
        }
    }
}
