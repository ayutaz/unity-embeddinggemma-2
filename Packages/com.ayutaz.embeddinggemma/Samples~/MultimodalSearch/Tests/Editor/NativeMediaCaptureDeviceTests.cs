using System;
using System.Collections.Generic;
using System.Reflection;
using NUnit.Framework;
using UnityEngine;

namespace EmbeddingGemma.Samples.Tests
{
    /// <summary>Injected native boundary, generated clips only. Never calls Unity Microphone.Start.</summary>
    public sealed class NativeMediaCaptureDeviceTests
    {
        sealed class Api : IMicrophoneCaptureApi
        {
            public bool PermissionGranted { get; set; } = true;
            public string[] Devices { get; set; } = new[] { "synthetic input" };
            public bool Recording = true; public int Position; public int Channels = 1;
            public int Starts, Ends; public bool Loop; public string LastDevice;
            public readonly List<AudioClip> Clips = new List<AudioClip>();
            public AudioClip Start(string device, bool loop, int seconds, int frequency)
            {
                Starts++; LastDevice = device; Loop = loop;
                var clip = AudioClip.Create("synthetic input buffer", frequency * seconds, Channels, frequency, false);
                Clips.Add(clip); return clip;
            }
            public int GetPosition(string device) => Position;
            public bool IsRecording(string device) => Recording;
            public void End(string device) { Ends++; LastDevice = device; Recording = false; }
        }
        Api api; UnityMediaCaptureDevice device; double now;
        [SetUp] public void Setup() { api = new Api(); now = 0; device = new UnityMediaCaptureDevice(api, () => now); }
        [TearDown] public void Cleanup()
        { device.Cancel(); foreach (var clip in api.Clips) if (clip != null) UnityEngine.Object.DestroyImmediate(clip); }

        [Test] public void ConstructionAndInspectionDoNotStartCapture()
        {
            Assert.That(device.PermissionGranted, Is.True); Assert.That(device.Devices, Has.Length.EqualTo(1));
            Assert.That(device.CapturedSamples, Is.Zero); device.Cancel();
            Assert.That(api.Starts, Is.Zero); Assert.That(api.Ends, Is.Zero);
        }
        [Test] public void InitialStartupGapCanRecoverBeforeTimeout()
        {
            api.Recording = false; device.Start("synthetic input", 16000, 179200);
            now = 1; Assert.That(device.CapturedSamples, Is.Zero);
            api.Recording = true; api.Position = 3200; now = 1.1;
            Assert.That(device.CapturedSamples, Is.EqualTo(3200)); Assert.That(api.Loop, Is.False);
        }
        [Test] public void NativeStoppedStateCannotReturnStaleRecordedCount()
        {
            device.Start("synthetic input", 16000, 179200); api.Position = 3200;
            Assert.That(device.CapturedSamples, Is.EqualTo(3200)); api.Recording = false;
            Assert.Throws<InvalidOperationException>(() => { var ignored = device.CapturedSamples; });
        }
        [TestCase(false)] [TestCase(true)]
        public void MissingStartupProgressHasABoundedFailureEvenIfNativeReportsRecording(bool recording)
        {
            api.Recording = recording; device.Start("synthetic input", 16000, 179200); now = 3;
            Assert.Throws<InvalidOperationException>(() => { var ignored = device.CapturedSamples; });
        }
        [Test] public void StalledPositionAfterProgressFailsWithoutPaddingOrReturningAudio()
        {
            device.Start("synthetic input", 16000, 179200); api.Position = 3200;
            Assert.That(device.CapturedSamples, Is.EqualTo(3200)); now = 3;
            Assert.Throws<InvalidOperationException>(() => { var ignored = device.CapturedSamples; });
        }
        [Test] public void NonLoopingPositionRegressionIsAnInterruption()
        {
            device.Start("synthetic input", 16000, 179200); api.Position = 3200;
            Assert.That(device.CapturedSamples, Is.EqualTo(3200)); api.Position = 1600;
            Assert.Throws<InvalidOperationException>(() => { var ignored = device.CapturedSamples; });
        }
        [Test] public void DefaultDeviceNameStillEndsCaptureAndReleasesClipExactlyOnce()
        {
            device.Start(null, 16000, 179200); var clip = api.Clips[0]; device.Cancel(); device.Cancel();
            Assert.That(api.Ends, Is.EqualTo(1)); Assert.That(api.LastDevice, Is.Null); Assert.That(clip == null, Is.True);
        }
        [Test] public void FailedNativeFormatIsReleasedAndCanBeRetriedDirectly()
        {
            api.Channels = 2; Assert.Throws<InvalidOperationException>(() => device.Start("synthetic input", 16000, 179200));
            Assert.That(api.Clips[0] == null, Is.True); Assert.That(api.Ends, Is.EqualTo(1));
            api.Channels = 1; api.Recording = true; device.Start("synthetic input", 16000, 179200);
            Assert.That(api.Starts, Is.EqualTo(2));
        }
        [Test] public void SampleInterruptionReleasesBufferAndAllowsExplicitRetryWithoutInference()
        {
            var host = new GameObject("synthetic native interruption");
            try
            {
                var sample = host.AddComponent<MultimodalSearchSample>(); sample.SetCaptureDevice(device); sample.StartRecording();
                api.Position = 3200; Assert.That(device.CapturedSamples, Is.EqualTo(3200)); api.Recording = false;
                typeof(MultimodalSearchSample).GetMethod("Update", BindingFlags.Instance | BindingFlags.NonPublic).Invoke(sample, null);
                Assert.That(sample.IsRecording, Is.False); Assert.That(sample.LastError, Is.Not.Null.And.Not.Empty);
                Assert.That(sample.CapturedPcm, Is.Null); Assert.That(sample.RecordingClip, Is.Null);
                Assert.That(api.Clips[0] == null, Is.True); Assert.That(sample.DocumentCount, Is.Zero);
                api.Recording = true; api.Position = 0; sample.StartRecording(); Assert.That(sample.IsRecording, Is.True);
                sample.CancelRecording(); Assert.That(sample.IsRecording, Is.False);
            }
            finally { UnityEngine.Object.DestroyImmediate(host); }
        }
    }
}
