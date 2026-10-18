using System;
using NUnit.Framework;
using UnityEngine;

namespace EmbeddingGemma.Samples.Tests
{
    public sealed class MediaPresentationTests
    {
        [TestCase("image.png", "image.png")]
        [TestCase("R:/fixtures/image.png", "image.png")]
        [TestCase(@"R:\fixtures\sound.wav", "sound.wav")]
        [TestCase("/fixtures/画像.png", "画像.png")]
        [TestCase("録音 1", "録音 1")]
        public void ResultLabelShowsScoreAndNameWithoutSourceDirectories(string title, string name)
        {
            var method = typeof(MultimodalSearchSample).GetMethod("ResultLabel");
            Assert.That(method, Is.Not.Null, "The displayed result must exclude source directory metadata.");
            var document = new SearchDocument("fixture", "R:/fixtures/source-only/input.png", title);
            var label = (string)method.Invoke(null, new object[] { document, .625 });
            Assert.That(label, Is.EqualTo(.625.ToString("F4") + "  " + name));
            Assert.That(document.Text, Is.EqualTo("R:/fixtures/source-only/input.png"), "Display privacy must not mutate input provenance.");
        }

        [TestCase(null)] [TestCase("")] [TestCase("Cannot read R:/fixtures/source-only/input.png")]
        public void ErrorLabelDoesNotExposeRawDiagnosticPaths(string diagnostic)
        {
            var method = typeof(MultimodalSearchSample).GetMethod("ErrorLabel");
            Assert.That(method, Is.Not.Null, "Raw exception messages must not be rendered in the sample UI.");
            var label = (string)method.Invoke(null, new object[] { diagnostic });
            if (string.IsNullOrEmpty(diagnostic)) Assert.That(label, Is.Null);
            else
            {
                Assert.That(label, Is.Not.Null.And.Not.Empty);
                Assert.That(label, Does.Contain("設定").And.Not.Contain("R:").And.Not.Contain("source-only"));
            }
        }

        sealed class Capture : IMediaCaptureDevice
        {
            public bool PermissionGranted { get; set; } = true;
            public string[] Devices { get; set; } = new[] { "fake input" };
            public int CapturedSamples { get; set; } = AudioPcmInput.ProfileSampleCount;
            public int Starts, Stops, Cancels, Rate, Maximum, Requested;
            public Exception StartError, StopError;
            public void Start(string device, int frequency, int maximumSamples)
            { Starts++; Rate = frequency; Maximum = maximumSamples; if (StartError != null) throw StartError; }
            public float[] Stop(int count)
            { Stops++; Requested = count; if (StopError != null) throw StopError; var values = new float[count]; values[0] = .25f; return values; }
            public void Cancel() { Cancels++; }
        }

        [Test] public void PreviewStoreKeepsMediaByIdAndReleasesReplacedAndDisposedOwners()
        {
            var first = new Texture2D(2, 3); var second = new Texture2D(4, 2);
            var audio = AudioClip.Create("generated contract PCM", 320, 1, 16000, false);
            var store = new MediaPreviewStore();
            try
            {
                store.SetImage("image", first); store.SetAudio("sound", audio);
                Assert.That(store.Count, Is.EqualTo(2)); Assert.That(store.Image("image"), Is.SameAs(first)); Assert.That(store.Audio("sound"), Is.SameAs(audio));
                store.SetImage("image", second); Assert.That(first == null, Is.True);
                Assert.That(store.Image("image"), Is.SameAs(second)); store.Dispose(); store.Dispose();
                Assert.That(second == null && audio == null, Is.True);
                Assert.Throws<ObjectDisposedException>(() => store.SetImage("late", second));
            }
            finally { store.Dispose(); if (first != null) UnityEngine.Object.DestroyImmediate(first); if (second != null) UnityEngine.Object.DestroyImmediate(second); if (audio != null) UnityEngine.Object.DestroyImmediate(audio); }
        }

        [Test] public void InvalidPreviewDoesNotTakeOwnershipOrReplaceAnExistingOwner()
        {
            using var store = new MediaPreviewStore(); var image = new Texture2D(2, 2);
            try
            {
                Assert.Throws<ArgumentException>(() => store.SetImage("", image)); Assert.That(image != null, Is.True);
                store.SetImage("valid", image); Assert.Throws<ArgumentNullException>(() => store.SetImage("valid", null));
                Assert.That(store.Image("valid"), Is.SameAs(image)); Assert.That(store.Audio("missing"), Is.Null);
            }
            finally { store.Dispose(); if (image != null) UnityEngine.Object.DestroyImmediate(image); }
        }

        [Test] public void RecordingStartsOnlyExplicitlyAndStopsWithOwnedMono16kPcm()
        {
            var device = new Capture(); using var capture = new MediaCaptureSession(device);
            Assert.That(device.Starts, Is.Zero); Assert.That(capture.IsRecording, Is.False);
            capture.Start(); Assert.That(capture.IsRecording, Is.True); Assert.That(device.Starts, Is.EqualTo(1));
            Assert.That(device.Rate, Is.EqualTo(16000)); Assert.That(device.Maximum, Is.EqualTo(179200));
            var pcm = capture.Stop(); Assert.That(pcm.SourceSampleCount, Is.EqualTo(179200)); Assert.That(pcm.Waveform[0], Is.EqualTo(.25f));
            Assert.That(capture.IsRecording, Is.False); Assert.That(device.Stops, Is.EqualTo(1));
            Assert.Throws<InvalidOperationException>(() => capture.Stop());
        }

        [Test] public void PermissionDeniedCannotStartAndCanRecoverAfterPermissionChanges()
        {
            var device = new Capture { PermissionGranted = false }; using var capture = new MediaCaptureSession(device);
            Assert.Throws<UnauthorizedAccessException>(() => capture.Start()); Assert.That(device.Starts, Is.Zero);
            device.PermissionGranted = true; capture.Start(); Assert.That(capture.IsRecording, Is.True);
        }

        [Test] public void MissingDeviceCannotStartAndDeviceArrivalCanRecover()
        {
            var device = new Capture { Devices = Array.Empty<string>() }; using var capture = new MediaCaptureSession(device);
            Assert.Throws<InvalidOperationException>(() => capture.Start()); Assert.That(device.Starts, Is.Zero);
            device.Devices = new[] { "new input" }; capture.Start(); Assert.That(device.Starts, Is.EqualTo(1));
        }

        [Test] public void OverlapAndInvalidLimitsAreRejectedBeforeTouchingTheDevice()
        {
            var device = new Capture(); using var capture = new MediaCaptureSession(device);
            Assert.Throws<ArgumentOutOfRangeException>(() => capture.Start(0)); Assert.Throws<ArgumentOutOfRangeException>(() => capture.Start(179201));
            capture.Start(134400); Assert.Throws<InvalidOperationException>(() => capture.Start());
            Assert.That(device.Starts, Is.EqualTo(1)); Assert.That(device.Maximum, Is.EqualTo(134400));
        }

        [Test] public void ExplicitCaptureLimitProducesMatchingProfileWithoutReadingPastTheLimit()
        {
            var device = new Capture { CapturedSamples = 140000 }; using var capture = new MediaCaptureSession(device);
            capture.Start(134400); Assert.That(capture.ReachedLimit, Is.True);
            var pcm = capture.Stop(); Assert.That(pcm.SourceSampleCount, Is.EqualTo(134400)); Assert.That(pcm.ValidFrameCount, Is.EqualTo(839));
            Assert.That(device.Requested, Is.EqualTo(134400));
        }

        [Test] public void EarlyStopNeverSilentlyPadsTheRecordedDuration()
        {
            var device = new Capture { CapturedSamples = 3200 }; using var capture = new MediaCaptureSession(device);
            capture.Start(); Assert.That(capture.ReachedLimit, Is.False); var pcm = capture.Stop();
            Assert.That(pcm.SourceSampleCount, Is.EqualTo(3200)); Assert.That(device.Requested, Is.EqualTo(3200));
        }

        [Test] public void DeviceStartFailureCleansResourcesAndCanRetry()
        {
            var device = new Capture { StartError = new InvalidOperationException("device busy") }; using var capture = new MediaCaptureSession(device);
            Assert.Throws<InvalidOperationException>(() => capture.Start()); Assert.That(capture.IsRecording, Is.False); Assert.That(device.Cancels, Is.EqualTo(1));
            device.StartError = null; capture.Start(); Assert.That(capture.IsRecording, Is.True);
        }

        [Test] public void DeviceReadFailureStopsAndCleansResourcesBeforeRetry()
        {
            var device = new Capture { StopError = new InvalidOperationException("device removed") }; using var capture = new MediaCaptureSession(device);
            capture.Start(); Assert.Throws<InvalidOperationException>(() => capture.Stop()); Assert.That(capture.IsRecording, Is.False); Assert.That(device.Cancels, Is.EqualTo(1));
            device.StopError = null; capture.Start(); Assert.That(capture.Stop(), Is.Not.Null);
        }

        [Test] public void CancelAndDisposeNeverReturnOrPersistRecordedDataAndAreIdempotent()
        {
            var device = new Capture(); var capture = new MediaCaptureSession(device); capture.Start(); capture.Cancel(); capture.Cancel();
            Assert.That(capture.IsRecording, Is.False); Assert.That(device.Cancels, Is.EqualTo(1)); Assert.That(device.Stops, Is.Zero);
            capture.Start(); capture.Dispose(); capture.Dispose(); Assert.That(device.Cancels, Is.EqualTo(2));
            Assert.Throws<ObjectDisposedException>(() => capture.Start());
        }

        [Test] public void EmptyCaptureIsRejectedAndDeviceIsReleased()
        {
            var device = new Capture { CapturedSamples = 0 }; using var capture = new MediaCaptureSession(device); capture.Start();
            Assert.Throws<ArgumentException>(() => capture.Stop()); Assert.That(capture.IsRecording, Is.False); Assert.That(device.Cancels, Is.EqualTo(1));
        }
    }
}
