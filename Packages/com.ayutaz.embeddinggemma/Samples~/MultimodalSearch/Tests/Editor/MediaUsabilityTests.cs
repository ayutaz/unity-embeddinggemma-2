using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Threading.Tasks;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;

namespace EmbeddingGemma.Samples.Tests
{
    public sealed class MediaUsabilityTests
    {
        sealed class Capture : IMediaCaptureDevice
        {
            public bool PermissionGranted { get; set; } = true;
            public string[] Devices => new[] { "contract input" };
            public int CapturedSamples { get; set; } = 179200;
            public int Starts, Stops, Cancels;
            public void Start(string device, int frequency, int maximum) { Starts++; }
            public float[] Stop(int count) { Stops++; var values = new float[count]; values[0] = .25f; return values; }
            public void Cancel() { Cancels++; }
        }
        GameObject host; MultimodalSearchSample sample; Capture input; string cache;
        [SetUp] public void Setup()
        {
            host = new GameObject("media usability contracts"); sample = host.AddComponent<MultimodalSearchSample>(); input = new Capture();
            cache = Path.Combine(Path.GetTempPath(), "media-no-implicit-work-" + Guid.NewGuid().ToString("N"));
            sample.CacheDirectory = cache; sample.SourceDirectory = cache + "-missing-source";
            sample.SetCaptureDevice(input);
        }
        [TearDown] public void Cleanup() { if (host != null) UnityEngine.Object.DestroyImmediate(host); Assert.That(Directory.Exists(cache), Is.False); }
        static IEnumerator Wait(Task task) { while (!task.IsCompleted) yield return null; }

        [Test] public void CaptureRequiresExplicitStartAndStopCreatesAPlayableOwnedClipWithoutInference()
        {
            Assert.That(input.Starts, Is.Zero); Assert.That(sample.IsRecording, Is.False); Assert.That(sample.CapturedPcm, Is.Null);
            sample.StartRecording(); Assert.That(sample.IsRecording, Is.True); Assert.That(input.Starts, Is.EqualTo(1));
            sample.StopRecording(); Assert.That(sample.IsRecording, Is.False); Assert.That(sample.CapturedPcm.SourceSampleCount, Is.EqualTo(179200));
            Assert.That(sample.RecordingClip.samples, Is.EqualTo(179200)); Assert.That(sample.RecordingClip.frequency, Is.EqualTo(16000));
            Assert.That(sample.RecordingClip.channels, Is.EqualTo(1)); Assert.That(sample.PreviewCount, Is.EqualTo(1)); Assert.That(sample.DocumentCount, Is.Zero);
        }

        [Test] public void StartingAgainReleasesThePreviousCapturedClipAndDoesNotOverlapRecordings()
        {
            sample.StartRecording(); Assert.Throws<InvalidOperationException>(() => sample.StartRecording());
            sample.StopRecording(); var previous = sample.RecordingClip; sample.StartRecording();
            Assert.That(previous == null, Is.True); Assert.That(sample.CapturedPcm, Is.Null);
            sample.CancelRecording(); Assert.That(sample.IsRecording, Is.False); Assert.That(input.Stops, Is.EqualTo(1));
        }

        [Test] public void PermissionFailureIsVisibleAndCanRecoverWithoutImplicitWork()
        {
            input.PermissionGranted = false; Assert.Throws<UnauthorizedAccessException>(() => sample.StartRecording());
            Assert.That(sample.LastError, Is.Not.Null.And.Not.Empty); Assert.That(input.Starts, Is.Zero);
            input.PermissionGranted = true; sample.StartRecording(); Assert.That(sample.IsRecording, Is.True); Assert.That(sample.LastError, Is.Null);
        }

        [UnityTest] public IEnumerator EarlyStopKeepsActualDurationAndUnsupportedProfileFailsBeforeCacheOrModelWork()
        {
            input.CapturedSamples = 3200; sample.StartRecording(); sample.StopRecording();
            Assert.That(sample.CapturedPcm.SourceSampleCount, Is.EqualTo(3200)); Assert.That(sample.RecordingClip.samples, Is.EqualTo(3200));
            var task = sample.AddRecordingAsync(); yield return Wait(task);
            Assert.That(task.Exception?.InnerException, Is.InstanceOf<ArgumentException>());
            Assert.That(sample.DocumentCount, Is.Zero); Assert.That(sample.IsBusy, Is.False);
        }

        [UnityTest] public IEnumerator DisableCancelsRecordingAndReleasesCapturedAudioWithoutSavingOrUploading()
        {
            sample.StartRecording(); host.SetActive(false); var task = sample.CloseAsync(); yield return Wait(task); task.GetAwaiter().GetResult();
            Assert.That(input.Cancels, Is.EqualTo(1)); Assert.That(input.Stops, Is.Zero);
            Assert.That(sample.IsRecording, Is.False); Assert.That(sample.CapturedPcm, Is.Null); Assert.That(sample.PreviewCount, Is.Zero);
        }

        [UnityTest] public IEnumerator ExplicitResetClearsCorpusAndAudioAndAllowsRecordingAgain()
        {
            sample.StartRecording(); sample.StopRecording(); var clip = sample.RecordingClip;
            var documents = (List<SearchDocument>)typeof(MultimodalSearchSample).GetField("documents", BindingFlags.NonPublic | BindingFlags.Instance).GetValue(sample);
            documents.Add(new SearchDocument("state-contract-only", "display metadata", "state fixture"));
            var reset = sample.ResetAsync(); yield return Wait(reset); reset.GetAwaiter().GetResult();
            Assert.That(sample.DocumentCount, Is.Zero); Assert.That(sample.Hits, Is.Empty); Assert.That(sample.PreviewCount, Is.Zero);
            Assert.That(sample.CapturedPcm, Is.Null); Assert.That(clip == null, Is.True);
            sample.StartRecording(); Assert.That(sample.IsRecording, Is.True);
        }

        [UnityTest] public IEnumerator ClosedSampleCannotStartADeviceUntilReenabledOrReset()
        {
            var close = sample.CloseAsync(); yield return Wait(close); close.GetAwaiter().GetResult();
            Assert.Throws<InvalidOperationException>(() => sample.StartRecording()); Assert.That(input.Starts, Is.Zero);
        }
    }
}
