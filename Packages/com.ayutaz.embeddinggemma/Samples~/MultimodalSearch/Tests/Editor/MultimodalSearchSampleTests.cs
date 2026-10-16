using System;
using System.Collections;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;

namespace EmbeddingGemma.Samples.Tests
{
    public sealed class MultimodalSearchSampleTests
    {
        GameObject host; string directory;
        [SetUp] public void Setup()
        { host = new GameObject("media sample contracts"); directory = Path.Combine(Path.GetTempPath(), "embeddinggemma-media-input-" + Guid.NewGuid().ToString("N")); Directory.CreateDirectory(directory); }
        [TearDown] public void Cleanup()
        { if (host != null) UnityEngine.Object.DestroyImmediate(host); if (Directory.Exists(directory)) Directory.Delete(directory, true); }
        static Type Required(string name)
        {
            var type = AppDomain.CurrentDomain.GetAssemblies().Select(assembly => assembly.GetType("EmbeddingGemma.Samples." + name)).FirstOrDefault(value => value != null);
            Assert.That(type, Is.Not.Null, "An importable media sample must connect real inputs, cache and GPU sessions."); return type;
        }
        Component Sample()
        {
            var value = host.AddComponent(Required("MultimodalSearchSample"));
            value.GetType().GetField("SourceDirectory").SetValue(value, Path.Combine(directory, "no-models"));
            value.GetType().GetField("CacheDirectory").SetValue(value, Path.Combine(directory, "cache")); return value;
        }
        static Task Call(Component sample, string method, CancellationToken token = default)
            => (Task)sample.GetType().GetMethod(method).Invoke(sample, new object[] { token });
        static IEnumerator Wait(Task task) { while (!task.IsCompleted) yield return null; }
        [Test] public void StartsIdleWithAnEmptyCorpusAndNoModelWork()
        {
            var sample = Sample(); Assert.That(sample.GetType().GetProperty("IsBusy").GetValue(sample), Is.False);
            Assert.That(sample.GetType().GetProperty("DocumentCount").GetValue(sample), Is.EqualTo(0)); Assert.That(Directory.Exists(Path.Combine(directory, "cache")), Is.False);
        }
        [UnityTest] public IEnumerator EmptyCorpusSearchFailsBeforeLoadingAnyModel()
        {
            var sample = Sample(); var task = Call(sample, "SearchAsync"); yield return Wait(task);
            Assert.That(task.Exception?.InnerException, Is.InstanceOf<InvalidOperationException>());
            Assert.That(Directory.Exists(Path.Combine(directory, "cache")), Is.False);
        }
        [UnityTest] public IEnumerator MissingInputFailsBeforeLoadingAnyModelAndCanBeRetried()
        {
            var sample = Sample(); sample.GetType().GetField("MediaPath").SetValue(sample, Path.Combine(directory, "missing.png"));
            var first = Call(sample, "AddAsync"); yield return Wait(first); Assert.That(first.Exception?.InnerException, Is.InstanceOf<IOException>());
            var second = Call(sample, "AddAsync"); yield return Wait(second); Assert.That(second.Exception?.InnerException, Is.InstanceOf<IOException>());
            Assert.That(sample.GetType().GetProperty("IsBusy").GetValue(sample), Is.False);
            Assert.That(Directory.Exists(Path.Combine(directory, "cache")), Is.False);
        }
        [UnityTest] public IEnumerator PreCancelledOperationCannotPopulateCorpusOrStartCacheWork()
        {
            var sample = Sample(); using var cancellation = new CancellationTokenSource(); cancellation.Cancel();
            var task = Call(sample, "AddAsync", cancellation.Token); yield return Wait(task); Assert.That(task.IsCanceled, Is.True);
            Assert.That(sample.GetType().GetProperty("DocumentCount").GetValue(sample), Is.EqualTo(0));
            Assert.That(Directory.Exists(Path.Combine(directory, "cache")), Is.False);
        }
        AudioPcmInput Read(string path)
            => (AudioPcmInput)Required("MediaSampleInput").GetMethod("ReadPcm").Invoke(null, new object[] { path });
        string Wave(int format, int bits, int rate = 16000, int channels = 1)
        {
            var path = Path.Combine(directory, "input.wav"); using var file = File.Create(path); using var writer = new BinaryWriter(file);
            var bytes = 256 * channels * bits / 8;
            writer.Write(System.Text.Encoding.ASCII.GetBytes("RIFF")); writer.Write(36 + bytes); writer.Write(System.Text.Encoding.ASCII.GetBytes("WAVEfmt "));
            writer.Write(16); writer.Write((short)format); writer.Write((short)channels); writer.Write(rate); writer.Write(rate * channels * bits / 8);
            writer.Write((short)(channels * bits / 8)); writer.Write((short)bits); writer.Write(System.Text.Encoding.ASCII.GetBytes("data")); writer.Write(bytes);
            for (var index = 0; index < 256 * channels; index++) { if (bits == 32) writer.Write(index % 2 == 0 ? .25f : -.5f); else writer.Write((short)(index % 2 == 0 ? 8192 : -16384)); }
            return path;
        }
        [TestCase(3, 32)] [TestCase(1, 16)] public void WavInputPreservesMonoSamplesWithoutResamplingOrClipping(int format, int bits)
        {
            var input = Read(Wave(format, bits)); Assert.That(input.SourceSampleCount, Is.EqualTo(256));
            Assert.That(input.Waveform[0], Is.EqualTo(.25f)); Assert.That(input.Waveform[1], Is.EqualTo(-.5f));
        }
        [Test] public void RawFp32InputRetainsFloatBitsAndValidPrefix()
        {
            var path = Path.Combine(directory, "input.f32"); var values = Enumerable.Repeat(.12345679f, 179200).ToArray();
            var bytes = new byte[values.Length * 4]; Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length); File.WriteAllBytes(path, bytes);
            var input = Read(path); Assert.That(input.SourceSampleCount, Is.EqualTo(179200));
            Assert.That(BitConverter.SingleToInt32Bits(input.Waveform[0]), Is.EqualTo(BitConverter.SingleToInt32Bits(values[0])));
        }
        [TestCase(44100, 1)] [TestCase(16000, 2)] public void UnsupportedAudioGeometryIsRejected(int rate, int channels)
        {
            Required("MediaSampleInput"); Assert.That(Assert.Throws<TargetInvocationException>(() => Read(Wave(3, 32, rate, channels))).InnerException, Is.InstanceOf<ArgumentException>());
        }
        [Test] public void TruncatedWavIsRejectedInsteadOfInventingSamples()
        {
            Required("MediaSampleInput"); var path = Wave(3, 32); using (var file = new FileStream(path, FileMode.Open, FileAccess.Write)) file.SetLength(file.Length - 1);
            Assert.That(Assert.Throws<TargetInvocationException>(() => Read(path)).InnerException, Is.InstanceOf<InvalidDataException>());
        }
    }
}
