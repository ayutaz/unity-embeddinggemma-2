using System;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using NUnit.Framework;

namespace EmbeddingGemma.Samples.Tests
{
    public sealed class TextSearchStreamHashTests
    {
        [TestCase("", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")]
        [TestCase("abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")]
        public void KnownVectorsUseTheActualPlatformImplementation(string text, string expected)
        {
            using var stream = new MemoryStream(Encoding.UTF8.GetBytes(text));
            Assert.That(TextSearchStreamHash.Compute(stream, out var backend), Is.EqualTo(expected));
#if UNITY_EDITOR_WIN
            Assert.That(backend, Is.EqualTo("windows_cng"), "This Windows Editor must exercise the real CNG implementation.");
#else
            Assert.That(backend, Is.EqualTo("dotnet_sha256"));
#endif
            Assert.That(stream.CanRead, Is.True, "The caller retains the audited handle for subsequent JSON reads.");
        }
        [Test] public void MultiChunkInputMatchesIndependentManagedSha256()
        {
            var bytes = new byte[5 * 1024 * 1024 + 17]; for (var index = 0; index < bytes.Length; index++) bytes[index] = (byte)(index % 251);
            using var reference = SHA256.Create(); var expected = BitConverter.ToString(reference.ComputeHash(bytes)).Replace("-", "").ToLowerInvariant();
            using var stream = new MemoryStream(bytes); stream.Position = stream.Length - 1;
            Assert.That(TextSearchStreamHash.Compute(stream, out _), Is.EqualTo(expected), "Always hash all bytes, including the prefix and final partial block.");
        }
        [Test] public void UnavailableNativeAfterConsumptionFallsBackToAllBytesOnTheSameHandle()
        {
            using var stream = new MemoryStream(Encoding.UTF8.GetBytes("abc"));
            var digest = TextSearchStreamHash.Compute(stream, out var backend, value => { value.Position = value.Length; throw new DllNotFoundException("unavailable"); });
            Assert.That(digest, Is.EqualTo("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"));
            Assert.That(backend, Is.EqualTo("dotnet_sha256")); Assert.That(stream.CanRead, Is.True);
        }
        [Test] public void InvalidNativeOutputCannotReplaceACompleteDigest()
        {
            foreach (var invalid in new[] { "garbage", new string('g', 64), null })
            {
                using var stream = new MemoryStream(Encoding.UTF8.GetBytes("abc"));
                var digest = TextSearchStreamHash.Compute(stream, out var backend, value => { value.Position = value.Length; return invalid; });
                Assert.That(digest, Is.EqualTo("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"));
                Assert.That(backend, Is.EqualTo("dotnet_sha256"));
            }
        }
        [Test] public void IoFailureCannotBeReportedAsAValidHash()
        {
            using var stream = new MemoryStream(new byte[] { 1 });
            Assert.Throws<IOException>(() => TextSearchStreamHash.Compute(stream, out _, _ => throw new IOException("read failure")));
        }
        [Test] public void UnreadableInputIsRejected()
        {
            using var stream = new MemoryStream(new byte[] { 1 }); stream.Dispose();
            Assert.Throws<ArgumentException>(() => TextSearchStreamHash.Compute(stream, out _));
        }
    }
}
