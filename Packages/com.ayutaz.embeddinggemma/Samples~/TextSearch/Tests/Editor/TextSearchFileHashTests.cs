using System;
using System.IO;
using System.Security.Cryptography;
using NUnit.Framework;

namespace EmbeddingGemma.Samples.Tests
{
    public sealed class TextSearchFileHashTests
    {
        const string AbcHash = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
        string directory, path;

        [SetUp] public void Setup()
        {
            directory = Path.Combine(Path.GetTempPath(), "embeddinggemma-hash-" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(directory);
            path = Path.Combine(directory, "models; test & 日本語.bin");
            File.WriteAllText(path, "abc");
        }
        [TearDown] public void Cleanup() { Directory.Delete(directory, true); }

        [Test] public void HashesTheWholeFileAndReturnsCanonicalSha256()
            => Assert.That(TextSearchFileHash.Compute(path), Is.EqualTo(AbcHash));

        [Test] public void AcceleratedLargeFileMatchesManagedSha256WithUnicodeAndShellCharacters()
        {
            var bytes = new byte[2 * 1024 * 1024];
            for (var i = 0; i < bytes.Length; i++) bytes[i] = (byte)(i % 251);
            File.WriteAllBytes(path, bytes);
            using var algorithm = SHA256.Create();
            var expected = BitConverter.ToString(algorithm.ComputeHash(bytes)).Replace("-", "").ToLowerInvariant();
            Assert.That(TextSearchFileHash.Compute(path), Is.EqualTo(expected));
        }

        [TestCase(null)] [TestCase("")] [TestCase("partial hash or error")] [TestCase("d41d8cd98f00b204e9800998ecf8427e")]
        public void UnavailableOrMalformedNativeOutputFallsBackToFullHash(string output)
            => Assert.That(TextSearchFileHash.Compute(path, _ => output), Is.EqualTo(AbcHash));

        [Test] public void NativeProcessFailureFallsBackToFullHash()
            => Assert.That(TextSearchFileHash.Compute(path, _ => throw new IOException("unavailable")), Is.EqualTo(AbcHash));

        [Test] public void MissingFileCannotBeAcceptedByAnAccelerator()
        {
            var calls = 0;
            Assert.Throws<FileNotFoundException>(() => TextSearchFileHash.Compute(Path.Combine(directory, "missing"), _ => { calls++; return AbcHash; }));
            Assert.That(calls, Is.Zero);
        }

        [Test] public void LocalizedHeaderAndUppercaseDigestAreAcceptedWithoutUsingTheHeader()
        {
            var raw = "SHA256 ハッシュ:\r\n" + AbcHash.ToUpperInvariant() + "\r\ncommand completed\r\n";
            Assert.That(TextSearchFileHash.ParseNativeOutput(raw), Is.EqualTo(AbcHash));
            var observed = "";
            Assert.That(TextSearchFileHash.Compute(path, fullPath => { observed = fullPath; return raw; }), Is.EqualTo(AbcHash));
            Assert.That(observed, Is.EqualTo(Path.GetFullPath(path)));
        }

        [Test] public void AmbiguousNativeOutputMustBeRejectedAndRehashed()
        {
            var raw = AbcHash + "\n" + new string('0', 64);
            Assert.That(TextSearchFileHash.ParseNativeOutput(raw), Is.Null);
            Assert.That(TextSearchFileHash.Compute(path, _ => raw), Is.EqualTo(AbcHash));
        }
    }
}
