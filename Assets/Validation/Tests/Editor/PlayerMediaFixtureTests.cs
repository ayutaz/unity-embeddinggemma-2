using System;
using System.IO;
using System.Linq;
using Newtonsoft.Json.Linq;
using NUnit.Framework;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class PlayerMediaFixtureTests
    {
        string root;
        const string EmptySha = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
        [SetUp] public void Setup() { root = Path.Combine(Path.GetTempPath(), "eg-media-fixture-" + Guid.NewGuid().ToString("N")); Directory.CreateDirectory(root); File.WriteAllBytes(Path.Combine(root, "empty.bin"), Array.Empty<byte>()); }
        [TearDown] public void Cleanup() { Directory.Delete(root, true); }
        [Test] public void AVerifiedRelativeFileCanBeOpened()
            => Assert.That(PlayerMediaFixture.VerifyFile(root, "empty.bin", 0, EmptySha), Is.EqualTo(Path.Combine(root, "empty.bin")));
        [TestCase("../empty.bin")] [TestCase("/empty.bin")] [TestCase("a\\empty.bin")] [TestCase("a//empty.bin")]
        public void NonCanonicalPathsAreRejected(string name)
            => Assert.Throws<ArgumentException>(() => PlayerMediaFixture.VerifyFile(root, name, 0, EmptySha));
        [TestCase(true)] [TestCase(false)] public void SizeAndDigestMustBothMatch(bool changeSize)
            => Assert.Throws<InvalidDataException>(() => PlayerMediaFixture.VerifyFile(root, "empty.bin", changeSize ? 1 : 0, changeSize ? EmptySha : new string('a', 64)));
        static JObject Reference(bool audio = false)
        {
            var imageIds = new[] { "square-0", "square-1", "landscape-0", "landscape-1", "portrait-0", "portrait-1" };
            var audioIds = new[] { "silence", "impulse", "sine", "nyquist", "noise", "padded_noise" };
            var vector = new float[768]; vector[0] = 1;
            return new JObject { ["cases"] = new JArray((audio ? audioIds : imageIds).Select(id => new JObject {
                ["id"] = id, ["role"] = "document", ["profile"] = audio ? "frames-1119-valid-" + (id == "padded_noise" ? "839" : "1119") : id.Split('-')[0],
                ["embedding"] = new JArray(vector) })) };
        }
        [TestCase("image-square", 2)] [TestCase("image-landscape", 2)] [TestCase("image-portrait", 2)]
        [TestCase("audio-1119", 5)] [TestCase("audio-839", 1)]
        public void EveryFixedProfileSelectsItsCompleteDocumentCases(string module, int count)
            => Assert.That(PlayerMediaFixture.SelectCases(Reference(module.StartsWith("audio")), module).Length, Is.EqualTo(count));
        [TestCase("missing")] [TestCase("duplicate")] [TestCase("role")] [TestCase("vector")]
        public void IncompleteOrChangedReferenceRowsCannotReduceCoverage(string problem)
        {
            var reference = Reference(); var rows = (JArray)reference["cases"];
            if (problem == "missing") rows.RemoveAt(0);
            if (problem == "duplicate") rows[0] = rows[1].DeepClone();
            if (problem == "role") rows[0]["role"] = "raw";
            if (problem == "vector") rows[0]["embedding"] = new JArray(0);
            Assert.Throws<ArgumentException>(() => PlayerMediaFixture.SelectCases(reference, "image-square"));
        }
        [Test] public void UnknownModuleCannotSelectAnEmptySuccess()
            => Assert.Throws<ArgumentException>(() => PlayerMediaFixture.SelectCases(Reference(), "image-other"));
        [Test] public void UnpinnedFilesCannotBeLoadedAsOriginalFixtures()
        {
            File.WriteAllText(Path.Combine(root, "image-embedding-reference.json"), "{}");
            Assert.Throws<InvalidDataException>(() => PlayerMediaFixture.Load(root, "image-square", new JObject { ["kind"] = "image" }));
        }
    }
}
