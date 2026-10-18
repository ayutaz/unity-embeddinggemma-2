using System;
using System.Collections.Generic;
using System.IO;
using NUnit.Framework;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class PlayerMediaConfigurationTests
    {
        static string[] Args() => new[] { "player.exe", "--embeddinggemma-media-bundle", "bundle", "--embeddinggemma-media-fixtures", "fixtures",
            "--embeddinggemma-media-module", "audio-839", "--embeddinggemma-media-precision", "fp32", "--embeddinggemma-media-descriptor", new string('a', 64),
            "--embeddinggemma-media-output", "result.json", "--embeddinggemma-media-run-id", new string('b', 32) };
        [Test] public void ExplicitCandidateFixtureAndUniqueRunArePreserved()
        {
            var args = new List<string>(Args()); args.AddRange(new[] { "-logFile", "private.log" });
            var result = PlayerMediaConfiguration.Parse(args.ToArray());
            Assert.That(result.Bundle, Is.EqualTo(Path.GetFullPath("bundle")));
            Assert.That(result.Fixtures, Is.EqualTo(Path.GetFullPath("fixtures")));
            Assert.That(result.Module, Is.EqualTo("audio-839")); Assert.That(result.Precision, Is.EqualTo("fp32"));
            Assert.That(result.DescriptorSha256, Is.EqualTo(new string('a', 64))); Assert.That(result.RunId, Is.EqualTo(new string('b', 32)));
        }
        [TestCase("duplicate")] [TestCase("missing")] [TestCase("unknown")] [TestCase("descriptor")]
        [TestCase("precision")] [TestCase("module")] [TestCase("run_id")]
        public void InvalidOrImplicitInputsCannotStartARealModelRun(string problem)
        {
            var args = new List<string>(Args());
            if (problem == "duplicate") args.AddRange(new[] { "--embeddinggemma-media-module", "image-square" });
            if (problem == "missing") args.RemoveRange(1, 2);
            if (problem == "unknown") args.AddRange(new[] { "--embeddinggemma-media-skip", "true" });
            if (problem == "descriptor") args[10] = "unfixed";
            if (problem == "precision") args[8] = "cpu";
            if (problem == "module") args[6] = "text";
            if (problem == "run_id") args[14] = "shared";
            Assert.Throws<ArgumentException>(() => PlayerMediaConfiguration.Parse(args.ToArray()));
        }
    }
}
