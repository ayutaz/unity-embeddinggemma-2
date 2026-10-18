using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using NUnit.Framework;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class PlayerMixedConfigurationTests
    {
        const string Prefix = "--embeddinggemma-mixed-";
        static Dictionary<string, string> Options(string precision = "fp32", string phase = "cold")
        {
            var root = Path.Combine(Path.GetTempPath(), "embeddinggemma-mixed-options-" + Guid.NewGuid().ToString("N"));
            return new Dictionary<string, string> {
                ["bundle"] = Path.Combine(root, "models"), ["fixtures"] = Path.Combine(root, "fixtures"),
                ["cache"] = Path.Combine(root, "cache"), ["output"] = Path.Combine(root, "results", "result.json"),
                ["run-id"] = new string('a', 32), ["phase"] = phase, ["precision"] = precision,
                ["descriptor"] = precision == "fp32" ? "40aefa5de06f575d31b7582c24314c1ab035da374116b3ce3036de0e919cd647"
                    : "8a6a90c5f656a00b024553a0412e5c4d32b4d2c9c7c5b3a0c2c146f557b4e27e" };
        }
        static object Parse(Dictionary<string, string> options, params string[] extra)
        {
            var type = AppDomain.CurrentDomain.GetAssemblies().Select(a => a.GetType("EmbeddingGemma.Validation.PlayerMixedConfiguration")).FirstOrDefault(t => t != null);
            Assert.That(type, Is.Not.Null, "The complete Player needs strict cold/warm configuration.");
            var args = new[] { "Player.exe", "-force-d3d12" }.Concat(options.SelectMany(p => new[] { Prefix + p.Key, p.Value })).Concat(extra).ToArray();
            return type.GetMethod("Parse").Invoke(null, new object[] { args });
        }
        static string Value(object config, string property) => (string)config.GetType().GetProperty(property).GetValue(config);
        static void Reject(Action action) => Assert.That(Assert.Throws<TargetInvocationException>(() => action()).InnerException, Is.InstanceOf<ArgumentException>());
        [TestCase("fp32", "cold")] [TestCase("fp32", "warm")] [TestCase("float16", "cold")] [TestCase("float16", "warm")]
        public void ExplicitProfilePinsAndIndependentProcessPhaseAreRetainedWithoutFileWrites(string precision, string phase)
        {
            var options = Options(precision, phase); var value = Parse(options);
            Assert.That(Value(value, "Precision"), Is.EqualTo(precision)); Assert.That(Value(value, "Phase"), Is.EqualTo(phase));
            Assert.That(Value(value, "DescriptorSha256"), Is.EqualTo(options["descriptor"])); Assert.That(Value(value, "RunId"), Is.EqualTo(options["run-id"]));
            foreach (var pair in new[] { ("Bundle", "bundle"), ("Fixtures", "fixtures"), ("Cache", "cache"), ("Output", "output") })
                Assert.That(Value(value, pair.Item1), Is.EqualTo(Path.GetFullPath(options[pair.Item2])));
            Assert.That(Directory.Exists(Path.GetDirectoryName(options["bundle"])), Is.False);
        }
        [TestCase("bundle")] [TestCase("fixtures")] [TestCase("cache")] [TestCase("output")]
        [TestCase("run-id")] [TestCase("phase")] [TestCase("precision")] [TestCase("descriptor")]
        public void EveryValidationArgumentIsMandatory(string key)
        { var options = Options(); options.Remove(key); Reject(() => Parse(options)); }
        [TestCase("duplicate")] [TestCase("unknown")] [TestCase("missing_value")] [TestCase("legacy_arguments")]
        [TestCase("wrong_pin")] [TestCase("wrong_profile_pin")] [TestCase("bad_run_id")] [TestCase("bad_phase")] [TestCase("blank_path")]
        public void AmbiguousInvalidOrUnreviewedInvocationsAreRejected(string problem)
        {
            var options = Options(); var extra = Array.Empty<string>();
            if (problem == "duplicate") extra = new[] { Prefix + "phase", "warm" };
            if (problem == "unknown") extra = new[] { Prefix + "allow-cpu", "true" };
            if (problem == "missing_value") extra = new[] { Prefix + "extra" };
            if (problem == "legacy_arguments") extra = new[] { "--embeddinggemma-media-module", "image-square" };
            if (problem == "wrong_pin") options["descriptor"] = new string('b', 64);
            if (problem == "wrong_profile_pin") options["precision"] = "float16";
            if (problem == "bad_run_id") options["run-id"] = "previous";
            if (problem == "bad_phase") options["phase"] = "auto";
            if (problem == "blank_path") options["cache"] = " ";
            Reject(() => Parse(options, extra));
        }
        [TestCase("cache_in_bundle")] [TestCase("cache_in_fixtures")] [TestCase("bundle_in_cache")]
        [TestCase("output_in_bundle")] [TestCase("output_in_fixtures")] [TestCase("output_in_cache")]
        public void WritableOutputAndCacheCannotMutateReadOnlyInputsOrEachOther(string overlap)
        {
            var options = Options();
            if (overlap == "cache_in_bundle") options["cache"] = Path.Combine(options["bundle"], "cache");
            if (overlap == "cache_in_fixtures") options["cache"] = Path.Combine(options["fixtures"], "cache");
            if (overlap == "bundle_in_cache") options["bundle"] = Path.Combine(options["cache"], "models");
            if (overlap == "output_in_bundle") options["output"] = Path.Combine(options["bundle"], "result.json");
            if (overlap == "output_in_fixtures") options["output"] = Path.Combine(options["fixtures"], "result.json");
            if (overlap == "output_in_cache") options["output"] = Path.Combine(options["cache"], "result.json");
            Reject(() => Parse(options));
        }
    }
}
