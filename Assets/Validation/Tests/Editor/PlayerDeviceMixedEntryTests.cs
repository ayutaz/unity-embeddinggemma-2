using System;
using System.Collections;
using System.IO;
using System.Linq;
using System.Reflection;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.Rendering;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class PlayerDeviceMixedEntryTests
    {
        static Type Required(string name)
        {
            var type = AppDomain.CurrentDomain.GetAssemblies().Select(a => a.GetType("EmbeddingGemma.Validation." + name)).FirstOrDefault(t => t != null);
            Assert.That(type, Is.Not.Null, "Device mixed execution needs explicit platform and non-CLI input entries."); return type;
        }
        static PlayerBuildInfo Build(string target) => new PlayerBuildInfo { codeCommit = new string('a', 40), unityVersion = "6000.3.21f1",
            sentisVersion = "2.6.1", target = target, scriptingBackend = "IL2CPP", stripping = "High",
            sourceSha256 = new JObject { ["Assets/Validation/Runtime/input.cs"] = new string('b', 64) } };
        static void Validate(PlayerBuildInfo build, RuntimePlatform platform, GraphicsDeviceType graphics, bool editor = false,
            bool development = false, bool compute = true, string unity = "6000.3.21f1")
            => Required("PlayerMixedRuntimePolicy").GetMethod("Validate").Invoke(null, new object[] { build, platform, graphics, editor, development, compute, unity });
        static JObject Config() => new JObject { ["schema_version"] = 1, ["precision"] = "fp32", ["phase"] = "cold",
            ["run_id"] = new string('a', 32), ["cache_id"] = new string('b', 32),
            ["bundle_descriptor_sha256"] = PlayerMixedConfiguration.Fp32Descriptor, ["fixture_manifest_sha256"] = new string('c', 64) };
        static object Parse(string json, string root) => Required("PlayerDeviceMixedConfiguration").GetMethod("Parse").Invoke(null, new object[] { json, root });
        static void Reject(Action action) => Assert.That(Assert.Throws<TargetInvocationException>(() => action()).InnerException,
            Is.InstanceOf<ArgumentException>().Or.InstanceOf<NotSupportedException>());
        [TestCase("StandaloneWindows64", RuntimePlatform.WindowsPlayer, GraphicsDeviceType.Direct3D12)]
        [TestCase("Android", RuntimePlatform.Android, GraphicsDeviceType.Vulkan)]
        [TestCase("iOS", RuntimePlatform.IPhonePlayer, GraphicsDeviceType.Metal)]
        public void MatchingReleasePlayerContextIsOnlyAdmissionNotExecutionEvidence(string target, RuntimePlatform platform, GraphicsDeviceType graphics)
        { Validate(Build(target), platform, graphics); }
        [TestCase("editor")] [TestCase("development")] [TestCase("no_compute")] [TestCase("unity_version")]
        [TestCase("target_mismatch")] [TestCase("mono")] [TestCase("stripping")] [TestCase("bad_source")]
        [TestCase("android_gles")] [TestCase("ios_vulkan")] [TestCase("unsupported_platform")] [TestCase("null_build")]
        public void UnsupportedOrUnprovenContextCannotAdmitGpuExecution(string problem)
        {
            var build = Build("Android"); var platform = RuntimePlatform.Android; var graphics = GraphicsDeviceType.Vulkan;
            if (problem == "target_mismatch") build.target = "iOS";
            if (problem == "mono") build.scriptingBackend = "Mono2x";
            if (problem == "stripping") build.stripping = "Low";
            if (problem == "bad_source") build.codeCommit = "unfixed";
            if (problem == "android_gles") graphics = GraphicsDeviceType.OpenGLES3;
            if (problem == "ios_vulkan") { build.target = "iOS"; platform = RuntimePlatform.IPhonePlayer; }
            if (problem == "unsupported_platform") { build.target = "StandaloneOSX"; platform = RuntimePlatform.OSXPlayer; graphics = GraphicsDeviceType.Metal; }
            Reject(() => Validate(problem == "null_build" ? null : build, platform, graphics, problem == "editor", problem == "development",
                problem != "no_compute", problem == "unity_version" ? "unknown" : "6000.3.21f1"));
        }
        [TestCase("fp32", "cold")] [TestCase("fp32", "warm")] [TestCase("float16", "cold")] [TestCase("float16", "warm")]
        public void LocalDeviceConfigurationCreatesIsolatedArgumentsWithoutWritingFiles(string precision, string phase)
        {
            var root = Path.Combine(Path.GetTempPath(), "embeddinggemma-device-config-" + Guid.NewGuid().ToString("N"));
            var json = Config(); json["precision"] = precision; json["phase"] = phase;
            json["bundle_descriptor_sha256"] = precision == "fp32" ? PlayerMixedConfiguration.Fp32Descriptor : PlayerMixedConfiguration.Float16Descriptor;
            var value = Parse(json.ToString(), root); var type = value.GetType();
            var args = (string[])type.GetMethod("MixedArguments").Invoke(value, null); var mixed = PlayerMixedConfiguration.Parse(args);
            Assert.That(mixed.Bundle, Is.EqualTo(Path.Combine(root, "EmbeddingGemmaDeviceValidation", "bundles", precision)));
            Assert.That(mixed.Fixtures, Is.EqualTo(Path.Combine(root, "EmbeddingGemmaDeviceValidation", "runs", new string('a', 32), "fixtures")));
            Assert.That(mixed.Cache, Is.EqualTo(Path.Combine(root, "EmbeddingGemmaDeviceValidation", "cache", new string('b', 32))));
            Assert.That(mixed.Output, Is.EqualTo(Path.Combine(root, "EmbeddingGemmaDeviceValidation", "reports", new string('a', 32) + ".json")));
            Assert.That(mixed.RunId, Is.EqualTo(new string('a', 32))); Assert.That(mixed.Precision, Is.EqualTo(precision)); Assert.That(mixed.Phase, Is.EqualTo(phase));
            Assert.That((string)type.GetProperty("FixtureManifestSha256").GetValue(value), Is.EqualTo(new string('c', 64)));
            Assert.That((string)type.GetProperty("StartupReport").GetValue(value), Is.Not.EqualTo(mixed.Output));
            Assert.That(Directory.Exists(root), Is.False);
        }
        [TestCase("schema_version")] [TestCase("precision")] [TestCase("phase")] [TestCase("run_id")]
        [TestCase("cache_id")] [TestCase("bundle_descriptor_sha256")] [TestCase("fixture_manifest_sha256")]
        public void EveryDeviceSettingIsExplicit(string key)
        { var json = Config(); json.Remove(key); Reject(() => Parse(json.ToString(), Path.GetTempPath())); }
        [TestCase("duplicate")] [TestCase("unknown")] [TestCase("bad_precision")] [TestCase("bad_phase")]
        [TestCase("run_path")] [TestCase("cache_path")] [TestCase("wrong_pin")] [TestCase("wrong_manifest")]
        [TestCase("wrong_schema_type")] [TestCase("non_string_run")] [TestCase("blank_root")] [TestCase("oversized")]
        public void AmbiguousUnpinnedOrPathBearingSettingsAreRejected(string problem)
        {
            var json = Config(); var root = Path.GetTempPath();
            if (problem == "unknown") json["allow_cpu"] = true;
            if (problem == "bad_precision") json["precision"] = "auto";
            if (problem == "bad_phase") json["phase"] = "auto";
            if (problem == "run_path") json["run_id"] = "../previous";
            if (problem == "cache_path") json["cache_id"] = "../models";
            if (problem == "wrong_pin") json["bundle_descriptor_sha256"] = new string('d', 64);
            if (problem == "wrong_manifest") json["fixture_manifest_sha256"] = "unknown";
            if (problem == "wrong_schema_type") json["schema_version"] = "1";
            if (problem == "non_string_run") json["run_id"] = 1;
            if (problem == "blank_root") root = " ";
            var text = problem == "duplicate" ? json.ToString().Insert(1, "\"phase\":\"warm\",") : json.ToString();
            if (problem == "oversized") text = new string(' ', 256 * 1024 + 1);
            Reject(() => Parse(text, root));
        }
        [Test] public void DedicatedDeviceRunnerExistsWithoutChangingTheOrdinaryPlayerBootstrap()
        { Assert.That(typeof(MonoBehaviour).IsAssignableFrom(Required("PlayerDeviceMixedRunner")), Is.True); }
        [Test] public void EditorDeviceEntryFailsBeforeReadingConfigurationOrMutatingData()
        {
            var type = Required("PlayerDeviceMixedRunner"); var report = new JObject(); var root = Path.Combine(Path.GetTempPath(), "embeddinggemma-device-reject-" + Guid.NewGuid().ToString("N"));
            var calls = 0; var succeeded = true;
            var routine = (IEnumerator)type.GetMethod("Execute").Invoke(null, new object[] { Build("Android"), root, "file:///fixtures", report,
                new Action<bool>(value => { calls++; succeeded = value; }) });
            while (routine.MoveNext()) { }
            Assert.That(calls, Is.EqualTo(1)); Assert.That(succeeded, Is.False); Assert.That((bool)report["success"], Is.False);
            Assert.That((bool)report["gpu_verified"], Is.False); Assert.That((string)report["error_type"], Is.EqualTo("NotSupportedException"));
            Assert.That(Directory.Exists(root), Is.False); Assert.That(report.ToString(), Does.Not.Contain(root));
        }
        [TestCase("null_report")] [TestCase("used_report")] [TestCase("missing_completion")]
        public void InvalidEntryContractsCannotStartInputWork(string problem)
        {
            var type = Required("PlayerDeviceMixedRunner"); var report = problem == "null_report" ? null : problem == "used_report" ? new JObject { ["success"] = true } : new JObject();
            var routine = (IEnumerator)type.GetMethod("Execute").Invoke(null, new object[] { Build("Android"), Path.GetTempPath(), "file:///fixtures", report,
                problem == "missing_completion" ? null : new Action<bool>(_ => { }) });
            Assert.Throws<ArgumentException>(() => routine.MoveNext());
        }
        static Type Builder() => AppDomain.CurrentDomain.GetAssemblies().Select(a => a.GetType("EmbeddingGemma.Validation.Editor.ValidationPlayerBuild")).First(t => t != null);
        static MethodInfo DeviceBuild(string name)
        {
            var method = Builder().GetMethod(name); Assert.That(method, Is.Not.Null, "Device mixed builds need dedicated opt-in markers and profile entry points."); return method;
        }
        [TestCase("BuildAndroidMixed")] [TestCase("BuildIOSMixed")]
        public void DeviceBuildsExposeExplicitPinnedSourceEntryPoints(string name)
        { Assert.That(DeviceBuild(name).GetParameters().Length, Is.EqualTo(3)); }
        [Test] public void DeviceMarkerIsSeparateFromTextMediaAndSampleValidation()
        {
            var field = Builder().GetField("DeviceMixedMarkerPath"); Assert.That(field, Is.Not.Null, "Device mixed builds need dedicated opt-in markers and profile entry points.");
            var path = (string)field.GetValue(null); Assert.That(path, Is.EqualTo("Assets/Validation/Generated/Resources/EmbeddingGemmaDeviceMixedValidationBuild.json"));
        }
        [TestCase("BuildAndroidMixed")] [TestCase("BuildIOSMixed")]
        public void WrongActiveTargetRejectsDeviceBuildBeforeCreatingOutputsOrMarkers(string name)
        {
            var root = Path.Combine(Path.GetTempPath(), "embeddinggemma-device-build-" + Guid.NewGuid().ToString("N"));
            var path = Path.Combine(root, name == "BuildAndroidMixed" ? "Validation.apk" : "Xcode");
            var error = Assert.Throws<TargetInvocationException>(() => DeviceBuild(name).Invoke(null, new object[] { path, new string('a', 40), null }));
            Assert.That(error.InnerException, Is.InstanceOf<InvalidOperationException>()); Assert.That(Directory.Exists(root), Is.False);
        }
    }
}
