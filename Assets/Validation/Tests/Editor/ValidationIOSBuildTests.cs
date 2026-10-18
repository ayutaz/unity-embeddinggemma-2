using System;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.ExceptionServices;
using EmbeddingGemma.Validation.Editor;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using UnityEditor;
using UnityEditor.Build;
using UnityEngine.Rendering;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class ValidationIOSBuildTests
    {
        string directory;
        [SetUp] public void SetUp() => directory = Path.Combine(Path.GetTempPath(), "embeddinggemma-ios-" + Guid.NewGuid().ToString("N"));
        [TearDown] public void TearDown() { if (Directory.Exists(directory)) Directory.Delete(directory, true); }
        static object Call(string method, params object[] args)
        {
            var entry = typeof(ValidationPlayerBuild).GetMethod(method, BindingFlags.Public | BindingFlags.Static);
            Assert.That(entry, Is.Not.Null, "An explicit iOS validation build profile is required.");
            try { return entry.Invoke(null, args); }
            catch (TargetInvocationException error) { ExceptionDispatchInfo.Capture(error.InnerException).Throw(); throw; }
        }
        BuildPlayerOptions Options(string output, string commit) => (BuildPlayerOptions)Call("IOSOptions", output, commit);
        static void Profile(Action action) => Call("WithIOSProfile", action);

        [Test] public void SelectsDeviceReleaseExportWithoutBuildingOrSigning()
        {
            var output = Path.Combine(directory, "Xcode"); var options = Options(output, new string('a', 40));
            Assert.That(options.target, Is.EqualTo(BuildTarget.iOS)); Assert.That(options.targetGroup, Is.EqualTo(BuildTargetGroup.iOS));
            Assert.That(options.options, Is.EqualTo(BuildOptions.None));
            Assert.That(options.scenes, Is.EqualTo(new[] { ValidationPlayerBuild.ScenePath }));
            Assert.That(options.locationPathName, Is.EqualTo(Path.GetFullPath(output)));
            Assert.That(Directory.Exists(directory), Is.False, "Option validation must not create output or settings.");
        }
        [TestCase(null)] [TestCase("")] [TestCase(" ")] [TestCase("abc")] [TestCase("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")]
        public void RejectsUnpinnedSource(string commit) => Assert.Throws<ArgumentException>(() => Options(Path.Combine(directory, "Xcode"), commit));
        [TestCase(null)] [TestCase("")] [TestCase(" ")]
        public void RejectsMissingOutput(string output) => Assert.Throws<ArgumentException>(() => Options(output, new string('a', 40)));
        [TestCase("Validation.ipa")] [TestCase("Validation.xcodeproj")]
        public void RequiresExportDirectoryRatherThanAnInstallableOrProjectFile(string name)
            => Assert.Throws<ArgumentException>(() => Options(Path.Combine(directory, name), new string('a', 40)));
        [TestCase(false)] [TestCase(true)]
        public void NeverOverwritesExistingSiblingOrExport(bool export)
        {
            Directory.CreateDirectory(directory); var previous = Path.Combine(directory, export ? "Xcode" : "earlier-result.txt");
            if (export) { Directory.CreateDirectory(previous); File.WriteAllText(Path.Combine(previous, "keep"), "original"); }
            else File.WriteAllText(previous, "original");
            Assert.Throws<IOException>(() => Options(Path.Combine(directory, "Xcode"), new string('a', 40)));
            Assert.That(File.ReadAllText(export ? Path.Combine(previous, "keep") : previous), Is.EqualTo("original"));
        }
        [Test] public void RejectsNullCallbackBeforeChangingSettings() => Assert.Throws<ArgumentNullException>(() => Profile(null));

        [TestCase(false, false)] [TestCase(true, false)] [TestCase(false, true)] [TestCase(true, true)]
        public void AppliesReleaseMetalDeviceProfileAndRestoresAfterSuccessOrFailure(bool fail, bool automaticBefore)
        {
            var target = NamedBuildTarget.iOS;
            var backend = PlayerSettings.GetScriptingBackend(target); var stripping = PlayerSettings.GetManagedStrippingLevel(target);
            var compiler = PlayerSettings.GetIl2CppCompilerConfiguration(target); var identifier = PlayerSettings.GetApplicationIdentifier(target);
            var sdk = PlayerSettings.iOS.sdkVersion; var device = PlayerSettings.iOS.targetDevice; var minimum = PlayerSettings.iOS.targetOSVersionString;
            var automatic = PlayerSettings.GetUseDefaultGraphicsAPIs(BuildTarget.iOS);
            PlayerSettings.SetUseDefaultGraphicsAPIs(BuildTarget.iOS, false); var apis = PlayerSettings.GetGraphicsAPIs(BuildTarget.iOS);
            PlayerSettings.SetUseDefaultGraphicsAPIs(BuildTarget.iOS, automatic);
            var signing = PlayerSettings.iOS.appleEnableAutomaticSigning; var team = PlayerSettings.iOS.appleDeveloperTeamID;
            var provisioning = PlayerSettings.iOS.iOSManualProvisioningProfileID; var profileType = PlayerSettings.iOS.iOSManualProvisioningProfileType;
            try
            {
                PlayerSettings.SetManagedStrippingLevel(target, ManagedStrippingLevel.Low);
                PlayerSettings.SetIl2CppCompilerConfiguration(target, Il2CppCompilerConfiguration.Debug);
                PlayerSettings.SetApplicationIdentifier(target, "com.example.validation.before");
                PlayerSettings.iOS.sdkVersion = iOSSdkVersion.SimulatorSDK; PlayerSettings.iOS.targetDevice = iOSTargetDevice.iPadOnly;
                PlayerSettings.iOS.targetOSVersionString = "16.0";
                PlayerSettings.SetUseDefaultGraphicsAPIs(BuildTarget.iOS, false);
                PlayerSettings.SetGraphicsAPIs(BuildTarget.iOS, new[] { GraphicsDeviceType.Metal });
                PlayerSettings.SetUseDefaultGraphicsAPIs(BuildTarget.iOS, automaticBefore);
                var calls = 0;
                Action execute = () =>
                {
                    calls++; Assert.That(PlayerSettings.GetScriptingBackend(target), Is.EqualTo(ScriptingImplementation.IL2CPP));
                    Assert.That(PlayerSettings.GetManagedStrippingLevel(target), Is.EqualTo(ManagedStrippingLevel.High));
                    Assert.That(PlayerSettings.GetIl2CppCompilerConfiguration(target), Is.EqualTo(Il2CppCompilerConfiguration.Release));
                    Assert.That(PlayerSettings.GetApplicationIdentifier(target), Is.EqualTo("com.ayutaz.embeddinggemma.validation"));
                    Assert.That(PlayerSettings.iOS.sdkVersion, Is.EqualTo(iOSSdkVersion.DeviceSDK));
                    Assert.That(PlayerSettings.iOS.targetDevice, Is.EqualTo(iOSTargetDevice.iPhoneAndiPad));
                    Assert.That(PlayerSettings.iOS.targetOSVersionString, Is.EqualTo("15.0"));
                    Assert.That(PlayerSettings.GetUseDefaultGraphicsAPIs(BuildTarget.iOS), Is.False);
                    Assert.That(PlayerSettings.GetGraphicsAPIs(BuildTarget.iOS), Is.EqualTo(new[] { GraphicsDeviceType.Metal }));
                    // Identity values are never part of a public report and are never changed by this profile.
                    Assert.That(PlayerSettings.iOS.appleEnableAutomaticSigning == signing, Is.True);
                    Assert.That(PlayerSettings.iOS.appleDeveloperTeamID == team, Is.True);
                    Assert.That(PlayerSettings.iOS.iOSManualProvisioningProfileID == provisioning, Is.True);
                    Assert.That(PlayerSettings.iOS.iOSManualProvisioningProfileType == profileType, Is.True);
                    var receipt = (JObject)Call("IOSProfileSettings");
                    Assert.That(receipt.Properties().Select(p => p.Name), Is.EquivalentTo(new[] { "application_identifier", "sdk", "device_family", "deployment_target", "automatic_graphics_api", "graphics_apis", "signing_scope" }));
                    Assert.That((string)receipt["sdk"], Is.EqualTo("DeviceSDK"));
                    Assert.That((string)receipt["signing_scope"], Is.EqualTo("Private Xcode signing; identities and provisioning values omitted"));
                    Assert.That(receipt["graphics_apis"].ToObject<string[]>(), Is.EqualTo(new[] { "Metal" }));
                    if (fail) throw new InvalidOperationException("injected iOS build failure");
                };
                if (fail) Assert.That(Assert.Throws<InvalidOperationException>(() => Profile(execute)).Message, Is.EqualTo("injected iOS build failure"));
                else Profile(execute);
                Assert.That(calls, Is.EqualTo(1));
                Assert.That(PlayerSettings.GetScriptingBackend(target), Is.EqualTo(backend));
                Assert.That(PlayerSettings.GetManagedStrippingLevel(target), Is.EqualTo(ManagedStrippingLevel.Low));
                Assert.That(PlayerSettings.GetIl2CppCompilerConfiguration(target), Is.EqualTo(Il2CppCompilerConfiguration.Debug));
                Assert.That(PlayerSettings.GetApplicationIdentifier(target), Is.EqualTo("com.example.validation.before"));
                Assert.That(PlayerSettings.iOS.sdkVersion, Is.EqualTo(iOSSdkVersion.SimulatorSDK));
                Assert.That(PlayerSettings.iOS.targetDevice, Is.EqualTo(iOSTargetDevice.iPadOnly)); Assert.That(PlayerSettings.iOS.targetOSVersionString, Is.EqualTo("16.0"));
                Assert.That(PlayerSettings.GetUseDefaultGraphicsAPIs(BuildTarget.iOS), Is.EqualTo(automaticBefore));
                PlayerSettings.SetUseDefaultGraphicsAPIs(BuildTarget.iOS, false);
                Assert.That(PlayerSettings.GetGraphicsAPIs(BuildTarget.iOS), Is.EqualTo(new[] { GraphicsDeviceType.Metal }));
            }
            finally
            {
                PlayerSettings.SetScriptingBackend(target, backend); PlayerSettings.SetManagedStrippingLevel(target, stripping);
                PlayerSettings.SetIl2CppCompilerConfiguration(target, compiler); PlayerSettings.SetApplicationIdentifier(target, identifier);
                PlayerSettings.iOS.sdkVersion = sdk; PlayerSettings.iOS.targetDevice = device; PlayerSettings.iOS.targetOSVersionString = minimum;
                PlayerSettings.SetGraphicsAPIs(BuildTarget.iOS, apis); PlayerSettings.SetUseDefaultGraphicsAPIs(BuildTarget.iOS, automatic);
            }
        }
        [Test] public void WrongActiveTargetDoesNotCreateBuildOrTouchSigning()
        {
            Assert.That(EditorUserBuildSettings.activeBuildTarget, Is.Not.EqualTo(BuildTarget.iOS));
            var calls = 0; Action<BuildPlayerOptions> build = _ => calls++;
            Assert.Throws<InvalidOperationException>(() => Call("BuildIOS", Path.Combine(directory, "Xcode"), new string('a', 40), build));
            Assert.That(calls, Is.Zero); Assert.That(Directory.Exists(directory), Is.False);
            Assert.That(File.Exists(ValidationPlayerBuild.ScenePath), Is.False); Assert.That(File.Exists(ValidationPlayerBuild.MarkerPath), Is.False);
        }
    }
}
