using System;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.ExceptionServices;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using UnityEditor;
using UnityEditor.Build;
using UnityEditor.SceneManagement;
using UnityEngine.Rendering;

namespace EmbeddingGemma.Tests
{
    public sealed class GpuConstantProbeBuildTests
    {
        [TestCase(false)] [TestCase(true)]
        public void ProbeBuildUsesIl2CppHighD3D12AndRestoresEverything(bool fail)
        {
            var type=AppDomain.CurrentDomain.GetAssemblies().Select(assembly=>assembly.GetType("EmbeddingGemma.Validation.Editor.GpuConstantProbeBuild")).FirstOrDefault(value=>value!=null);
            Assert.That(type,Is.Not.Null,"Probe build helper is missing.");
            var target=NamedBuildTarget.Standalone;var backend=PlayerSettings.GetScriptingBackend(target);
            var stripping=PlayerSettings.GetManagedStrippingLevel(target);var compiler=PlayerSettings.GetIl2CppCompilerConfiguration(target);
            var automatic=PlayerSettings.GetUseDefaultGraphicsAPIs(BuildTarget.StandaloneWindows64);
            var graphics=PlayerSettings.GetGraphicsAPIs(BuildTarget.StandaloneWindows64);
            var scenes=EditorSceneManager.GetSceneManagerSetup().Select(scene=>scene.path).ToArray();
            var directory=Path.GetFullPath(Path.Combine("artifacts","gpu-probe-build-test-"+Guid.NewGuid().ToString("N")));
            var calls=0; Action<BuildPlayerOptions> execute=options=>
            {
                calls++;Assert.That(options.options,Is.EqualTo(BuildOptions.None));Assert.That(options.target,Is.EqualTo(BuildTarget.StandaloneWindows64));
                Assert.That(PlayerSettings.GetScriptingBackend(target),Is.EqualTo(ScriptingImplementation.IL2CPP));
                Assert.That(PlayerSettings.GetManagedStrippingLevel(target),Is.EqualTo(ManagedStrippingLevel.High));
                Assert.That(PlayerSettings.GetIl2CppCompilerConfiguration(target),Is.EqualTo(Il2CppCompilerConfiguration.Release));
                Assert.That(PlayerSettings.GetUseDefaultGraphicsAPIs(BuildTarget.StandaloneWindows64),Is.False);
                Assert.That(PlayerSettings.GetGraphicsAPIs(BuildTarget.StandaloneWindows64),Is.EqualTo(new[] {GraphicsDeviceType.Direct3D12}));
                Assert.That(File.Exists("Assets/Validation/Generated/Resources/GpuConstantProbeBuild.json"),Is.True);
                Assert.That(File.Exists("Assets/Validation/Generated/Resources/EmbeddingGemmaValidationBuild.json"),Is.False);
                if(fail)throw new InvalidOperationException("probe build failure fixture");
            };
            void Build()
            {
                try{type.GetMethod("Build").Invoke(null,new object[] {Path.Combine(directory,"Probe.exe"),new string('a',40),execute});}
                catch(TargetInvocationException error){ExceptionDispatchInfo.Capture(error.InnerException).Throw();throw;}
            }
            if(fail)Assert.Throws<InvalidOperationException>(Build);else Build();
            Assert.That(calls,Is.EqualTo(1));Assert.That(PlayerSettings.GetScriptingBackend(target),Is.EqualTo(backend));
            Assert.That(PlayerSettings.GetManagedStrippingLevel(target),Is.EqualTo(stripping));Assert.That(PlayerSettings.GetIl2CppCompilerConfiguration(target),Is.EqualTo(compiler));
            Assert.That(PlayerSettings.GetUseDefaultGraphicsAPIs(BuildTarget.StandaloneWindows64),Is.EqualTo(automatic));
            Assert.That(PlayerSettings.GetGraphicsAPIs(BuildTarget.StandaloneWindows64),Is.EqualTo(graphics));
            Assert.That(EditorSceneManager.GetSceneManagerSetup().Select(scene=>scene.path),Is.EqualTo(scenes));
            Assert.That(File.Exists("Assets/Validation/Generated/Resources/GpuConstantProbeBuild.json"),Is.False);
            var receipt=JObject.Parse(File.ReadAllText(Path.Combine(directory,"build.json")));
            Assert.That(receipt.Value<bool>("success"),Is.False,"Injected build must never count as a real build.");
            Assert.That(receipt["source_sha256"]["runtime/GpuConstantWorker.cs"],Is.Not.Null);
            Assert.That(receipt["source_sha256"]["runtime/link.xml"],Is.Not.Null);
        }
    }
}
