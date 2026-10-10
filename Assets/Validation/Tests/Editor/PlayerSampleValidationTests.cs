using System;
using System.IO;
using System.Linq;
using System.Reflection;
using EmbeddingGemma.Validation.Editor;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEditor;
using UnityEditor.Build;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class PlayerSampleValidationTests
    {
        string directory;
        GameObject owner;
        [SetUp] public void Setup() { directory = Path.Combine(Path.GetTempPath(), "embeddinggemma-sample-" + Guid.NewGuid().ToString("N")); Directory.CreateDirectory(directory); }
        [TearDown] public void Cleanup() { if(owner != null) UnityEngine.Object.DestroyImmediate(owner); Directory.Delete(directory,true); }
        static PlayerRow Query() => new() { id="query", ranking=Enumerable.Range(0,6).Select(i=>new PlayerRanking {document_id="doc-"+i,score=1-i*0.1}).ToArray() };
        static PlayerRanking[] Hits() => Enumerable.Range(0,6).Select(i=>new PlayerRanking { document_id="doc-"+i, score=1-i*0.1 }).ToArray();
        [Test] public void CompleteRankingAndFiniteScoresAreAccepted() => PlayerSampleQueryValidation.Check(Query(),Hits());
        [TestCase("count")] [TestCase("order")] [TestCase("score")] [TestCase("nan")] [TestCase("infinity")] [TestCase("duplicate")]
        public void IncorrectOrNonfiniteRankingCannotPass(string fault)
        {
            var hits=Hits();
            if(fault=="count") hits=hits.Take(5).ToArray();
            if(fault=="order") Array.Reverse(hits);
            if(fault=="score") hits[0]=new PlayerRanking { document_id=hits[0].document_id,score=0.5 };
            if(fault=="nan") hits[0]=new PlayerRanking { document_id=hits[0].document_id,score=double.NaN };
            if(fault=="infinity") hits[0]=new PlayerRanking { document_id=hits[0].document_id,score=double.PositiveInfinity };
            if(fault=="duplicate") hits[1]=hits[0];
            Assert.Throws<ArgumentException>(()=>PlayerSampleQueryValidation.Check(Query(),hits));
        }
        [Test] public void InvalidReferenceCannotAuthorizeAResult() => Assert.Throws<ArgumentException>(()=>PlayerSampleQueryValidation.Check(new PlayerRow(),Hits()));
        [Test] public void SampleBuildUsesImportedSceneAndReleaseOptions()
        {
            var options=ValidationPlayerBuild.WindowsSampleOptions(Path.Combine(directory,"Sample.exe"),new string('a',40));
            Assert.That(options.scenes,Is.EqualTo(new[]{ValidationPlayerBuild.SampleScenePath}));
            Assert.That(options.target,Is.EqualTo(BuildTarget.StandaloneWindows64));
            Assert.That(options.options,Is.EqualTo(BuildOptions.None));
        }
        [TestCase(false)] [TestCase(true)]
        public void SampleBuildMarkerIsIsolatedAndSettingsAreRestored(bool fail)
        {
            var target=NamedBuildTarget.Standalone;
            var backend=PlayerSettings.GetScriptingBackend(target);var stripping=PlayerSettings.GetManagedStrippingLevel(target);var compiler=PlayerSettings.GetIl2CppCompilerConfiguration(target);
            var scenes=EditorSceneManager.GetSceneManagerSetup().Select(s=>s.path).ToArray();
            Action<BuildPlayerOptions> injected=options=>
            {
                Assert.That(options.scenes,Is.EqualTo(new[]{ValidationPlayerBuild.SampleScenePath}));
                Assert.That(File.Exists(ValidationPlayerBuild.SampleMarkerPath),Is.True);
                Assert.That(File.Exists(ValidationPlayerBuild.MarkerPath),Is.False,"The full precision runner must not bootstrap alongside the sample runner.");
                Assert.That(PlayerSettings.GetScriptingBackend(target),Is.EqualTo(ScriptingImplementation.IL2CPP));
                Assert.That(PlayerSettings.GetManagedStrippingLevel(target),Is.EqualTo(ManagedStrippingLevel.High));
                if(fail)throw new InvalidOperationException("injected sample failure");
            };
            if(fail) Assert.That(Assert.Throws<InvalidOperationException>(()=>ValidationPlayerBuild.BuildWindowsSample(Path.Combine(directory,"Sample.exe"),new string('a',40),injected)).Message,Is.EqualTo("injected sample failure"));
            else ValidationPlayerBuild.BuildWindowsSample(Path.Combine(directory,"Sample.exe"),new string('a',40),injected);
            Assert.That(PlayerSettings.GetScriptingBackend(target),Is.EqualTo(backend));Assert.That(PlayerSettings.GetManagedStrippingLevel(target),Is.EqualTo(stripping));Assert.That(PlayerSettings.GetIl2CppCompilerConfiguration(target),Is.EqualTo(compiler));
            Assert.That(EditorSceneManager.GetSceneManagerSetup().Select(s=>s.path),Is.EqualTo(scenes));
            Assert.That(File.Exists(ValidationPlayerBuild.SampleMarkerPath),Is.False);
            var receipt=Newtonsoft.Json.Linq.JObject.Parse(File.ReadAllText(Path.Combine(directory,"build.json")));
            Assert.That((bool)receipt["success"],Is.False,"Injected build does not prove Player success.");
        }
        sealed class FakeEmbedder:ITextEmbedder
        {
            public BackendType Backend=>BackendType.GPUCompute;
            public bool Disposed;
            public float[] EmbedQuery(string query)=>Vector(); public float[] EmbedDocument(string document,string title=null)=>Vector();
            static float[] Vector(){var v=new float[768];v[0]=1;return v;}
            public void Dispose()=>Disposed=true;
        }
        [Test] public void ProxyDrivesRealSampleSearchGuardAndReleaseWithoutCallingFakeGpuReal()
        {
            var type=Type.GetType("EmbeddingGemma.Samples.TextSearchSample,EmbeddingGemma.TextSearch.Sample");Assert.That(type,Is.Not.Null,"Import the pinned sample before this contract test.");
            owner=new GameObject("Sample proxy contract");var sample=(Component)owner.AddComponent(type);
            type.GetField("Backend").SetValue(sample,BackendType.GPUCompute);
            var fake=new FakeEmbedder();var prepare=type.GetMethod("PrepareDocuments");
            Assert.That((bool)prepare.Invoke(sample,new object[]{(Func<ITextEmbedder>)(()=>fake)}),Is.True);
            var proxy=new PlayerSampleView(sample);
            Assert.That(proxy.Ready,Is.True);
            Assert.That(proxy.Search("猫"),Is.True);Assert.That(proxy.Results.Count,Is.EqualTo(6));
            Assert.That(proxy.ActualBackend,Is.Null,"A synthetic embedder is not an actual Sentis Worker.");
            Assert.That(proxy.Search(""),Is.False);Assert.That(proxy.Results,Is.Empty);
            proxy.Release();Assert.That(proxy.Ready,Is.False);Assert.That(fake.Disposed,Is.True);
        }
    }
}
