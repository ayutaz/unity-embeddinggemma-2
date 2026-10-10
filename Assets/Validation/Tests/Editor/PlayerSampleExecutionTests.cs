using System;
using System.Collections;
using System.IO;
using System.Linq;
using System.Xml.Linq;
using EmbeddingGemma.Validation.Editor;
using Newtonsoft.Json.Linq;
using NUnit.Framework;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class PlayerSampleExecutionTests
    {
        string directory;
        [SetUp] public void Setup() { directory = Path.Combine(Path.GetTempPath(), "embeddinggemma-sample-run-" + Guid.NewGuid().ToString("N")); Directory.CreateDirectory(directory); }
        [TearDown] public void Cleanup() => Directory.Delete(directory, true);
        sealed class View : IPlayerSampleView
        {
            public string Fault;
            public bool Released, Destroyed;
            public int Queries;
            public bool Ready => Fault != "preparation";
            public string Error => "injected preparation failure";
            public string ActualBackend => Fault == "backend" ? "CPU" : "GPUCompute";
            public bool FontAlive => !Destroyed || Fault == "font";
            public int FontInstanceId => 42;
            public JObject CacheSnapshot => new JObject { ["Success"] = true, ["CacheReused"] = false, ["TransferredFiles"] = 3 };
            public IEnumerator Prepare(string bundle, string cacheRoot) { yield break; }
            public PlayerRanking[] SearchRanking(string query)
            {
                Queries++;
                var ranks = Reference().Queries[0].ranking.Select(rank => new PlayerRanking { document_id = rank.document_id, score = rank.score }).ToArray();
                if (Fault == "ranking") ranks[0].score = double.NaN;
                return ranks;
            }
            public bool RejectBlank() => Fault != "blank";
            public bool RejectMissingModel() => Fault != "missing";
            public bool ReleaseAndCheckNativeOwner() { Released = true; return Fault != "release"; }
            public void DestroyView() => Destroyed = true;
        }
        static PlayerReferenceSet Reference() => new PlayerReferenceSet { Queries = Enumerable.Range(0, 4).Select(q => new PlayerRow {
            id = "query-" + q, text = "query text " + q, ranking = Enumerable.Range(0, 6).Select(i => new PlayerRanking { document_id = "doc-" + i, score = 1 - i * 0.1 }).ToArray()
        }).ToArray() };
        static void Drain(IEnumerator operation)
        {
            try { while (operation.MoveNext()) if (operation.Current is IEnumerator child) Drain(child); }
            finally { (operation as IDisposable)?.Dispose(); }
        }
        JObject Run(View view, bool failCapture = false, bool failLoad = false)
        {
            var config = new PlayerRunConfiguration { Bundle = directory, Output = Path.Combine(directory, "results.json"), RunId = new string('a', 32) };
            JObject result = null;
            Drain(PlayerSampleExecution.Run(config, PlayerRunProtocolTests.Build(), _ => { }, report => result = report,
                () => view, _ => failLoad ? throw new IOException("injected audit failure") : new PlayerBundle { Reference = Reference(), Receipt = new JObject { ["success"] = true } },
                name => failCapture ? throw new IOException("injected capture failure") : new JObject { ["name"] = name, ["width"] = 1280, ["height"] = 800 }));
            return result;
        }
        [Test] public void ContractRunChecksAllQueriesAndReleasesTheViewWithoutClaimingRealGpu()
        {
            var view = new View(); var result = Run(view);
            Assert.That((bool)result["success"], Is.True);
            Assert.That((string)result["validation_mode"], Is.EqualTo("injected_contract"));
            Assert.That((bool)result["gpu_verified"], Is.False);
            Assert.That(view.Queries, Is.EqualTo(4)); Assert.That((JArray)result["queries"], Has.Count.EqualTo(4));
            Assert.That((JArray)result["screenshots"], Has.Count.EqualTo(2));
            Assert.That(view.Released, Is.True); Assert.That(view.Destroyed, Is.True);
            Assert.That((bool)result["font_destroyed"], Is.True);
        }
        [TestCase("preparation")] [TestCase("backend")] [TestCase("ranking")] [TestCase("blank")] [TestCase("release")] [TestCase("font")] [TestCase("missing")]
        public void RequiredChecksCannotBeOmittedAndFailureStillReleasesOwners(string fault)
        {
            var view = new View { Fault = fault }; var result = Run(view);
            Assert.That((bool)result["success"], Is.False); Assert.That((bool)result["gpu_verified"], Is.False);
            Assert.That((string)result["error"], Is.Not.Null.And.Not.Empty);
            Assert.That(view.Released, Is.True); Assert.That(view.Destroyed, Is.True);
        }
        [Test] public void ScreenshotFailureIsRecordedAndReleasesTheWorker()
        {
            var view = new View(); var result = Run(view, failCapture: true);
            Assert.That((bool)result["success"], Is.False); Assert.That((string)result["error"], Does.Contain("capture failure"));
            Assert.That(view.Released, Is.True); Assert.That(view.Destroyed, Is.True);
        }
        [Test] public void AuditFailureDoesNotBecomeAPassingSampleRun()
        {
            var view = new View(); var result = Run(view, failLoad: true);
            Assert.That((bool)result["success"], Is.False); Assert.That((string)result["error"], Does.Contain("audit failure"));
        }
        [Test] public void SampleReportCanBeSavedWithoutReplacingAnOlderRun()
        {
            var path = Path.Combine(directory, "results.json"); var writer = new PlayerReportFile(path);
            writer.Save(new JObject { ["success"] = false, ["phase"] = "sample_query" });
            Assert.That((string)JObject.Parse(File.ReadAllText(path))["phase"], Is.EqualTo("sample_query"));
            Assert.Throws<IOException>(() => new PlayerReportFile(path));
            Assert.That(File.Exists(path + ".pending"), Is.False);
        }
        [Test] public void LinkerPreservesOptionalSampleAndThePrivateWorkerOwnerFields()
        {
            var root = XDocument.Load("Assets/Validation/Runtime/link.xml").Root;
            var sample = root.Elements("assembly").SingleOrDefault(a => (string)a.Attribute("fullname") == "EmbeddingGemma.TextSearch.Sample");
            Assert.That(sample, Is.Not.Null); Assert.That((string)sample.Attribute("preserve"), Is.EqualTo("all"));
            Assert.That((string)sample.Attribute("ignoreIfMissing"), Is.EqualTo("1"));
            var runtime = root.Elements("assembly").SingleOrDefault(a => (string)a.Attribute("fullname") == "EmbeddingGemma.Runtime");
            Assert.That(runtime, Is.Not.Null);
            foreach (var type in new[] { "TextEmbedder", "TextSearchSession" })
                Assert.That(runtime.Elements("type").Any(t => (string)t.Attribute("fullname") == "EmbeddingGemma." + type && (string)t.Attribute("preserve") == "fields"), Is.True);
        }
        [TestCase(false, true, true)] [TestCase(true, true, false)] [TestCase(false, false, false)] [TestCase(true, false, false)]
        public void OnlyTheOptInPlayerMarkerStartsTheRunner(bool editor, bool marker, bool expected)
            => Assert.That(PlayerSampleRunner.ShouldBootstrap(editor, marker), Is.EqualTo(expected));
        [Test] public void SampleReceiptIncludesTheActualSceneCorpusAndLinkerHashes()
        {
            var info = ValidationPlayerBuild.BuildWindowsSample(Path.Combine(directory, "Sample.exe"), new string('a', 40), _ => { });
            foreach (var name in new[] { "Assets/EmbeddingGemmaTextSearch/TextSearch.unity", "Assets/EmbeddingGemmaTextSearch/Resources/EmbeddingGemmaTextSearch/corpus.json", "Assets/Validation/Runtime/link.xml" })
                Assert.That(info.sourceSha256[name]?.Type, Is.EqualTo(JTokenType.String), name);
        }
    }
}
