using System;
using System.Collections;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using UnityEditor;
using UnityEditor.Search;
using UnityEngine;
using UnityEngine.TestTools;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class EditorSearchRebuildObservationTests
    {
        const string Folder = "Assets/EmbeddingGemmaIndexRebuildProbe";
        const string Fixture = Folder + "/EmbeddingGemmaProbe.txt";
        const string Query = "embeddinggemmaprobe";
        const BindingFlags Flags = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;

        static object Property(object database, string name)
        {
            var property = database.GetType().GetProperty(name, Flags)
                ?? throw new NotSupportedException("Pinned Search database property missing: " + name);
            return property.GetValue(database);
        }
        static bool Boolean(object database, string name)
            => Property(database, name) is bool value ? value : throw new NotSupportedException("Expected Boolean: " + name);
        static bool Settled(object database)
            => Boolean(database, "ready") && !Boolean(database, "updating") && !Boolean(database, "needsUpdate");
        static SearchIndexer Index(object database)
            => Property(database, "index") as SearchIndexer ?? throw new NotSupportedException("Pinned SearchIndexer unavailable.");
        static JObject Snapshot(object database)
            => new JObject { ["ready"] = Boolean(database, "ready"), ["updating"] = Boolean(database, "updating"),
                ["needs_update"] = Boolean(database, "needsUpdate"), ["loaded"] = Boolean(database, "loaded"),
                ["loading_state"] = Property(database, "LoadingState").ToString(), ["document_count"] = Index(database).documentCount };
        static bool FindsFixture(object database, string guid)
        {
            var index = Index(database);
            return index.Search(Query, int.MaxValue, 1024).Any(result =>
            {
                var document = index.GetDocument(result.index);
                return document.valid && (document.source == Fixture || document.id == Fixture || document.id == guid);
            });
        }

        [UnityTest, Timeout(180000)]
        public IEnumerator ExplicitDefaultDatabaseRebuildCompletesAndDirectIndexKeepsFixture()
        {
            var output = Environment.GetEnvironmentVariable("EG_EDITOR_INDEX_REBUILD_REPORT");
            Assert.That(output, Is.Not.Null.And.Not.Empty); Assert.That(File.Exists(output), Is.False);
            Assert.That(Directory.Exists(Folder), Is.False, "Only this fresh consumer's owned fixture may be created/deleted.");
            var requireSamples = Environment.GetEnvironmentVariable("EG_EDITOR_INDEX_EXPECT_SAMPLES") == "1";
            var sampleScenes = requireSamples ? new[] { "Assets/EmbeddingGemmaTextSearch/TextSearch.unity",
                "Assets/EmbeddingGemmaMultimodalSearch/MultimodalSearch.unity" } : Array.Empty<string>();
            foreach (var scene in sampleScenes) Assert.That(AssetDatabase.AssetPathToGUID(scene), Is.Not.Empty, scene);
            var report = new JObject { ["success"] = false, ["unity"] = Application.unityVersion,
                ["runtime_source"] = Environment.GetEnvironmentVariable("EG_ACCEPTANCE_SOURCE"),
                ["validation_source"] = Environment.GetEnvironmentVariable("EG_VALIDATION_SOURCE"),
                ["model_loaded"] = false, ["gpu_execution_validated"] = false,
                ["scope"] = "Explicit Build() on the default Assets Search database after readiness, twice in one Editor. Existing asset/index artifacts can be reused. Direct SearchIndexer queries exclude file-path provider fallback. CPU is process-wide including background work; not exclusive indexing CPU or cold artifact regeneration.",
                ["fixture"] = Fixture, ["query"] = Query, ["required_sample_scenes"] = new JArray(sampleScenes), ["rounds"] = new JArray() };
            try
            {
                Directory.CreateDirectory(Folder);
                File.WriteAllText(Fixture, "EmbeddingGemma explicit default index rebuild fixture\n");
                AssetDatabase.ImportAsset(Fixture, ImportAssetOptions.ForceSynchronousImport);
                var guid = AssetDatabase.AssetPathToGUID(Fixture); Assert.That(guid, Is.Not.Empty);
                report["fixture_guid"] = guid;
                var candidates = SearchService.EnumerateDatabases().Where(db => db.roots != null
                    && db.roots.Any(root => string.Equals(root.TrimEnd('/'), "Assets", StringComparison.OrdinalIgnoreCase))).ToArray();
                Assert.That(candidates.Length, Is.EqualTo(1), "A unique Assets database is required; don't rebuild an arbitrary index.");
                var database = (object)candidates[0]; report["database_name"] = candidates[0].name;
                report["database_type"] = database.GetType().FullName;
                var readiness = Stopwatch.StartNew();
                while (!Settled(database) && readiness.Elapsed.TotalSeconds < 60)
                { EditorApplication.QueuePlayerLoopUpdate(); yield return null; }
                report["initial_fixture_readiness_seconds"] = readiness.Elapsed.TotalSeconds;
                var initialIndex = Index(database);
                report["initial_direct_index_results"] = new JArray(initialIndex.Search(Query, int.MaxValue, 1024).Select(result =>
                {
                    var document = initialIndex.GetDocument(result.index);
                    return new JObject { ["result_id"] = result.id, ["document_index"] = result.index,
                        ["document_id"] = document.id, ["document_source"] = document.source, ["document_valid"] = document.valid };
                }));
                report["before_rebuild"] = Snapshot(database); File.WriteAllText(output, report.ToString());
                Assert.That(Settled(database), Is.True);
                // A loaded cache can omit newly imported assets. Record that state;
                // the explicit rebuild must index the current fixture and both samples.
                report["initial_fixture_found"] = FindsFixture(database, guid);
                foreach (var round in new[] { "first-explicit-rebuild", "repeat-explicit-rebuild" })
                {
                    using var process = Process.GetCurrentProcess(); process.Refresh();
                    var begin = DateTime.UtcNow; var cpu = process.TotalProcessorTime;
                    var workingSet = process.WorkingSet64; var timer = Stopwatch.StartNew();
                    EditorIndexRebuild.Begin(database);
                    var afterSubmission = Snapshot(database); var submissionSeconds = timer.Elapsed.TotalSeconds;
                    Assert.That(afterSubmission.Value<bool>("ready"), Is.False, "A ready snapshot alone cannot prove a rebuild started.");
                    var polls = 0;
                    while (!Settled(database) && timer.Elapsed.TotalSeconds < 60)
                    { polls++; EditorApplication.QueuePlayerLoopUpdate(); yield return null; }
                    timer.Stop(); process.Refresh();
                    var end = DateTime.UtcNow; var processCpu = (process.TotalProcessorTime - cpu).TotalSeconds;
                    var completed = Snapshot(database); var found = FindsFixture(database, guid);
                    var index = Index(database);
                    var scenesFound = new JArray(sampleScenes.Select(path =>
                    {
                        var query = Path.GetFileNameWithoutExtension(path).ToLowerInvariant();
                        var foundScene = index.Search(query, int.MaxValue, 1024).Any(result =>
                        {
                            var document = index.GetDocument(result.index);
                            return document.valid && document.source == path;
                        });
                        return new JObject { ["path"] = path, ["query"] = query, ["found"] = foundScene };
                    }));
                    var row = new JObject { ["round"] = round, ["begin_utc"] = begin.ToString("O"), ["end_utc"] = end.ToString("O"),
                        ["submission_wall_seconds"] = submissionSeconds, ["completion_wall_seconds"] = timer.Elapsed.TotalSeconds,
                        ["process_cpu_seconds"] = processCpu, ["polls"] = polls,
                        ["after_submission"] = afterSubmission, ["after_completion"] = completed, ["direct_index_fixture_found"] = found,
                        ["required_sample_scenes"] = scenesFound,
                        ["begin_process_working_set_bytes"] = workingSet, ["end_process_working_set_bytes"] = process.WorkingSet64,
                        ["process_lifetime_peak_working_set_bytes"] = process.PeakWorkingSet64,
                        ["exclusive_background_allocation_bytes"] = null,
                        ["allocation_scope"] = "Not measured: synchronous main-thread marker bytes cannot describe background rebuild allocation. No zero/all-thread attribution inferred." };
                    ((JArray)report["rounds"]).Add(row); File.WriteAllText(output, report.ToString());
                    if (Environment.GetEnvironmentVariable("EG_EDITOR_REBUILD_FRAME_ALLOCATION") == "1")
                        Assert.That(row["frame_managed_allocation"], Is.Not.Null,
                            "An explicit rebuild allocation observation must not inherit the synchronous search marker's pass.");
                    Assert.That(polls, Is.GreaterThan(0)); Assert.That(Settled(database), Is.True);
                    Assert.That(found, Is.True); Assert.That(timer.Elapsed.TotalSeconds, Is.GreaterThan(0));
                    Assert.That(scenesFound.All(scene => scene.Value<bool>("found")), Is.True,
                        "Explicit rebuild must include both imported sample scenes in the actual index.");
                }
                report["success"] = true;
            }
            finally
            {
                if (Directory.Exists(Folder)) report["owned_fixture_removed"] = AssetDatabase.DeleteAsset(Folder);
                File.WriteAllText(output, report.ToString());
            }
        }
    }
}
