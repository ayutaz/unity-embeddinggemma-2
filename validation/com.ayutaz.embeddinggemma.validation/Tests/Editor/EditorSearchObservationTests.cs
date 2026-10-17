using System;
using System.Collections;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using UnityEditor;
using UnityEditor.Search;
using UnityEngine;
using UnityEngine.TestTools;

namespace EmbeddingGemma.Validation.Tests
{
    /// <summary>Default Editor Search observations; never model or GPU qualification.</summary>
    public sealed class EditorSearchObservationTests
    {
        [UnityTest, Timeout(120000)]
        public IEnumerator ExistingSearchDatabasesEnumerateAndFixedPackageAssetRemainsSearchable()
        {
            var output = Environment.GetEnvironmentVariable("EG_EDITOR_SEARCH_REPORT");
            Assert.That(output, Is.Not.Null.And.Not.Empty);
            Assert.That(File.Exists(output), Is.False);
            const string expected = "Packages/com.ayutaz.embeddinggemma/Runtime/TextEmbedder.cs";
            var expectedGuid = AssetDatabase.AssetPathToGUID(expected);
            Assert.That(expectedGuid, Is.Not.Null.And.Not.Empty, "Pinned SDK fixture must exist before search measurement.");
            var report = new JObject
            {
                ["success"] = false, ["unity"] = Application.unityVersion,
                ["model_loaded"] = false, ["gpu_execution_validated"] = false,
                ["scope"] = "Default Search database initialization/readiness, enumeration and fixed SDK asset lookup. Search APIs can initialize their default Library index. No explicit CreateIndex/rebuild, model inference or clean-cache claim. Round first is after readiness, not cold index creation.",
                ["expected_asset"] = expected, ["expected_guid"] = expectedGuid,
                ["search_query"] = "TextEmbedder",
                ["search_route"] = "asset provider with Packages and WantsMore; file-path results plus default index, not isolated index-only latency",
                ["rounds"] = new JArray()
            };
            try
            {
                using var initializationProcess = Process.GetCurrentProcess();
                var initializationCpu = initializationProcess.TotalProcessorTime;
                var initialization = Stopwatch.StartNew();
                var initialDatabases = SearchService.EnumerateDatabases().ToArray();
                initialization.Stop(); initializationProcess.Refresh();
                report["initial_enumeration_wall_seconds"] = initialization.Elapsed.TotalSeconds;
                report["initial_enumeration_process_cpu_seconds"] = (initializationProcess.TotalProcessorTime - initializationCpu).TotalSeconds;
                Assert.That(initialDatabases.Length, Is.GreaterThan(0), "Empty database list cannot establish readiness.");
                report["initial_databases"] = new JArray(initialDatabases.Select(db => new JObject
                { ["name"] = db.name, ["ready"] = SearchService.IsIndexReady(db.name) }));
                File.WriteAllText(output, report.ToString());
                initializationCpu = initializationProcess.TotalProcessorTime; initialization.Restart();
                var polls = 0;
                while (!SearchService.IsIndexReady(null) && initialization.Elapsed.TotalSeconds < 90)
                {
                    polls++; EditorApplication.QueuePlayerLoopUpdate(); yield return null;
                }
                initialization.Stop(); initializationProcess.Refresh();
                report["index_readiness_wait_wall_seconds"] = initialization.Elapsed.TotalSeconds;
                report["index_readiness_wait_process_cpu_seconds"] = (initializationProcess.TotalProcessorTime - initializationCpu).TotalSeconds;
                report["index_readiness_poll_count"] = polls;
                report["index_ready_before_rounds"] = SearchService.IsIndexReady(null);
                File.WriteAllText(output, report.ToString());
                Assert.That(report.Value<bool>("index_ready_before_rounds"), Is.True, "Not-ready index is unavailable, never zero-time success.");
                foreach (var round in new[] { "first", "warm", "repeat-1", "repeat-2" })
                {
                    using var process = Process.GetCurrentProcess();
                    var begin = DateTime.UtcNow; var cpu = process.TotalProcessorTime;
                    var timer = Stopwatch.StartNew();
                    var databases = SearchService.EnumerateDatabases().ToArray();
                    timer.Stop(); process.Refresh();
                    var enumerationWall = timer.Elapsed.TotalSeconds;
                    var enumerationCpu = (process.TotalProcessorTime - cpu).TotalSeconds;
                    // Keep readiness outside the enumeration/search timing windows.
                    var readiness = new JArray(databases.Select(db => new JObject
                    {
                        ["name"] = db.name, ["ready"] = SearchService.IsIndexReady(db.name)
                    }));
                    cpu = process.TotalProcessorTime; timer.Restart();
                    const SearchFlags flags = SearchFlags.Synchronous | SearchFlags.Packages | SearchFlags.WantsMore;
                    using var context = SearchService.CreateContext("asset", "TextEmbedder", flags);
                    var items = SearchService.GetItems(context, flags).ToArray();
                    timer.Stop(); process.Refresh();
                    var found = items.Any(item => item.id == expected || item.id == expectedGuid
                        || AssetDatabase.GetAssetPath(item.ToObject()) == expected);
                    ((JArray)report["rounds"]).Add(new JObject
                    {
                        ["round"] = round, ["begin_utc"] = begin.ToString("O"),
                        ["enumeration_wall_seconds"] = enumerationWall,
                        ["enumeration_process_cpu_seconds"] = enumerationCpu,
                        ["databases"] = readiness,
                        ["search_wall_seconds"] = timer.Elapsed.TotalSeconds,
                        ["search_process_cpu_seconds"] = (process.TotalProcessorTime - cpu).TotalSeconds,
                        ["result_count"] = items.Length, ["expected_asset_found"] = found,
                        ["allocation_bytes"] = null,
                        ["allocation_scope"] = "Unavailable in this probe; never inferred from managed heap size or reported as zero."
                    });
                    File.WriteAllText(output, report.ToString());
                    Assert.That(found, Is.True, "A completed search without the known asset is not accepted.");
                }
                report["success"] = true;
            }
            finally { File.WriteAllText(output, report.ToString()); }
        }
    }
}
