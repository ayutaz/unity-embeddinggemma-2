using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using UnityEditor;
using UnityEditor.Search;
using UnityEngine;

namespace EmbeddingGemma.Validation.Tests
{
    /// <summary>Read-only Editor Search observations; never model or GPU qualification.</summary>
    public sealed class EditorSearchObservationTests
    {
        [Test, Timeout(120000)]
        public void ExistingSearchDatabasesEnumerateAndFixedPackageAssetRemainsSearchable()
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
                ["scope"] = "Read-only existing Search database enumeration and synchronous fixed SDK asset lookup. No index creation/rebuild, no model inference, no clean-cache claim.",
                ["expected_asset"] = expected, ["expected_guid"] = expectedGuid,
                ["rounds"] = new JArray()
            };
            try
            {
                foreach (var round in new[] { "first", "warm", "repeat-1", "repeat-2" })
                {
                    using var process = Process.GetCurrentProcess();
                    var begin = DateTime.UtcNow; var cpu = process.TotalProcessorTime;
                    var timer = Stopwatch.StartNew();
                    var databases = SearchService.EnumerateDatabases().ToArray();
                    timer.Stop(); process.Refresh();
                    var enumerationWall = timer.Elapsed.TotalSeconds;
                    var enumerationCpu = (process.TotalProcessorTime - cpu).TotalSeconds;
                    // Readiness is an observation, not an instruction to build or change an index.
                    var readiness = new JArray(databases.Select(db => new JObject
                    {
                        ["name"] = db.name, ["ready"] = SearchService.IsIndexReady(db.name)
                    }));
                    cpu = process.TotalProcessorTime; timer.Restart();
                    using var context = SearchService.CreateContext("asset", "a:packages TextEmbedder", SearchFlags.Synchronous);
                    var items = SearchService.GetItems(context, SearchFlags.Synchronous).ToArray();
                    timer.Stop(); process.Refresh();
                    var found = items.Any(item => item.id == expected || item.id == expectedGuid);
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
            catch (Exception error) { report["error"] = error.ToString(); throw; }
            finally { File.WriteAllText(output, report.ToString()); }
        }
    }
}
