using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Security.Cryptography;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEditor;
using UnityEngine;

namespace EmbeddingGemma.Tests
{
    // Diagnostic readbacks only. Passing this fixture does not pass FP32 acceptance.
    public sealed class VisionDiagnosticReferenceTests
    {
        static string Root => Path.GetFullPath(Path.Combine(Application.dataPath, "../artifacts/vision-diagnostics"));
        static string OriginalRoot => Path.GetFullPath(Path.Combine(Application.dataPath, "../artifacts/vision-encoder"));
        static string ResultPath => Path.GetFullPath(Path.Combine(Root, "../vision-diagnostic-results/results.json"));
        JObject reference, originalReference, report;
        JArray conditions;
        string profile, savedTrace;

        [OneTimeSetUp]
        public void AuditAndPrepareOneTrace()
        {
            conditions = new JArray();
            report = new JObject { ["stage"] = "vision_sentis_diagnostics_only", ["success"] = false,
                ["known_fp32_failure_resolved"] = false, ["end_to_end_validated"] = false,
                ["gpu_preprocessing_validated"] = false, ["unity"] = Application.unityVersion,
                ["gpu"] = SystemInfo.graphicsDeviceName, ["graphics_api"] = SystemInfo.graphicsDeviceType.ToString(),
                ["conditions"] = conditions, ["phase"] = "reference_audit" };
            Save();
            reference = JObject.Parse(File.ReadAllText(Path.Combine(Root, "vision-diagnostics.json")));
            Assert.That(reference.Value<bool>("success"), Is.True);
            Assert.That(reference.Value<string>("stage"), Is.EqualTo("vision_encoder_diagnostics_only"));
            Assert.That(reference.Value<bool>("trace_matches_original_saved"), Is.True);
            Assert.That(reference["metadata"].Value<string>("model_revision"), Is.EqualTo("914f7f89142e33e77833254d9c9b90c3cef7303b"));
            Assert.That(reference["metadata"].Value<string>("source_commit"),
                Is.EqualTo(File.ReadAllText(Path.Combine(Root, "expected-source.txt")).Trim()));
            Assert.That(reference.Value<double>("feature_atol"), Is.EqualTo(1e-5));
            Assert.That(reference.Value<double>("feature_rtol"), Is.EqualTo(1e-4));
            Assert.That(((JArray)reference["cases"]).Count, Is.EqualTo(2));
            foreach (var file in (JObject)reference["files"])
            {
                var path = StagePath(file.Key);
                Assert.That(new FileInfo(path).Length, Is.EqualTo(file.Value.Value<long>("bytes")));
                Assert.That(Hash(path), Is.EqualTo(file.Value.Value<string>("sha256")));
            }
            originalReference = JObject.Parse(File.ReadAllText(Path.Combine(OriginalRoot, "vision-reference.json")));
            Assert.That(originalReference["metadata"].Value<string>("model_revision"),
                Is.EqualTo(reference["metadata"].Value<string>("model_revision")));
            profile = ((JObject)originalReference["profiles"]).Properties().First().Name;
            Assert.That(originalReference["profiles"][profile].Value<int>("feature_width"), Is.EqualTo(512));
            Assert.That(reference["metadata"].Value<int>("feature_count"),
                Is.EqualTo(originalReference["profiles"][profile].Value<int>("feature_count")));
            report["profile"] = profile; report["diagnostic_source"] = reference["metadata"]["source_commit"];
            report["original_reference_source"] = originalReference["metadata"]["source_commit"];
            report["phase"] = "trace_import"; Save();
            var assetPath = "Assets/VisionDiagnosticGenerated/vision-stages.pt2";
            Directory.CreateDirectory(Path.GetDirectoryName(assetPath));
            File.Copy(StagePath("vision-stages.pt2"), assetPath, true);
            AssetDatabase.ImportAsset(assetPath, ImportAssetOptions.ForceSynchronousImport);
            var asset = AssetDatabase.LoadAssetAtPath<ModelAsset>(assetPath);
            Assert.That(asset, Is.Not.Null);
            var model = ModelLoader.Load(asset);
            Assert.That(model.outputs.Count, Is.EqualTo(((JArray)reference["stages"]).Count));
            savedTrace = StagePath("vision-stages-fp32.sentis");
            TextModelFile.Save(model, savedTrace);
            report["trace_saved_sha256"] = Hash(savedTrace);
            report["trace_saved_bytes"] = new FileInfo(savedTrace).Length;
            report["phase"] = "prepared"; Save();
        }

        [OneTimeTearDown]
        public void CompleteReport()
        {
            if (report == null) return;
            report["success"] = conditions.Count == 2 && conditions.All(value => value.Value<bool>("success"));
            report["completed_utc"] = DateTime.UtcNow.ToString("O"); Save();
        }

        [TestCase(BackendType.CPU)]
        [TestCase(BackendType.GPUCompute)]
        [Timeout(1200000)]
        public void RealStagesLocalizeOnlyTheUnchangedOriginalFailure(BackendType backend)
        {
            var condition = new JObject { ["requested_backend"] = backend.ToString(), ["success"] = false };
            var rows = new JArray(); condition["cases"] = rows; conditions.Add(condition); Save();
            try
            {
                if (backend == BackendType.GPUCompute) Assert.That(SystemInfo.supportsComputeShaders, Is.True, "Never skip actual GPU.");
                var originalPath = Path.Combine(OriginalRoot, profile + "-fp32.sentis");
                var prior = JObject.Parse(File.ReadAllText(Path.Combine(OriginalRoot, "../m3-vision/results.json")));
                var priorCondition = ((JArray)prior["conditions"]).Single(value => value.Value<string>("profile") == profile
                    && value.Value<string>("precision") == "fp32" && value.Value<string>("requested_backend") == backend.ToString());
                Assert.That(priorCondition.Value<double>("maximum_fp32_tolerance_ratio"), Is.GreaterThan(1));
                Assert.That(Hash(originalPath), Is.EqualTo(priorCondition.Value<string>("saved_sha256")));
                condition["original_saved_sha256"] = priorCondition["saved_sha256"];
                var originalModel = TextModelFile.Load(originalPath);
                var traceModel = TextModelFile.Load(savedTrace);
                using var original = new Worker(originalModel, backend);
                using var traced = new Worker(traceModel, backend);
                condition["original_backend_audit"] = Audit(original, originalModel, backend);
                condition["trace_backend_audit"] = Audit(traced, traceModel, backend);
                var oldCases = ((JArray)originalReference["cases"]).Where(value => value.Value<string>("profile") == profile).ToArray();
                Assert.That(oldCases.Length, Is.EqualTo(2));
                var cases = (JArray)reference["cases"];
                for (var caseIndex = 0; caseIndex < cases.Count; caseIndex++)
                {
                    var spec = cases[caseIndex];
                    var item = new JObject { ["id"] = spec.Value<string>("id"), ["localization_validated"] = false };
                    var stages = new JArray(); item["stages"] = stages; rows.Add(item); Save();
                    var pixelsPath = StagePath(spec.Value<string>("pixels_file"));
                    Assert.That(Hash(pixelsPath), Is.EqualTo(Hash(Path.Combine(OriginalRoot, oldCases[caseIndex].Value<string>("pixels")))));
                    using var input = new Tensor<float>(new TensorShape(reference["metadata"]["pixel_shape"].ToObject<int[]>()), Read(pixelsPath));
                    if (backend == BackendType.GPUCompute) ComputeTensorData.Pin(input);
                    original.Schedule(input);
                    var oldOutput = original.PeekOutput() as Tensor<float>;
                    Assert.That(oldOutput.backendType, Is.EqualTo(backend));
                    var oldValues = oldOutput.DownloadToArray();
                    traced.Schedule(input);
                    var final = traced.PeekOutput(traceModel.outputs.Last().name) as Tensor<float>;
                    Assert.That(final.backendType, Is.EqualTo(backend));
                    var finalValues = final.DownloadToArray(); // All inference finishes before diagnostic reads.
                    var golden = oldCases[caseIndex]["features"].ToObject<float[][]>().SelectMany(value => value).ToArray();
                    var specs = (JArray)spec["stages"];
                    var nativeFinal = Read(StagePath(specs.Last().Value<string>("reference_file")));
                    var context = VisionDiagnosticAnalysis.AnalyzePinnedReference(golden, oldValues, finalValues, nativeFinal,
                        spec.Value<bool>("trace_matches_original_saved"));
                    item["original_mismatched_elements"] = context.Original.Mismatches;
                    item["original_maximum_tolerance_ratio"] = context.Original.MaximumToleranceRatio;
                    item["trace_matches_original_saved"] = context.TraceMatchesOriginal;
                    item["native_reference_matches_pinned"] = context.NativeReference.Passed;
                    item["native_reference_mismatched_elements"] = context.NativeReference.Mismatches;
                    item["native_reference_maximum_tolerance_ratio"] = context.NativeReference.MaximumToleranceRatio;
                    Save();
                    Assert.That(context.NativeReference.Passed, Is.True, "New native final must fit the original pinned reference.");
                    string firstNonzero = null, firstFailure = null;
                    for (var stageIndex = 0; stageIndex < specs.Count; stageIndex++)
                    {
                        var stage = specs[stageIndex]; var output = traceModel.outputs[stageIndex];
                        var tensor = traced.PeekOutput(output.name) as Tensor<float>;
                        Assert.That(tensor.shape, Is.EqualTo(new TensorShape(stage["shape"].ToObject<int[]>())));
                        var constant = traceModel.constants.Any(value => value.index == output.index);
                        if (!constant) Assert.That(tensor.backendType, Is.EqualTo(backend));
                        var native = Read(StagePath(stage.Value<string>("reference_file")));
                        var metrics = VisionDiagnosticAnalysis.Compare(native, tensor.DownloadToArray());
                        var name = stage.Value<string>("name");
                        if (firstNonzero == null && metrics.MaximumAbsoluteError > 0) firstNonzero = name;
                        if (firstFailure == null && !metrics.Passed) firstFailure = name;
                        stages.Add(new JObject { ["name"] = name, ["shape"] = stage["shape"].DeepClone(),
                            ["constant_output"] = constant, ["actual_backend"] = tensor.backendType.ToString(),
                            ["maximum_absolute_error"] = metrics.MaximumAbsoluteError,
                            ["maximum_tolerance_ratio"] = metrics.MaximumToleranceRatio, ["mismatched_elements"] = metrics.Mismatches });
                        Save();
                    }
                    item["first_nonzero_stage"] = firstNonzero;
                    item["first_divergent_stage"] = firstFailure;
                    item["localization_validated"] = context.CanLocalize; Save();
                }
                Assert.That(rows.All(value => value.Value<bool>("trace_matches_original_saved")), Is.True, "Diagnostic graph changed the original Sentis path.");
                Assert.That(rows.Any(value => value.Value<bool>("localization_validated")), Is.True, "Original failed FP32 output was not reproduced.");
                condition["success"] = true;
            }
            catch (Exception error) { condition["error"] = error.ToString(); throw; }
            finally { Save(); }
        }

        static JObject Audit(Worker worker, Model model, BackendType backend)
        {
            Assert.That(worker.backendType, Is.EqualTo(backend));
            const BindingFlags flags = BindingFlags.Instance | BindingFlags.NonPublic;
            var resident = (HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallback", flags).GetValue(worker);
            var cpu = resident.Intersect(model.layers.SelectMany(layer => layer.outputs).Concat(model.inputs.Select(value => value.index))).ToArray();
            var flush = (HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallbackShouldFlushGPU", flags).GetValue(worker);
            if (backend == BackendType.GPUCompute) { Assert.That(cpu, Is.Empty); Assert.That(flush, Is.Empty); }
            return new JObject { ["actual_backend"] = worker.backendType.ToString(), ["cpu_computation_or_inputs"] = new JArray(cpu), ["readback_flush"] = new JArray(flush) };
        }

        static float[] Read(string path)
        {
            Assert.That(BitConverter.IsLittleEndian, Is.True);
            var bytes = File.ReadAllBytes(path); Assert.That(bytes.Length % 4, Is.EqualTo(0));
            var values = new float[bytes.Length / 4]; Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length); return values;
        }
        static string StagePath(string name)
        {
            var path = Path.GetFullPath(Path.Combine(Root, name));
            Assert.That(path.StartsWith(Root + Path.DirectorySeparatorChar, StringComparison.Ordinal), Is.True); return path;
        }
        static string Hash(string path)
        {
            var helper = AppDomain.CurrentDomain.GetAssemblies().Select(value => value.GetType("EmbeddingGemma.Samples.TextSearchFileHash"))
                .FirstOrDefault(value => value != null);
            if (helper != null) return (string)helper.GetMethod("Compute").Invoke(null, new object[] { path, null });
            using var stream = File.OpenRead(path); using var hash = SHA256.Create();
            return BitConverter.ToString(hash.ComputeHash(stream)).Replace("-", "").ToLowerInvariant();
        }
        void Save()
        {
            Directory.CreateDirectory(Path.GetDirectoryName(ResultPath)); File.WriteAllText(ResultPath, report.ToString());
        }
    }
}
