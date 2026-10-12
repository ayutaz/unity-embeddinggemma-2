using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Security.Cryptography;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEditor;
using UnityEngine;

namespace EmbeddingGemma.Tests
{
    internal sealed class VisionFixtureCache<T>
    {
        readonly Dictionary<string, T> prepared = new Dictionary<string, T>();
        internal T Get(string key, Func<T> prepare)
        {
            if (!prepared.TryGetValue(key, out var value)) prepared.Add(key, value = prepare());
            return value;
        }
        internal void Clear() => prepared.Clear();
    }

    internal static class VisionBackendAudit
    {
        internal static int[] CpuComputationOrInputIndices(HashSet<int> resident,
            IEnumerable<int> computationOutputs, IEnumerable<int> modelInputs)
        {
            return resident.Intersect(computationOutputs.Concat(modelInputs)).OrderBy(index => index).ToArray();
        }
    }

    public sealed class VisionBackendAuditTests
    {
        [Test]
        public void CpuAndGpuReuseOnePreparationForSameProfileAndPrecision()
        {
            var cache = new VisionFixtureCache<int>();
            var calls = 0;
            var cpu = cache.Get("square/fp32", () => ++calls);
            var gpu = cache.Get("square/fp32", () => ++calls);
            Assert.That(gpu, Is.EqualTo(cpu));
            Assert.That(calls, Is.EqualTo(1));
        }

        [Test]
        public void DifferentPrecisionOrProfileDoesNotReusePreparedArtifact()
        {
            var cache = new VisionFixtureCache<int>();
            var calls = 0;
            Assert.That(cache.Get("square/fp32", () => ++calls), Is.EqualTo(1));
            Assert.That(cache.Get("square/float16", () => ++calls), Is.EqualTo(2));
            Assert.That(cache.Get("portrait/fp32", () => ++calls), Is.EqualTo(3));
        }

        [Test]
        public void NewFixtureAuditStartsWithFreshPreparation()
        {
            var cache = new VisionFixtureCache<int>();
            var calls = 0;
            cache.Get("square/fp32", () => ++calls);
            cache.Clear();
            Assert.That(cache.Get("square/fp32", () => ++calls), Is.EqualTo(2));
        }

        [Test]
        public void CpuShapeConstantsDoNotClaimCpuComputation()
        {
            Assert.That(VisionBackendAudit.CpuComputationOrInputIndices(new HashSet<int> { 7, 12 },
                new[] { 90, 91 }, new[] { 0 }), Is.Empty);
        }

        [Test]
        public void ActualCpuProducedTensorIsRejectedEvenAmongConstants()
        {
            Assert.That(VisionBackendAudit.CpuComputationOrInputIndices(new HashSet<int> { 7, 12, 90 },
                new[] { 90, 91 }, new[] { 0 }), Is.EqualTo(new[] { 90 }));
        }

        [Test]
        public void InputRequiringCpuDataCannotPassGpuChainAudit()
        {
            Assert.That(VisionBackendAudit.CpuComputationOrInputIndices(new HashSet<int> { 0 },
                new[] { 90, 91 }, new[] { 0 }), Is.EqualTo(new[] { 0 }));
        }
    }

    // Acceptance of the already TDD-tested Python adapter against real Sentis.
    // Feature readback is diagnostic here; production GPU preprocessing and
    // final backbone integration are separate gates.
    public sealed class VisionReferenceTests
    {
        static string DirectoryPath => Path.GetFullPath(Path.Combine(Application.dataPath, "../artifacts/vision-encoder"));
        JObject reference;
        JObject report;
        JArray conditions;
        readonly VisionFixtureCache<ModelAsset> importedAssets = new VisionFixtureCache<ModelAsset>();
        readonly VisionFixtureCache<SavedVision> preparedModels = new VisionFixtureCache<SavedVision>();

        sealed class SavedVision
        {
            internal string Path;
            internal string Hash;
            internal long Bytes;
        }

        [OneTimeSetUp]
        public void PrepareReportAndAuditCompletedArtifact()
        {
            importedAssets.Clear();
            preparedModels.Clear();
            conditions = new JArray();
            report = new JObject { ["success"] = false, ["stage"] = "vision_encoder_projector_only",
                ["end_to_end_validated"] = false, ["gpu_preprocessing_validated"] = false,
                ["expected_conditions"] = 12, ["unity"] = Application.unityVersion,
                ["gpu"] = SystemInfo.graphicsDeviceName, ["graphics_api"] = SystemInfo.graphicsDeviceType.ToString(),
                ["started_utc"] = DateTime.UtcNow.ToString("O"), ["conditions"] = conditions };
            Save();
            var expectedSource = File.ReadAllText(Path.Combine(DirectoryPath, "expected-source.txt")).Trim();
            Assert.That(expectedSource, Does.Match("^[0-9a-f]{40}$"));
            var validation = JObject.Parse(File.ReadAllText(Path.Combine(DirectoryPath, "vision-validation.json")));
            Assert.That(validation.Value<bool>("success"), Is.True);
            Assert.That(validation.Value<string>("stage"), Is.EqualTo("vision_encoder_projector_only"));
            Assert.That(validation.Value<int>("profile_count"), Is.EqualTo(3));
            Assert.That(validation.Value<int>("case_count"), Is.EqualTo(6));
            foreach (var file in (JObject)validation["files"])
            {
                var path = Path.GetFullPath(Path.Combine(DirectoryPath, file.Key));
                Assert.That(path.StartsWith(DirectoryPath + Path.DirectorySeparatorChar, StringComparison.Ordinal), Is.True);
                Assert.That(new FileInfo(path).Length, Is.EqualTo(file.Value.Value<long>("bytes")), file.Key);
                Assert.That(Hash(path), Is.EqualTo(file.Value.Value<string>("sha256")), file.Key);
            }
            reference = JObject.Parse(File.ReadAllText(Path.Combine(DirectoryPath, "vision-reference.json")));
            Assert.That(reference["metadata"].Value<string>("model_revision"), Is.EqualTo("914f7f89142e33e77833254d9c9b90c3cef7303b"));
            Assert.That(reference["metadata"].Value<string>("source_commit"), Is.EqualTo(expectedSource));
            report["reference_source"] = expectedSource;
            Save();
        }

        [OneTimeTearDown]
        public void CompleteReport()
        {
            if (report == null) return;
            report["success"] = conditions.Count == 12 && conditions.All(row => row.Value<bool>("success"));
            report["completed_utc"] = DateTime.UtcNow.ToString("O");
            Save();
        }

        [TestCase("square", "fp32", BackendType.CPU)]
        [TestCase("square", "fp32", BackendType.GPUCompute)]
        [TestCase("square", "float16", BackendType.CPU)]
        [TestCase("square", "float16", BackendType.GPUCompute)]
        [TestCase("landscape", "fp32", BackendType.CPU)]
        [TestCase("landscape", "fp32", BackendType.GPUCompute)]
        [TestCase("landscape", "float16", BackendType.CPU)]
        [TestCase("landscape", "float16", BackendType.GPUCompute)]
        [TestCase("portrait", "fp32", BackendType.CPU)]
        [TestCase("portrait", "fp32", BackendType.GPUCompute)]
        [TestCase("portrait", "float16", BackendType.CPU)]
        [TestCase("portrait", "float16", BackendType.GPUCompute)]
        [Timeout(1200000)]
        public void SavedSentisMatchesEveryOfficialProfileFeature(string profileName, string precision, BackendType backend)
        {
            var condition = new JObject { ["profile"] = profileName, ["precision"] = precision,
                ["requested_backend"] = backend.ToString(), ["success"] = false };
            conditions.Add(condition);
            Save();
            try
            {
                if (backend == BackendType.GPUCompute)
                    Assert.That(SystemInfo.supportsComputeShaders, Is.True, "Actual GPU execution is required; never skip.");
                var profile = reference["profiles"][profileName];
                var shape = profile["pixel_shape"].ToObject<int[]>();
                var count = profile.Value<int>("feature_count");
                Assert.That(shape, Is.EqualTo(new[] { 1, 2520, 768 }));
                Assert.That(profile.Value<int>("feature_width"), Is.EqualTo(512), "Tiny models cannot satisfy this gate.");
                Assert.That(count, Is.InRange(1, 280));
                var saved = preparedModels.Get(profileName + "/" + precision, () => PrepareSavedModel(profileName, precision, profile));
                condition["saved_bytes"] = saved.Bytes;
                condition["saved_sha256"] = saved.Hash;
                var loaded = TextModelFile.Load(saved.Path);
                using var worker = new Worker(loaded, backend);
                condition["actual_worker_backend"] = worker.backendType.ToString();
                Assert.That(worker.backendType, Is.EqualTo(backend));
                var fallbackField = typeof(Worker).GetField("m_LayerCPUFallback", BindingFlags.NonPublic | BindingFlags.Instance);
                Assert.That(fallbackField, Is.Not.Null, "Pinned backend fallback audit must remain available.");
                var fallbacks = (HashSet<int>)fallbackField.GetValue(worker);
                condition["cpu_resident_tensor_indices"] = new JArray(fallbacks.OrderBy(index => index));
                var actualCpu = VisionBackendAudit.CpuComputationOrInputIndices(fallbacks,
                    loaded.layers.SelectMany(layer => layer.outputs), loaded.inputs.Select(input => input.index));
                condition["cpu_computation_or_input_indices"] = new JArray(actualCpu);
                var flushField = typeof(Worker).GetField("m_LayerCPUFallbackShouldFlushGPU", BindingFlags.NonPublic | BindingFlags.Instance);
                Assert.That(flushField, Is.Not.Null);
                var flushes = (HashSet<int>)flushField.GetValue(worker);
                condition["cpu_readback_flush_indices"] = new JArray(flushes.OrderBy(index => index));
                if (backend == BackendType.GPUCompute)
                {
                    Assert.That(actualCpu, Is.Empty, "No CPU computation or input readback may count as GPU pass.");
                    Assert.That(flushes, Is.Empty, "No intermediate GPU-to-CPU transfer may count as GPU pass.");
                }
                var rows = ((JArray)reference["cases"]).Where(row => row.Value<string>("profile") == profileName).ToArray();
                Assert.That(rows, Has.Length.EqualTo(2));
                double minimumCosine = 1, maximumAbsoluteError = 0, maximumToleranceRatio = 0;
                foreach (var row in rows)
                {
                    var bytes = File.ReadAllBytes(Path.Combine(DirectoryPath, row.Value<string>("pixels")));
                    Assert.That(BitConverter.IsLittleEndian, Is.True);
                    Assert.That(bytes.Length, Is.EqualTo(shape.Aggregate(1, (left, right) => left * right) * 4));
                    var pixels = new float[bytes.Length / 4];
                    Buffer.BlockCopy(bytes, 0, pixels, 0, bytes.Length);
                    using var input = new Tensor<float>(new TensorShape(shape), pixels);
                    worker.Schedule(input);
                    var output = worker.PeekOutput() as Tensor<float>;
                    Assert.That(output, Is.Not.Null);
                    condition["actual_output_backend"] = output.backendType.ToString();
                    Assert.That(output.backendType, Is.EqualTo(backend));
                    Assert.That(output.shape, Is.EqualTo(new TensorShape(count, 512)));
                    using var readable = output.ReadbackAndClone();
                    var actual = readable.DownloadToArray();
                    var expected = row["features"].ToObject<float[][]>();
                    Assert.That(expected, Has.Length.EqualTo(count));
                    for (var feature = 0; feature < count; feature++)
                    {
                        Assert.That(expected[feature], Has.Length.EqualTo(512));
                        double dot = 0, leftSquared = 0, rightSquared = 0;
                        for (var channel = 0; channel < 512; channel++)
                        {
                            var left = actual[feature * 512 + channel]; var right = expected[feature][channel];
                            Assert.That(float.IsNaN(left) || float.IsInfinity(left), Is.False, row.Value<string>("id"));
                            dot += (double)left * right; leftSquared += (double)left * left; rightSquared += (double)right * right;
                            var difference = Math.Abs((double)left - right);
                            maximumAbsoluteError = Math.Max(maximumAbsoluteError, difference);
                            maximumToleranceRatio = Math.Max(maximumToleranceRatio, difference / (1e-5 + 1e-4 * Math.Abs(right)));
                        }
                        Assert.That(leftSquared * rightSquared, Is.GreaterThan(0));
                        minimumCosine = Math.Min(minimumCosine, dot / Math.Sqrt(leftSquared * rightSquared));
                    }
                }
                condition["minimum_cosine"] = minimumCosine;
                condition["maximum_absolute_error"] = maximumAbsoluteError;
                condition["maximum_fp32_tolerance_ratio"] = maximumToleranceRatio;
                Assert.That(minimumCosine, Is.GreaterThanOrEqualTo(precision == "fp32" ? 0.999 : 0.99));
                if (precision == "fp32") Assert.That(maximumToleranceRatio, Is.LessThanOrEqualTo(1), "Pinned elementwise tolerance.");
                condition["success"] = true;
            }
            catch (Exception error) { condition["error"] = error.ToString(); throw; }
            finally { Save(); }
        }

        SavedVision PrepareSavedModel(string profileName, string precision, JToken profile)
        {
            var asset = importedAssets.Get(profileName, () =>
            {
                var assetPath = "Assets/M3Generated/" + profileName + ".pt2";
                var sourcePath = Path.Combine(DirectoryPath, profile.Value<string>("graph"));
                Directory.CreateDirectory(Path.GetDirectoryName(assetPath));
                if (!File.Exists(assetPath) || Hash(assetPath) != Hash(sourcePath)) File.Copy(sourcePath, assetPath, true);
                AssetDatabase.ImportAsset(assetPath, ImportAssetOptions.ForceSynchronousImport);
                var imported = AssetDatabase.LoadAssetAtPath<ModelAsset>(assetPath);
                Assert.That(imported, Is.Not.Null, "Sentis must import the real graph.");
                return imported;
            });
            var model = ModelLoader.Load(asset);
            if (precision == "float16") ModelQuantizer.QuantizeWeights(QuantizationType.Float16, ref model);
            var path = Path.Combine(DirectoryPath, profileName + "-" + precision + ".sentis");
            TextModelFile.Save(model, path);
            var saved = new SavedVision { Path = path, Bytes = new FileInfo(path).Length, Hash = Hash(path) };
            model = null;
            GC.Collect();
            GC.WaitForPendingFinalizers();
            return saved;
        }

        void Save()
        {
            var resultDirectory = Path.GetFullPath(Path.Combine(Application.dataPath, "../artifacts/m3-vision"));
            Directory.CreateDirectory(resultDirectory);
            File.WriteAllText(Path.Combine(resultDirectory, "results.json"), report.ToString(Formatting.Indented));
        }

        static string Hash(string path)
        {
            // Reuse the audited sample hash implementation when present, without
            // introducing a required dependency on the optional sample assembly.
            var existing = AppDomain.CurrentDomain.GetAssemblies().Select(assembly => assembly.GetType("EmbeddingGemma.Samples.TextSearchFileHash"))
                .FirstOrDefault(type => type != null);
            if (existing != null) return (string)existing.GetMethod("Compute").Invoke(null, new object[] { path, null });
            using var stream = File.OpenRead(path);
            using var hash = SHA256.Create();
            return BitConverter.ToString(hash.ComputeHash(stream)).Replace("-", "").ToLowerInvariant();
        }
    }
}
