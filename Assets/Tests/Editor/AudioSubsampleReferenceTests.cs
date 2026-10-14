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
    // Real weights of the subsample only. Does not accept the full encoder.
    public sealed class AudioSubsampleReferenceTests
    {
        string root, savedPath, resultPath;
        JObject reference, report;
        JArray conditions;

        [OneTimeSetUp]
        public void AuditPinnedSubsampleAndImportOnce()
        {
            root = Path.GetFullPath(Path.Combine(Application.dataPath, "../artifacts/audio-subsample"));
            resultPath = Path.Combine(root, "../audio-subsample-results/results.json");
            conditions = new JArray();
            report = new JObject { ["success"] = false, ["scope"] = "real_subsample_only",
                ["full_encoder_validated"] = false, ["end_to_end_validated"] = false,
                ["unity"] = Application.unityVersion, ["gpu"] = SystemInfo.graphicsDeviceName,
                ["graphics_api"] = SystemInfo.graphicsDeviceType.ToString(), ["conditions"] = conditions };
            Save();
            reference = JObject.Parse(File.ReadAllText(Path.Combine(root, "subsample-reference.json")));
            var source = File.ReadAllText(Path.Combine(root, "expected-source.txt")).Trim();
            Assert.That(source, Does.Match("^[0-9a-f]{40}$"));
            Assert.That(reference.Value<string>("source_commit"), Is.EqualTo(source));
            Assert.That(reference.Value<string>("scope"), Is.EqualTo("real_subsample_only"));
            Assert.That(reference["diagnostic"].Value<bool>("trace_matches_full_subsample"), Is.True);
            Assert.That(reference["diagnostic"].Value<bool>("native_reference_matches_full_subsample"), Is.True);
            Assert.That(BitConverter.IsLittleEndian, Is.True);
            foreach (var entry in (JObject)reference["diagnostic"]["files"])
            {
                var path = Resolve(entry.Value.Value<string>("path"));
                Assert.That(new FileInfo(path).Length, Is.EqualTo(entry.Value.Value<long>("bytes")));
                Assert.That(Hash(path), Is.EqualTo(entry.Value.Value<string>("sha256")));
            }
            report["reference_source"] = source;
            const string assetPath = "Assets/AudioSubsampleGenerated/subsample.onnx";
            Directory.CreateDirectory(Path.GetDirectoryName(assetPath));
            File.Copy(Artifact("graph"), assetPath, true);
            AssetDatabase.ImportAsset(assetPath, ImportAssetOptions.ForceSynchronousImport);
            var asset = AssetDatabase.LoadAssetAtPath<ModelAsset>(assetPath);
            Assert.That(asset, Is.Not.Null);
            var model = ModelLoader.Load(asset);
            Assert.That(model.inputs.Count, Is.EqualTo(1));
            Assert.That(model.outputs.Count, Is.EqualTo(10));
            savedPath = Path.Combine(root, "subsample-fp32.sentis");
            TextModelFile.Save(model, savedPath);
            report["operators"] = new JArray(model.layers.Select(layer => layer.GetType().Name).Distinct());
            Save();
        }

        [TestCase(BackendType.CPU)]
        [TestCase(BackendType.GPUCompute)]
        [Timeout(300000)]
        public void SavedRealSubsampleMatchesPinnedNativeOutput(BackendType backend)
        {
            var condition = new JObject { ["requested_backend"] = backend.ToString(), ["success"] = false };
            conditions.Add(condition); Save();
            try
            {
                if (backend == BackendType.GPUCompute)
                    Assert.That(SystemInfo.supportsComputeShaders, Is.True, "GPU must execute; do not skip.");
                var model = TextModelFile.Load(savedPath);
                using var worker = new Worker(model, backend);
                Assert.That(worker.backendType, Is.EqualTo(backend));
                const BindingFlags flags = BindingFlags.Instance | BindingFlags.NonPublic;
                var resident = (HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallback", flags).GetValue(worker);
                var cpu = resident.Intersect(model.layers.SelectMany(layer => layer.outputs)
                    .Concat(model.inputs.Select(input => input.index))).ToArray();
                var flush = (HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallbackShouldFlushGPU", flags).GetValue(worker);
                condition["cpu_computation_or_inputs"] = new JArray(cpu);
                condition["readback_flush"] = new JArray(flush);
                if (backend == BackendType.GPUCompute) { Assert.That(cpu, Is.Empty); Assert.That(flush, Is.Empty); }
                using var input = new Tensor<float>(new TensorShape(reference["feature_shape"].ToObject<int[]>()), Floats(Artifact("native_input")));
                if (backend == BackendType.GPUCompute) ComputeTensorData.Pin(input);
                worker.Schedule(input);
                var output = worker.PeekOutput(model.outputs.Last().name) as Tensor<float>;
                Assert.That(output, Is.Not.Null);
                Assert.That(output.shape, Is.EqualTo(new TensorShape(reference["output_shape"].ToObject<int[]>())));
                condition["actual_backend"] = output.backendType.ToString();
                Assert.That(output.backendType, Is.EqualTo(backend));
                var values = output.DownloadToArray();
                var expected = Floats(Artifact("native_subsample"));
                Assert.That(values.Length, Is.EqualTo(expected.Length));
                int mismatches = 0, exact = 0;
                double maximum = 0, ratio = 0;
                for (var index = 0; index < values.Length; index++)
                {
                    Assert.That(float.IsNaN(values[index]) || float.IsInfinity(values[index]), Is.False);
                    var difference = Math.Abs((double)values[index] - expected[index]);
                    var bound = 1e-5 + 1e-4 * Math.Abs((double)expected[index]);
                    maximum = Math.Max(maximum, difference); ratio = Math.Max(ratio, difference / bound);
                    if (difference > bound) mismatches++;
                    if (values[index] != expected[index]) exact++;
                }
                condition["maximum_absolute_error"] = maximum;
                condition["maximum_tolerance_ratio"] = ratio;
                condition["mismatched_elements"] = mismatches;
                condition["exact_value_mismatches"] = exact;
                Save(); Assert.That(mismatches, Is.Zero);
                condition["success"] = true;
            }
            catch (Exception error) { condition["error"] = error.ToString(); throw; }
            finally { Save(); }
        }

        [OneTimeTearDown]
        public void Complete()
        {
            if (report == null) return;
            report["success"] = conditions.Count == 2 && conditions.All(item => item.Value<bool>("success"));
            report["completed_utc"] = DateTime.UtcNow.ToString("O"); Save();
        }
        string Resolve(string name)
        {
            var path = Path.GetFullPath(Path.Combine(root, name));
            Assert.That(path.StartsWith(root + Path.DirectorySeparatorChar, StringComparison.Ordinal), Is.True);
            return path;
        }
        string Artifact(string kind) => Resolve(reference["diagnostic"]["files"][kind].Value<string>("path"));
        static float[] Floats(string path)
        {
            var bytes = System.IO.File.ReadAllBytes(path);
            Assert.That(bytes.Length % sizeof(float), Is.Zero);
            var values = new float[bytes.Length / sizeof(float)];
            Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length); return values;
        }
        static string Hash(string path)
        {
            using var input = System.IO.File.OpenRead(path); using var hash = SHA256.Create();
            return BitConverter.ToString(hash.ComputeHash(input)).Replace("-", "").ToLowerInvariant();
        }
        void Save()
        {
            Directory.CreateDirectory(Path.GetDirectoryName(resultPath));
            System.IO.File.WriteAllText(resultPath, report.ToString());
        }
    }
}
