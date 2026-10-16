using System;
using System.IO;
using System.Linq;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEngine;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class VisionStaticShapeMatmulGpuTests
    {
        // Native CPU evaluated all 2520 rows with the saved transposed-weight
        // layout. Only 16 independent rows are dispatched here; this is not
        // an encoder reference and cannot qualify the original full graph.
        [TestCase("mm_1")] [TestCase("mm_5")] [TestCase("mm_6")] [TestCase("mm_7")]
        public void SavedStaticShapeBlockOrderMustMatchNativePrefixBits(string node)
        {
            Assert.That(SystemInfo.supportsComputeShaders, Is.True, "Actual GPU required; no skip.");
            var source = Environment.GetEnvironmentVariable("EG_ACCEPTANCE_SOURCE");
            var root = Path.GetFullPath(Path.Combine(Application.dataPath, "../../numerical-acceptance-state/mm38734071287"));
            var destination = Path.GetFullPath(Path.Combine(Application.dataPath, "../artifacts/vision-static-shape-matmul-gpu-" + source + "-" + node + ".json"));
            Assert.That(File.Exists(destination), Is.False);
            var candidates = new JArray();
            var report = new JObject
            {
                ["success"] = false, ["runtime_source"] = source, ["node"] = node,
                ["reference_source"] = "750258ae760c1ed11674e8af8c354b949324d336",
                ["scope"] = "native_full_shape_static_gemm_synthetic_input_first_16_rows_only",
                ["reference_cpu_capability"] = "AVX2",
                ["original_full_encoder_reference_reproduced"] = false,
                ["full_encoder_validated"] = false, ["end_to_end_validated"] = false,
                ["candidates"] = candidates
            };
            void Save() { Directory.CreateDirectory(Path.GetDirectoryName(destination)); File.WriteAllText(destination, report.ToString()); }
            void Check(string path, string expected) { using var stream = File.OpenRead(path); Assert.That(PlayerFileHash.Compute(stream, out _), Is.EqualTo(expected)); }
            Save();
            try
            {
                var packages = JObject.Parse(File.ReadAllText(Path.Combine(Application.dataPath, "../Packages/packages-lock.json")));
                foreach (var package in new[] { "com.ayutaz.embeddinggemma", "com.ayutaz.embeddinggemma.validation" })
                    Assert.That(packages["dependencies"][package].Value<string>("hash"), Is.EqualTo(source));
                var manifest = Path.Combine(root, "matmul-reference.json");
                Check(manifest, "f43cfd94d4d451b108de28b7b792edacfc4ce8f934f3d654986308b8c48f1841");
                var reference = JObject.Parse(File.ReadAllText(manifest));
                Assert.That(reference.Value<bool>("native_patch_matches_original"), Is.True);
                Assert.That(reference.Value<bool>("full_encoder_executed"), Is.False);
                Assert.That(reference["runtime"].Value<string>("cpu_capability"), Is.EqualTo("AVX2"));
                var matrix = reference["matrices"].Single(row => row.Value<string>("node") == node);
                foreach (var key in new[] { "input", "weights", "expected" })
                    Check(Path.Combine(root, "captures", matrix[key].Value<string>("file")), matrix[key].Value<string>("sha256"));
                var shape = matrix["saved_input_shape"].Values<int>().ToArray();
                var weightsShape = matrix["weights"]["shape"].Values<int>().ToArray();
                Assert.That(shape[0], Is.EqualTo(16));
                Assert.That(matrix["native_input_shape"][0].Value<int>(), Is.EqualTo(2520));
                Assert.That(matrix["weight_stride"].Values<int>().ToArray(), Is.EqualTo(new[] { 1, weightsShape[0] }));
                var values = Floats(Path.Combine(root, "captures", matrix["input"].Value<string>("file")));
                var weights = Floats(Path.Combine(root, "captures", matrix["weights"].Value<string>("file")));
                var expected = Floats(Path.Combine(root, "captures", matrix["expected"].Value<string>("file")));
                using var input = new Tensor<float>(new TensorShape(shape), values); ComputeTensorData.Pin(input);
                foreach (var block in new[] { 96, 128, 192, 256, 384, 512, 768 })
                {
                    using var operation = new GpuBlockedMatmul(weightsShape[0], weightsShape[1], weights, block);
                    using var result = operation.Execute(input);
                    Assert.That(result.backendType, Is.EqualTo(BackendType.GPUCompute));
                    var actual = result.DownloadToArray();
                    Assert.That(actual.Length, Is.EqualTo(expected.Length));
                    var bytes = new byte[actual.Length * 4]; Buffer.BlockCopy(actual, 0, bytes, 0, bytes.Length);
                    var capture = Path.ChangeExtension(destination, null) + "-block" + block + ".bin";
                    File.WriteAllBytes(capture, bytes);
                    int differences = 0, violations = 0; double maximum = 0, ratio = 0;
                    for (var i = 0; i < actual.Length; i++)
                    {
                        Assert.That(float.IsNaN(actual[i]) || float.IsInfinity(actual[i]), Is.False);
                        if (BitConverter.SingleToInt32Bits(actual[i]) != BitConverter.SingleToInt32Bits(expected[i])) differences++;
                        var delta = Math.Abs((double)actual[i] - expected[i]); var bound = 1e-5 + 1e-4 * Math.Abs((double)expected[i]);
                        if (delta > bound) violations++; maximum = Math.Max(maximum, delta); ratio = Math.Max(ratio, delta / bound);
                    }
                    using var stream = new MemoryStream(bytes, false);
                    candidates.Add(new JObject { ["block_size"] = block, ["bit_differences"] = differences,
                        ["mismatched_elements"] = violations, ["maximum_absolute_error"] = maximum,
                        ["maximum_tolerance_ratio"] = ratio, ["capture_sha256"] = PlayerFileHash.Compute(stream, out _) });
                    Save();
                }
                Assert.That(input.DownloadToArray(), Is.EqualTo(values));
                report["borrowed_input_preserved"] = true; report["actual_backend"] = "GPUCompute";
                report["cpu_fallback_count"] = 0; report["intermediate_readback_count"] = 0;
                report["native_input_shape"] = matrix["native_input_shape"].DeepClone();
                report["weight_shape"] = matrix["weights"]["shape"].DeepClone();
                report["phase"] = "completed"; Save();
                Assert.That(candidates.Single(row => row.Value<int>("block_size") == 192).Value<int>("bit_differences"), Is.Zero,
                    "Current block192 must retain native shape/layout rounding; other candidates are diagnostics only.");
                report["success"] = true;
            }
            catch (Exception error) { report["error"] = error.ToString(); throw; }
            finally { Save(); }
        }
        static float[] Floats(string path)
        {
            var bytes = File.ReadAllBytes(path); Assert.That(bytes.Length % 4, Is.Zero);
            var values = new float[bytes.Length / 4]; Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length); return values;
        }
    }
}
