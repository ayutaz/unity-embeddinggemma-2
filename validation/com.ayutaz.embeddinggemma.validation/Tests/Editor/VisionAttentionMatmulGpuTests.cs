using System;
using System.IO;
using System.Linq;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEngine;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class VisionAttentionMatmulGpuTests
    {
        [TestCase(0)] [TestCase(1)]
        public void CapturedAttentionRowsMatchIndependentCurrentCpuMatmulOnGpu(int ordinal)
        {
            Assert.That(SystemInfo.supportsComputeShaders, Is.True, "Actual GPU required; no skip.");
            var source = Environment.GetEnvironmentVariable("EG_ACCEPTANCE_SOURCE"); var root = Path.Combine(Application.dataPath, "GpuVisionAttentionMatmulReference");
            var output = Path.GetFullPath(Path.Combine(Application.dataPath, "../artifacts/vision-attention-matmul-equation-gpu-" + source + "-" + ordinal + ".json"));
            Assert.That(File.Exists(output), Is.False);
            var report = new JObject { ["success"] = false, ["runtime_source"] = source, ["ordinal"] = ordinal,
                ["scope"] = "first_16_query_rows_captured_gpu_attention_matmul_current_cpu_equation_only",
                ["original_internal_native_reference_available"] = false, ["full_encoder_validated"] = false, ["end_to_end_validated"] = false };
            void Save() { Directory.CreateDirectory(Path.GetDirectoryName(output)); File.WriteAllText(output, report.ToString()); }
            void Check(string path, string expected) { using var stream = File.OpenRead(path); Assert.That(PlayerFileHash.Compute(stream, out _), Is.EqualTo(expected)); }
            Save();
            try
            {
                var locked = JObject.Parse(File.ReadAllText(Path.Combine(Application.dataPath, "../Packages/packages-lock.json")));
                foreach (var package in new[] { "com.ayutaz.embeddinggemma", "com.ayutaz.embeddinggemma.validation" })
                    Assert.That(locked["dependencies"][package].Value<string>("hash"), Is.EqualTo(source));
                var manifest = Path.Combine(root, "matmul-reference.json"); Check(manifest, "de8c267ed994afbeeea3388f767521e38eb935a59a5c800feb98ab7dc06d7579");
                var reference = JObject.Parse(File.ReadAllText(manifest));
                foreach (var key in new[] { "original_layer_model", "original_patch_input", "capture" })
                    Check(reference[key].Value<string>("file"), reference[key].Value<string>("sha256"));
                var row = reference["rows"].Single(value => value.Value<int>("ordinal") == ordinal);
                foreach (var key in new[] { "left", "right", "captured_left_source", "captured_output", "expected" })
                    Check(Path.Combine(root, row[key].Value<string>("file")), row[key].Value<string>("sha256"));
                var a = Floats(Path.Combine(root, row["left"].Value<string>("file"))); var b = Floats(Path.Combine(root, row["right"].Value<string>("file")));
                var expected = Floats(Path.Combine(root, row["expected"].Value<string>("file")));
                using var left = new Tensor<float>(new TensorShape(row["left"]["shape"].Values<int>().ToArray()), a); ComputeTensorData.Pin(left);
                using var right = new Tensor<float>(new TensorShape(row["right"]["shape"].Values<int>().ToArray()), b); ComputeTensorData.Pin(right);
                using var operation = new GpuBatchedMatmul(); using var result = operation.Execute(left, right);
                Assert.That(result.backendType, Is.EqualTo(BackendType.GPUCompute)); Assert.That(result.shape, Is.EqualTo(new TensorShape(row["expected"]["shape"].Values<int>().ToArray())));
                var actual = result.DownloadToArray(); report["baseline_captured_full_shape"] = Metrics(Floats(row["captured_output"].Value<string>("file")), expected); report["candidate"] = Metrics(actual, expected);
                var raw = new byte[actual.Length * 4]; Buffer.BlockCopy(actual, 0, raw, 0, raw.Length); File.WriteAllBytes(Path.ChangeExtension(output, ".bin"), raw);
                using (var stream = new MemoryStream(raw, false)) report["capture_sha256"] = PlayerFileHash.Compute(stream, out _);
                Assert.That(left.DownloadToArray(), Is.EqualTo(a)); Assert.That(right.DownloadToArray(), Is.EqualTo(b));
                report["borrowed_inputs_preserved"] = true; report["actual_backend"] = "GPUCompute"; report["cpu_computation_or_input_count"] = 0; report["readback_flush_count"] = 0;
                report["reference_sha256"] = "de8c267ed994afbeeea3388f767521e38eb935a59a5c800feb98ab7dc06d7579"; Save();
                Assert.That(report["candidate"].Value<int>("mismatched_elements"), Is.Zero, "Keep the unchanged FP32 equation tolerance."); report["success"] = true;
            }
            catch (Exception error) { report["error"] = error.ToString(); throw; }
            finally { Save(); }
        }
        static JObject Metrics(float[] actual, float[] expected)
        {
            Assert.That(actual.Length, Is.EqualTo(expected.Length)); Assert.That(Array.TrueForAll(actual, value => !float.IsNaN(value) && !float.IsInfinity(value)), Is.True);
            int bits = 0, bad = 0; double maximum = 0, ratio = 0;
            for (var i = 0; i < actual.Length; i++)
            {
                if (BitConverter.SingleToInt32Bits(actual[i]) != BitConverter.SingleToInt32Bits(expected[i])) bits++;
                var delta = Math.Abs((double)actual[i] - expected[i]); var allowed = 1e-5 + 1e-4 * Math.Abs((double)expected[i]);
                if (delta > allowed) bad++; maximum = Math.Max(maximum, delta); ratio = Math.Max(ratio, delta / allowed);
            }
            return new JObject { ["bit_differences"] = bits, ["mismatched_elements"] = bad, ["maximum_absolute_error"] = maximum, ["maximum_tolerance_ratio"] = ratio };
        }
        static float[] Floats(string path) { var bytes = File.ReadAllBytes(path); var values = new float[bytes.Length / 4]; Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length); return values; }
    }
}
