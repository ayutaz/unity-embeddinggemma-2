using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using EmbeddingGemma.Tests;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEngine;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class VisionSoftmaxObservationGpuTests
    {
        [Test]
        public void OriginalLayerZeroSoftmaxRowPrefixesPreserveNormalGpuOutput()
        {
            Assert.That(SystemInfo.supportsComputeShaders, Is.True, "Actual GPU required; no skip.");
            var source = Environment.GetEnvironmentVariable("EG_ACCEPTANCE_SOURCE");
            var output = Path.GetFullPath(Path.Combine(Application.dataPath, "../artifacts/vision-softmax-row-observation-gpu-" + source + ".json"));
            Assert.That(File.Exists(output), Is.False);
            var report = new JObject { ["success"] = false, ["runtime_source"] = source,
                ["scope"] = "original_layer0_softmax_first_16_rows_per_head_only", ["diagnostic_readbacks_only"] = true,
                ["original_internal_native_reference_available"] = false, ["full_encoder_validated"] = false, ["end_to_end_validated"] = false };
            void Save() { Directory.CreateDirectory(Path.GetDirectoryName(output)); File.WriteAllText(output, report.ToString()); }
            string Hash(string path) { using var stream = File.OpenRead(path); return PlayerFileHash.Compute(stream, out _); }
            void Check(string path, string expected) { Assert.That(Hash(path), Is.EqualTo(expected)); }
            Save();
            try
            {
                var locked = JObject.Parse(File.ReadAllText(Path.Combine(Application.dataPath, "../Packages/packages-lock.json")));
                foreach (var package in new[] { "com.ayutaz.embeddinggemma", "com.ayutaz.embeddinggemma.validation" })
                    Assert.That(locked["dependencies"][package].Value<string>("hash"), Is.EqualTo(source));
                var path = Path.GetFullPath(Path.Combine(Application.dataPath, "../artifacts/vision-layer0-original-patch-gpu-e39bdf1b5d0faf2ea424023d83076315736c6fde-captures/layer0.sentis"));
                const string modelHash = "d09b8884fae82ecd0245967774a0c8671efe09156c50367c45b9f133fc53983d";
                const string inputHash = "bc4c19c9aaf9688801c99ed073fdde22f0daf6b822dc2c9c4b7a3f7c9aded636";
                Check(path, modelHash); var model = TextModelFile.Load(path);
                var softmax = model.layers.Single(layer => layer.GetType().Name == "Softmax");
                var axis = (int)softmax.GetType().GetField("axis").GetValue(softmax);
                report["softmax_input_index"] = softmax.inputs[0]; report["softmax_output_index"] = softmax.outputs[0]; report["softmax_axis"] = axis;
                using (var audit = new Worker(model, BackendType.GPUCompute))
                {
                    var cpu = (HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallback", BindingFlags.Instance | BindingFlags.NonPublic).GetValue(audit);
                    var flush = (HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallbackShouldFlushGPU", BindingFlags.Instance | BindingFlags.NonPublic).GetValue(audit);
                    var count = cpu.Intersect(model.layers.SelectMany(layer => layer.outputs).Concat(model.inputs.Select(value => value.index))).Count();
                    report["cpu_computation_or_input_count"] = count; report["readback_flush_count"] = flush.Count;
                    Assert.That(count, Is.Zero); Assert.That(flush.Count, Is.Zero);
                }
                var inputPath = Path.GetFullPath(Path.Combine(Application.dataPath, "../../k/artifacts/vision-diagnostics/native-stage-0.bin"));
                Check(inputPath, inputHash);
                var folder = Path.ChangeExtension(output, null) + "-captures"; Assert.That(Directory.Exists(folder), Is.False); Directory.CreateDirectory(folder);
                var beforePath = Path.Combine(folder, "before.sentis"); ModelWriter.Save(beforePath, model); var before = Hash(beforePath);
                using var input = new Tensor<float>(new TensorShape(1, 2520, 768), Floats(inputPath)); ComputeTensorData.Pin(input);
                var selected = new HashSet<int> { softmax.inputs[0], softmax.outputs[0] };
                report["phase"] = "observe"; Save();
                var observed = WorkerObservation.ObserveRowPrefixes(model, input, selected, 16);
                Assert.That(observed.FinalMatchesBaselineBits, Is.True);
                Assert.That(observed.Snapshots.Select(value => value.TensorIndex), Is.EquivalentTo(selected));
                JObject Write(string name, float[] values, TensorShape shape)
                {
                    var bytes = new byte[values.Length * 4]; Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
                    var target = Path.Combine(folder, name + ".bin"); File.WriteAllBytes(target, bytes);
                    return new JObject { ["file"] = target, ["sha256"] = Hash(target), ["shape"] = Shape(shape), ["elements"] = values.Length };
                }
                foreach (var pair in new[] { ("input", softmax.inputs[0]), ("output", softmax.outputs[0]) })
                {
                    var snapshot = observed.Snapshots.Single(value => value.TensorIndex == pair.Item2);
                    Assert.That(snapshot.Backend, Is.EqualTo(BackendType.GPUCompute));
                    var original = snapshot.OriginalShape; Assert.That(original.rank, Is.GreaterThanOrEqualTo(3));
                    Assert.That(original[original.rank - 1], Is.EqualTo(2520)); Assert.That(original[original.rank - 2], Is.EqualTo(2520));
                    Assert.That(original.length, Is.EqualTo(12 * 2520 * 2520)); Assert.That(axis < 0 ? original.rank + axis : axis, Is.EqualTo(original.rank - 1));
                    Assert.That(snapshot.Values.Length, Is.EqualTo(12 * 16 * 2520));
                    var row = Write(pair.Item1, snapshot.Values, snapshot.Shape); row["original_shape"] = Shape(original); row["tensor_index"] = snapshot.TensorIndex;
                    report[pair.Item1] = row;
                }
                report["final"] = Write("layer0-output", observed.Final, new TensorShape(1, 2520, 768));
                Assert.That(report["final"].Value<string>("sha256"), Is.EqualTo("bc417e7f0e3486a5ac0025878d58c2c0b612ea5dea99a53e6b8db0e6f93b1513"));
                var afterPath = Path.Combine(folder, "after.sentis"); ModelWriter.Save(afterPath, model); Assert.That(Hash(afterPath), Is.EqualTo(before));
                Check(path, modelHash); Check(inputPath, inputHash);
                report["original_layer_model"] = new JObject { ["file"] = path, ["sha256"] = modelHash };
                report["original_patch_input"] = new JObject { ["file"] = inputPath, ["sha256"] = inputHash };
                report["normal_final_matches_observed_bits"] = true; report["source_model_bytes_preserved"] = true;
                report["source_model_serialized_sha256"] = before; report["actual_backend"] = "GPUCompute"; report["rows_per_head"] = 16;
                report["phase"] = "complete"; report["success"] = true;
            }
            catch (Exception error) { report["error"] = error.ToString(); throw; }
            finally { Save(); }
        }
        static JArray Shape(TensorShape shape) => JArray.FromObject(Enumerable.Range(0, shape.rank).Select(axis => shape[axis]).ToArray());
        static float[] Floats(string path) { var bytes = File.ReadAllBytes(path); var values = new float[bytes.Length / 4]; Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length); return values; }
    }
}
