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
    public sealed class VisionAttentionMatmulObservationGpuTests
    {
        [Test]
        public void OriginalAttentionOperandsAndOutputRowPrefixesPreserveNormalLayerBits()
        {
            Assert.That(SystemInfo.supportsComputeShaders, Is.True, "Actual GPU required; no skip.");
            var source = Environment.GetEnvironmentVariable("EG_ACCEPTANCE_SOURCE");
            var output = Path.GetFullPath(Path.Combine(Application.dataPath, "../artifacts/vision-attention-matmul-observation-gpu-" + source + ".json"));
            Assert.That(File.Exists(output), Is.False);
            var report = new JObject { ["success"] = false, ["runtime_source"] = source,
                ["scope"] = "original_layer0_two_attention_matmuls_bounded_outputs_only", ["diagnostic_readbacks_only"] = true,
                ["original_internal_native_reference_available"] = false, ["full_encoder_validated"] = false, ["end_to_end_validated"] = false };
            void Save() { Directory.CreateDirectory(Path.GetDirectoryName(output)); File.WriteAllText(output, report.ToString()); }
            string Hash(string path) { using var stream = File.OpenRead(path); return PlayerFileHash.Compute(stream, out _); }
            Save();
            try
            {
                var locked = JObject.Parse(File.ReadAllText(Path.Combine(Application.dataPath, "../Packages/packages-lock.json")));
                foreach (var package in new[] { "com.ayutaz.embeddinggemma", "com.ayutaz.embeddinggemma.validation" })
                    Assert.That(locked["dependencies"][package].Value<string>("hash"), Is.EqualTo(source));
                var path = Path.GetFullPath(Path.Combine(Application.dataPath, "../artifacts/vision-layer0-original-patch-gpu-e39bdf1b5d0faf2ea424023d83076315736c6fde-captures/layer0.sentis"));
                Assert.That(Hash(path), Is.EqualTo("d09b8884fae82ecd0245967774a0c8671efe09156c50367c45b9f133fc53983d"));
                var model = TextModelFile.Load(path); var before = SerializedHash(model); var constants = new HashSet<int>(model.constants.Select(value => value.index));
                var selected = model.layers.Where(layer => layer.GetType().Name == "MatMul" && !constants.Contains(layer.inputs[1])).ToArray(); Assert.That(selected.Length, Is.EqualTo(2));
                var inference = typeof(Worker).Assembly.GetType("Unity.InferenceEngine.Compiler.Analyser.PartialInferenceAnalysis");
                var context = inference.GetMethod("InferModelPartialTensors").Invoke(null, new object[] { model });
                TensorShape Shape(int index)
                {
                    var partial = context.GetType().GetMethod("GetPartialTensor").Invoke(context, new object[] { index });
                    var shape = (DynamicTensorShape)partial.GetType().GetProperty("shape").GetValue(partial); Assert.That(shape.IsStatic(), Is.True); return shape.ToTensorShape();
                }
                using (var audit = new Worker(model, BackendType.GPUCompute))
                {
                    var cpu = (HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallback", BindingFlags.Instance | BindingFlags.NonPublic).GetValue(audit);
                    var flush = (HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallbackShouldFlushGPU", BindingFlags.Instance | BindingFlags.NonPublic).GetValue(audit);
                    var count = cpu.Intersect(model.layers.SelectMany(layer => layer.outputs).Concat(model.inputs.Select(value => value.index))).Count();
                    report["cpu_computation_or_input_count"] = count; report["readback_flush_count"] = flush.Count; Assert.That(count, Is.Zero); Assert.That(flush.Count, Is.Zero);
                }
                Assert.That(Shape(selected[0].inputs[0]), Is.EqualTo(new TensorShape(1, 12, 2520, 64)));
                Assert.That(Shape(selected[0].inputs[1]), Is.EqualTo(new TensorShape(1, 12, 64, 2520)));
                Assert.That(Shape(selected[1].inputs[0]), Is.EqualTo(new TensorShape(1, 12, 2520, 2520)));
                Assert.That(Shape(selected[1].inputs[1]), Is.EqualTo(new TensorShape(1, 12, 2520, 64)));
                var small = new HashSet<int>(selected.SelectMany(layer => layer.inputs).Where(index => Shape(index).length <= 12 * 2520 * 64)); Assert.That(small.Count, Is.EqualTo(3));
                var prefixes = new HashSet<int>(selected.SelectMany(layer => layer.outputs).Concat(selected.SelectMany(layer => layer.inputs).Where(index => !small.Contains(index))));
                Assert.That(prefixes.Count, Is.EqualTo(3));
                var inputPath = Path.GetFullPath(Path.Combine(Application.dataPath, "../../k/artifacts/vision-diagnostics/native-stage-0.bin"));
                Assert.That(Hash(inputPath), Is.EqualTo("bc4c19c9aaf9688801c99ed073fdde22f0daf6b822dc2c9c4b7a3f7c9aded636"));
                using var input = new Tensor<float>(new TensorShape(1, 2520, 768), Floats(inputPath)); ComputeTensorData.Pin(input);
                report["phase"] = "observe_small_operands"; Save();
                var full = WorkerObservation.ObserveIndexes(model, BackendType.GPUCompute, input, small);
                Assert.That(full.FinalMatchesBaselineBits, Is.True); Assert.That(full.Snapshots.Select(value => value.TensorIndex), Is.EquivalentTo(small));
                report["phase"] = "observe_output_prefixes"; Save(); var sampled = WorkerObservation.ObserveRowPrefixes(model, input, prefixes, 16);
                Assert.That(sampled.FinalMatchesBaselineBits, Is.True); Assert.That(sampled.Snapshots.Select(value => value.TensorIndex), Is.EquivalentTo(prefixes));
                var folder = Path.ChangeExtension(output, null) + "-captures"; Assert.That(Directory.Exists(folder), Is.False); Directory.CreateDirectory(folder);
                JObject Write(string name, float[] values, TensorShape shape)
                {
                    var raw = new byte[values.Length * 4]; Buffer.BlockCopy(values, 0, raw, 0, raw.Length); var target = Path.Combine(folder, name + ".bin"); File.WriteAllBytes(target, raw);
                    return new JObject { ["file"] = target, ["sha256"] = Hash(target), ["shape"] = Dimensions(shape), ["elements"] = values.Length };
                }
                var tensors = new JObject(); report["tensors"] = tensors;
                foreach (var snapshot in full.Snapshots.Concat(sampled.Snapshots))
                {
                    Assert.That(snapshot.Backend, Is.EqualTo(BackendType.GPUCompute)); var row = Write("tensor-" + snapshot.TensorIndex, snapshot.Values, snapshot.Shape);
                    row["original_shape"] = Dimensions(snapshot.OriginalShape); row["row_prefix_only"] = prefixes.Contains(snapshot.TensorIndex); tensors[snapshot.TensorIndex.ToString()] = row;
                }
                var matmuls = new JArray(); report["matmuls"] = matmuls;
                foreach (var layer in selected)
                {
                    Assert.That((bool?)layer.GetType().GetField("transposeA")?.GetValue(layer) ?? false, Is.False);
                    Assert.That((bool?)layer.GetType().GetField("transposeB")?.GetValue(layer) ?? false, Is.False);
                    matmuls.Add(new JObject { ["left"] = layer.inputs[0], ["right"] = layer.inputs[1], ["output"] = layer.outputs[0] });
                }
                foreach (var values in new[] { full.Final, sampled.Final })
                {
                    var final = Write(values == full.Final ? "full-observer-final" : "prefix-observer-final", values, new TensorShape(1, 2520, 768));
                    Assert.That(final.Value<string>("sha256"), Is.EqualTo("bc417e7f0e3486a5ac0025878d58c2c0b612ea5dea99a53e6b8db0e6f93b1513"));
                }
                Assert.That(SerializedHash(model), Is.EqualTo(before)); report["source_model_bytes_preserved"] = true; report["source_model_serialized_sha256"] = before;
                report["original_layer_model"] = new JObject { ["file"] = path, ["sha256"] = Hash(path) };
                report["original_patch_input"] = new JObject { ["file"] = inputPath, ["sha256"] = Hash(inputPath) };
                report["normal_final_matches_observed_bits"] = true; report["actual_backend"] = "GPUCompute"; report["phase"] = "complete"; report["success"] = true;
            }
            catch (Exception error) { report["error"] = error.ToString(); throw; }
            finally { Save(); }
        }
        static JArray Dimensions(TensorShape shape) => JArray.FromObject(Enumerable.Range(0, shape.rank).Select(axis => shape[axis]).ToArray());
        static string SerializedHash(Model model) { using var stream = new MemoryStream(); ModelWriter.Save(stream, model); stream.Position = 0; return PlayerFileHash.Compute(stream, out _); }
        static float[] Floats(string path) { var bytes = File.ReadAllBytes(path); var values = new float[bytes.Length / 4]; Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length); return values; }
    }
}
