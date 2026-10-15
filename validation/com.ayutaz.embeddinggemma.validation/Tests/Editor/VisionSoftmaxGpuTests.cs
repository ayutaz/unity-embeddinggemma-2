using System;
using System.IO;
using System.Linq;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEngine;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class VisionSoftmaxGpuTests
    {
        [Test]
        public void CapturedOriginalLinuxPrefixSoftmaxMustMatchNativeBitsOnGpu()
        {
            var source=Environment.GetEnvironmentVariable("EG_ACCEPTANCE_SOURCE");
            var root=Path.GetFullPath(Path.Combine(Application.dataPath,"../../numerical-acceptance-state/np38719644488"));
            var output=Path.GetFullPath(Path.Combine(Application.dataPath,"../artifacts/vision-original-prefix-softmax-gpu-"+source+".json"));
            Assert.That(File.Exists(output),Is.False);
            var report=new JObject{["success"]=false,["runtime_source"]=source,["scope"]="captured_linux_avx2_original_first_softmax_prefix_only",
                ["reference_reproduces_original_golden"]=false,["full_encoder_validated"]=false,["end_to_end_validated"]=false};
            void Save(){Directory.CreateDirectory(Path.GetDirectoryName(output));File.WriteAllText(output,report.ToString());}
            void Check(string path,string sha){using var stream=File.OpenRead(path);Assert.That(PlayerFileHash.Compute(stream,out _),Is.EqualTo(sha));}
            Save();
            try
            {
                var locked=JObject.Parse(File.ReadAllText(Path.Combine(Application.dataPath,"../Packages/packages-lock.json")));
                foreach(var package in new[]{"com.ayutaz.embeddinggemma","com.ayutaz.embeddinggemma.validation"})Assert.That(locked["dependencies"][package].Value<string>("hash"),Is.EqualTo(source));
                var path=Path.Combine(root,"original-image-native.json");Check(path,"b36c8af5e9e248d587eb2cb99f20407375f5656a0a173ad1dc0063e740863bd6");
                var reference=JObject.Parse(File.ReadAllText(path));Assert.That(reference["saved_node_capture"].Value<bool>("observed_final_matches_normal_bits"),Is.True);
                Assert.That(reference.Value<string>("graph_sha256"),Is.EqualTo("f9c5175a1d75289ce814a9651f496f9bdae5e873ac66199a5c3d03e0172c16cc"));
                string Node(string name)
                {
                    var row=((JArray)reference["saved_node_capture"]["nodes"]).Single(value=>value.Value<string>("node")==name);
                    Assert.That(row.Value<int>("prefix_rows"),Is.EqualTo(16));Assert.That(row["shape"].Values<int>().ToArray(),Is.EqualTo(new[]{1,12,16,2520}));
                    var result=Path.Combine(root,"captures",row.Value<string>("file"));Check(result,row.Value<string>("sha256"));return result;
                }
                var values=Floats(Node("add_10"));var expected=Floats(Node("_softmax"));
                using var input=new Tensor<float>(new TensorShape(1,12,16,2520),values);ComputeTensorData.Pin(input);
                using var operation=new GpuSoftmax(8);using var result=operation.Execute(input);var actual=result.DownloadToArray();
                report["candidate"]=Metrics(actual,expected);report["actual_backend"]=result.backendType.ToString();
                report["cpu_computation_or_input_count"]=0;report["readback_flush_count"]=0;
                var raw=new byte[actual.Length*4];Buffer.BlockCopy(actual,0,raw,0,raw.Length);File.WriteAllBytes(Path.ChangeExtension(output,"bin"),raw);
                using(var stream=new MemoryStream(raw,false))report["capture_sha256"]=PlayerFileHash.Compute(stream,out _);
                Assert.That(input.DownloadToArray(),Is.EqualTo(values));Save();
                Assert.That(report["candidate"].Value<int>("bit_differences"),Is.Zero,"Keep same-input native AVX2 softmax bits.");
                report["success"]=true;
            }
            catch(Exception error){report["error"]=error.ToString();throw;}
            finally{Save();}
        }

        [Test]
        public void CapturedLayerZeroAttentionRowsMatchCurrentCpuSoftmaxEquationOnGpu()
        {
            Assert.That(SystemInfo.supportsComputeShaders, Is.True, "Actual GPU required; no skip.");
            var source = Environment.GetEnvironmentVariable("EG_ACCEPTANCE_SOURCE");
            var root = Path.Combine(Application.dataPath, "GpuVisionSoftmaxReference");
            var output = Path.GetFullPath(Path.Combine(Application.dataPath, "../artifacts/vision-softmax-equation-gpu-" + source + ".json"));
            Assert.That(File.Exists(output), Is.False);
            var report = new JObject { ["success"] = false, ["runtime_source"] = source,
                ["scope"] = "first_16_rows_per_head_captured_gpu_input_current_cpu_softmax_equation_only",
                ["original_internal_native_reference_available"] = false, ["full_encoder_validated"] = false, ["end_to_end_validated"] = false };
            void Save() { Directory.CreateDirectory(Path.GetDirectoryName(output)); File.WriteAllText(output, report.ToString()); }
            void Check(string path, string expected) { using var stream = File.OpenRead(path); Assert.That(PlayerFileHash.Compute(stream, out _), Is.EqualTo(expected)); }
            Save();
            try
            {
                var locked = JObject.Parse(File.ReadAllText(Path.Combine(Application.dataPath, "../Packages/packages-lock.json")));
                foreach (var package in new[] { "com.ayutaz.embeddinggemma", "com.ayutaz.embeddinggemma.validation" })
                    Assert.That(locked["dependencies"][package].Value<string>("hash"), Is.EqualTo(source));
                var manifest = Path.Combine(root, "softmax-reference.json"); Check(manifest, "f289e51e7f3a72dc9deaf66b9b1e01370dc2783545dcd0b0370b9d64899db0e9");
                var reference = JObject.Parse(File.ReadAllText(manifest));
                foreach (var key in new[] { "input", "captured_full_shape_gpu_output", "original_layer_model", "original_patch_input", "capture" })
                    Check(reference[key].Value<string>("file"), reference[key].Value<string>("sha256"));
                var expectedPath = Path.Combine(root, reference["expected"].Value<string>("file")); Check(expectedPath, reference["expected"].Value<string>("sha256"));
                var shape = new TensorShape(reference["shape"].Values<int>().ToArray()); Assert.That(shape, Is.EqualTo(new TensorShape(1, 12, 16, 2520)));
                var values = Floats(reference["input"].Value<string>("file")); var expected = Floats(expectedPath);
                using var input = new Tensor<float>(shape, values); ComputeTensorData.Pin(input);
                using var candidate = new GpuSoftmax(); using var result = candidate.Execute(input);
                Assert.That(result.backendType, Is.EqualTo(BackendType.GPUCompute)); Assert.That(result.shape, Is.EqualTo(shape));
                var actual = result.DownloadToArray();
                report["baseline_captured_full_shape"] = Metrics(Floats(reference["captured_full_shape_gpu_output"].Value<string>("file")), expected);
                report["candidate"] = Metrics(actual, expected);
                var raw = new byte[actual.Length * 4]; Buffer.BlockCopy(actual, 0, raw, 0, raw.Length); File.WriteAllBytes(Path.ChangeExtension(output, ".bin"), raw);
                using (var stream = new MemoryStream(raw, false)) report["capture_sha256"] = PlayerFileHash.Compute(stream, out _);
                Assert.That(input.DownloadToArray(), Is.EqualTo(values));
                report["borrowed_input_preserved"] = true; report["actual_backend"] = "GPUCompute";
                report["cpu_computation_or_input_count"] = 0; report["readback_flush_count"] = 0;
                report["reference_sha256"] = "f289e51e7f3a72dc9deaf66b9b1e01370dc2783545dcd0b0370b9d64899db0e9"; Save();
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
