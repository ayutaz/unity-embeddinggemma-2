using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using EmbeddingGemma.Tests;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using UnityEditor;
using Unity.InferenceEngine;
using UnityEngine;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class SavedOnnxAudioGpuReferenceTests
    {
        [TestCase("silence")]
        [TestCase("impulse")]
        [TestCase("sine")]
        [TestCase("nyquist")]
        [TestCase("noise")]
        [TestCase("padded_noise")]
        public void SavedCandidateGpuMatchesOriginalGolden(string signal) => Execute(signal,"GpuAudioCandidate");

        [TestCase("silence")]
        [TestCase("impulse")]
        [TestCase("sine")]
        [TestCase("nyquist")]
        [TestCase("noise")]
        [TestCase("padded_noise")]
        public void RefinedInverseCandidateGpuMatchesOriginalGolden(string signal) => Execute(signal,"GpuAudioRefinedCandidate");

        [TestCase("silence")]
        [TestCase("impulse")]
        [TestCase("sine")]
        [TestCase("nyquist")]
        [TestCase("noise")]
        [TestCase("padded_noise")]
        public void RefinedStaticProjectionCandidateGpuMatchesOriginalGolden(string signal) => Execute(signal,"GpuAudioRefinedCandidate",true);

        [TestCase("silence")]
        [TestCase("impulse")]
        [TestCase("sine")]
        [TestCase("nyquist")]
        [TestCase("noise")]
        [TestCase("padded_noise")]
        public void RefinedAttentionCandidateGpuMatchesOriginalGolden(string signal) => Execute(signal,"GpuAudioRefinedCandidate",true,16);

        [TestCase("silence",32)]
        [TestCase("silence",16)]
        public void RefinedFineProjectionGpuMatchesOriginalGolden(string signal,int projectionChunk) => Execute(signal,"GpuAudioRefinedCandidate",true,16,projectionChunk);

        [TestCase("silence")]
        [TestCase("impulse")]
        [TestCase("sine")]
        [TestCase("nyquist")]
        [TestCase("noise")]
        [TestCase("padded_noise")]
        public void CascadeRmsCandidateGpuMatchesOriginalGolden(string signal) => Execute(signal,"GpuAudioCascadeCandidate",true,16);

        [TestCase("silence")]
        [TestCase("impulse")]
        [TestCase("sine")]
        [TestCase("nyquist")]
        [TestCase("noise")]
        [TestCase("padded_noise")]
        public void CascadeLayoutRmsCandidateGpuMatchesOriginalGolden(string signal) => Execute(signal,"GpuAudioCascadeLayoutCandidate",true,16);

        [TestCase("silence")]
        [TestCase("impulse")]
        [TestCase("sine")]
        [TestCase("nyquist")]
        [TestCase("noise")]
        [TestCase("padded_noise")]
        public void CompactOuterRmsCandidateGpuMatchesOriginalGolden(string signal) => Execute(signal,"GpuAudioCompactOuterCandidate",true,16);

        [TestCase("silence")]
        [TestCase("impulse")]
        [TestCase("sine")]
        [TestCase("nyquist")]
        [TestCase("noise")]
        [TestCase("padded_noise")]
        public void CompensatedConvCandidateGpuMatchesOriginalGolden(string signal) => Execute(signal,"GpuAudioCompensatedConvCandidate",true,16);

        void Execute(string signal,string candidateFolder,bool compensateStatic=false,int attentionChunk=0,int projectionChunk=64)
        {
            Assert.That(SystemInfo.supportsComputeShaders, Is.True, "Actual GPU required; no CPU substitute or skip.");
            var root = Path.GetFullPath(Path.Combine(Application.dataPath, "../../k/artifacts/audio-real-chain/audio-runtime-reference"));
            var candidateRoot = Path.Combine(Application.dataPath, candidateFolder);
            var source = Environment.GetEnvironmentVariable("EG_ACCEPTANCE_SOURCE");
            var variant=candidateFolder=="GpuAudioCandidate"?"":attentionChunk>0?"-refined-attention16-static"+projectionChunk:compensateStatic?"-refined-static64":"-refined-inverse";
            if(candidateFolder=="GpuAudioCascadeCandidate")variant="-cascade-attention16-static64";
            if(candidateFolder=="GpuAudioCascadeLayoutCandidate")variant="-cascade-layout-attention16-static64";
            if(candidateFolder=="GpuAudioCompactOuterCandidate")variant="-compact-outer-attention16-static64";
            if(candidateFolder=="GpuAudioCompensatedConvCandidate")variant="-conv64-attention16-static64";
            var output = Path.GetFullPath(Path.Combine(Application.dataPath, "../artifacts/saved-onnx-audio-gpu-" + source + variant + "-" + signal + ".json"));
            Assert.That(File.Exists(output), Is.False, "Preserve each invocation's evidence.");
            var report = new JObject { ["success"] = false, ["scope"] = "one_saved_onnx_audio_encoder_gpu_vs_immutable_original",
                ["signal"] = signal, ["runtime_source"] = source, ["requested_backend"] = "GPUCompute",
                ["unity"] = Application.unityVersion, ["sentis"] = "2.6.1", ["gpu"] = SystemInfo.graphicsDeviceName,
                ["graphics_api"] = SystemInfo.graphicsDeviceType.ToString(), ["end_to_end_validated"] = false,
                ["all_original_signals_validated"] = false, ["phase"] = "audit" };
            void Save() { Directory.CreateDirectory(Path.GetDirectoryName(output)); File.WriteAllText(output, report.ToString()); }
            void Audit(string path, JToken spec)
            {
                Assert.That(new FileInfo(path).Length, Is.EqualTo(spec.Value<long>("bytes")));
                using var stream = File.OpenRead(path);
                Assert.That(PlayerFileHash.Compute(stream, out _), Is.EqualTo(spec.Value<string>("sha256")));
            }
            Save();
            try
            {
                var locked = JObject.Parse(File.ReadAllText(Path.Combine(Application.dataPath, "../Packages/packages-lock.json")));
                foreach (var package in new[] { "com.ayutaz.embeddinggemma", "com.ayutaz.embeddinggemma.validation" })
                    Assert.That(locked["dependencies"][package].Value<string>("hash"), Is.EqualTo(source));
                var originalPath = Path.Combine(root, "audio-encoder-reference.json");
                using (var stream = File.OpenRead(originalPath))
                    Assert.That(PlayerFileHash.Compute(stream, out _), Is.EqualTo("ac23f39904b8149dd5ccb6485afcc93d96047f0977b5faf1d640c8bd018c13c9"));
                var original = JObject.Parse(File.ReadAllText(originalPath));
                var originalAudit = JObject.Parse(File.ReadAllText(Path.Combine(root, "audio-encoder-validation.json")));
                var candidate = JObject.Parse(File.ReadAllText(Path.Combine(candidateRoot, "audio-encoder-reference.json")));
                var candidateAudit = JObject.Parse(File.ReadAllText(Path.Combine(candidateRoot, "audio-encoder-validation.json")));
                Audit(Path.Combine(candidateRoot, "audio-encoder-reference.json"), candidateAudit["files"]["audio-encoder-reference.json"]);
                foreach (var field in new[] { "model_revision", "input_sha256", "versions" })
                    Assert.That(JToken.DeepEquals(candidate["metadata"][field], original["metadata"][field]), Is.True);
                report["candidate_source"] = candidate["metadata"]["source_commit"];
                report["reference_source"] = original["metadata"]["source_commit"];
                report["candidate_python_gate_passed"] = candidateAudit.Value<bool>("success");
                report["candidate_folder"] = candidateFolder;
                if(candidateFolder=="GpuAudioRefinedCandidate" || candidateFolder=="GpuAudioCascadeCandidate" || candidateFolder=="GpuAudioCascadeLayoutCandidate" || candidateFolder=="GpuAudioCompactOuterCandidate" || candidateFolder=="GpuAudioCompensatedConvCandidate")
                {
                    Assert.That(candidateAudit["graph_transform"],Is.Not.Null,"Transformed graph provenance required.");
                    report["graph_transform"]=candidateAudit["graph_transform"].DeepClone();
                    Assert.That(candidateAudit["graph_transform"].Value<bool>("existing_initializer_bytes_preserved"),Is.True);
                    Assert.That(candidateAudit["graph_transform"].Value<int>("replaced_inverse_count"),Is.EqualTo(2));
                    Assert.That(candidateAudit.Value<bool>("onnx_numeric_validation_performed"),Is.False,"Do not reuse the original graph's ONNX results.");
                    if(candidateFolder=="GpuAudioCascadeCandidate" || candidateFolder=="GpuAudioCascadeLayoutCandidate" || candidateFolder=="GpuAudioCompactOuterCandidate" || candidateFolder=="GpuAudioCompensatedConvCandidate")
                    {
                        var excluded=candidateFolder=="GpuAudioCascadeLayoutCandidate";
                        Assert.That(candidateAudit["rms_graph_transform"].Value<int>("replaced_rms_count"),Is.EqualTo(excluded?97:109));
                        if(excluded)Assert.That(((JArray)candidateAudit["rms_graph_transform"]["excluded_mean_names"]).Count,Is.EqualTo(12));
                        if(candidateFolder=="GpuAudioCompactOuterCandidate" || candidateFolder=="GpuAudioCompensatedConvCandidate")
                        {
                            Assert.That(((JArray)candidateAudit["rms_graph_transform"]["explicit_mean_override_names"]).Count,Is.EqualTo(12));
                            Assert.That(((JArray)candidateAudit["rms_graph_transform"]["excluded_mean_names"]).Count,Is.Zero);
                        }
                        Assert.That(candidateAudit["rms_graph_transform"].Value<bool>("existing_initializer_bytes_preserved"),Is.True);
                        report["rms_graph_transform"]=candidateAudit["rms_graph_transform"].DeepClone();
                    }
                    if(candidateFolder=="GpuAudioCompensatedConvCandidate")
                    {
                        var transform=candidateAudit["conv_graph_transform"];
                        Assert.That(transform.Value<int>("replaced_conv_count"),Is.EqualTo(1));
                        Assert.That(transform.Value<string>("source_graph_sha256"),Is.EqualTo("11607255142478773e621f5e9feebf2d3269457feb67d63215cecbc55fc61b67"));
                        Assert.That(transform.Value<string>("target_node"),Is.EqualTo("/subsample_conv_projection/layer1/conv/Conv"));
                        Assert.That(transform.Value<string>("weight_sha256"),Is.EqualTo("68309f01ea3e738fccb24a0db5db608841a3a8e59b0a4fa5d0fb521ed60bf43c"));
                        Assert.That(transform.Value<bool>("existing_initializer_bytes_preserved"),Is.True);
                        report["conv_graph_transform"]=transform.DeepClone();
                    }
                }
                var row = ((JArray)original["cases"]).Single(r => r.Value<string>("id") == signal);
                var candidateRow = ((JArray)candidate["cases"]).Single(r => r.Value<string>("id") == signal);
                var profileName = row.Value<string>("profile");
                Assert.That(candidateRow.Value<string>("profile"), Is.EqualTo(profileName));
                var profile = original["profiles"][profileName]; var candidateProfile = candidate["profiles"][profileName];
                var graphName = candidateProfile.Value<string>("graph");
                var assetPath = "Assets/"+candidateFolder+"/" + profileName + ".onnx";
                Audit(Path.Combine(Application.dataPath, candidateFolder, profileName + ".onnx"), candidateAudit["files"][graphName]);
                var inputName = row.Value<string>("input_features"); var maskName = profile.Value<string>("frame_mask");
                Audit(Path.Combine(root, inputName), originalAudit["files"][inputName]);
                Audit(Path.Combine(root, maskName), originalAudit["files"][maskName]);
                Assert.That(candidateAudit["files"][candidateRow.Value<string>("input_features")].Value<string>("sha256"),
                    Is.EqualTo(originalAudit["files"][inputName].Value<string>("sha256")));
                Assert.That(candidateAudit["files"][candidateProfile.Value<string>("frame_mask")].Value<string>("sha256"),
                    Is.EqualTo(originalAudit["files"][maskName].Value<string>("sha256")));
                report["graph_sha256"] = candidateAudit["files"][graphName]["sha256"]; report["phase"] = "load"; Save();
                var asset = AssetDatabase.LoadAssetAtPath<ModelAsset>(assetPath);
                Assert.That(asset, Is.Not.Null, "The audited ONNX must import before execution.");
                var model = ModelLoader.Load(asset);
                report["saved_graph_layer_count"]=model.layers.Count;
                if(attentionChunk>0)
                {
                    report["phase"]="transform_attention";Save();
                    model=CompensatedModel.CreateWithDynamicMatMuls(model,attentionChunk);
                    report["attention_compensated_layer_count"]=model.layers.Count;
                }
                if(compensateStatic)
                {
                    report["phase"]="transform_static_projections";Save();
                    model=CompensatedModel.Create(model,projectionChunk);
                    report["runtime_graph_transform"]=attentionChunk>0?"dynamic16_then_static"+projectionChunk:"CompensatedModel.Create/static64";
                    report["projection_chunk"]=projectionChunk;
                }
                var maskValues = Ints(Path.Combine(root, maskName)); var validFrames = maskValues.Sum();
                report["layer_count"] = model.layers.Count; report["phase"] = "worker"; Save();
                using var api = new AudioFeatureEncoder(model, BackendType.GPUCompute, new AudioEncoderProfile(validFrames));
                var worker = (Worker)typeof(AudioFeatureEncoder).GetField("worker", BindingFlags.Instance | BindingFlags.NonPublic).GetValue(api);
                var cpu = (HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallback", BindingFlags.Instance | BindingFlags.NonPublic).GetValue(worker);
                var count = cpu.Intersect(model.layers.SelectMany(l => l.outputs).Concat(model.inputs.Select(i => i.index))).Count();
                var flush = (HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallbackShouldFlushGPU", BindingFlags.Instance | BindingFlags.NonPublic).GetValue(worker);
                report["cpu_computation_or_input_count"] = count; report["readback_flush_count"] = flush.Count;
                Assert.That(count, Is.Zero); Assert.That(flush.Count, Is.Zero); Save();
                var features = new Tensor<float>(new TensorShape(1,1119,128), Floats(Path.Combine(root,inputName)));
                var mask = new Tensor<int>(new TensorShape(1,1119), maskValues);
                ComputeTensorData.Pin(features); ComputeTensorData.Pin(mask);
                using var input = (AudioPreprocessingResult)Activator.CreateInstance(typeof(AudioPreprocessingResult),
                    BindingFlags.Instance | BindingFlags.NonPublic, null, new object[] { features,mask,validFrames },null);
                report["phase"] = "inference"; Save();
                using var encoded = api.Encode(input);
                report["actual_backend"] = encoded.SoftTokens.backendType.ToString();
                Assert.That(encoded.SoftTokens.backendType, Is.EqualTo(BackendType.GPUCompute));
                var actual = encoded.SoftTokens.DownloadToArray();
                var expected = row["features"].ToObject<float[][]>().SelectMany(v => v).ToArray();
                Assert.That(actual.Length, Is.EqualTo(expected.Length));
                var mismatch = 0; double maximum=0, ratio=0;
                for (var i=0;i<actual.Length;i++)
                {
                    Assert.That(float.IsNaN(actual[i]) || float.IsInfinity(actual[i]), Is.False);
                    var difference=Math.Abs((double)actual[i]-expected[i]); var allowed=1e-5+1e-4*Math.Abs((double)expected[i]);
                    maximum=Math.Max(maximum,difference); ratio=Math.Max(ratio,difference/allowed); if(difference>allowed)mismatch++;
                }
                report["mismatched_elements"]=mismatch; report["maximum_absolute_error"]=maximum;
                report["maximum_tolerance_ratio"]=ratio; report["phase"]="completed"; Save();
                Assert.That(mismatch,Is.Zero,"Preserve the original FP32 encoder numeric gate."); report["success"]=true;
            }
            catch(Exception error) { report["error"]=error.ToString(); throw; }
            finally { Save(); }
        }
        static float[] Floats(string path) { var bytes=File.ReadAllBytes(path); var values=new float[bytes.Length/4]; Buffer.BlockCopy(bytes,0,values,0,bytes.Length); return values; }
        static int[] Ints(string path) { var bytes=File.ReadAllBytes(path); var values=new int[bytes.Length/4]; Buffer.BlockCopy(bytes,0,values,0,bytes.Length); return values; }
    }
}
