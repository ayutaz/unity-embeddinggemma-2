using System;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Security.Cryptography;
using EmbeddingGemma.Tests;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEngine;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class VisionNativePatchWholeGpuTests
    {
        [TestCase("square-0")] [TestCase("square-1")]
        [TestCase("landscape-0")] [TestCase("landscape-1")]
        [TestCase("portrait-0")] [TestCase("portrait-1")]
        public void NativePatch384OriginalVisionEncoderMustMatchGoldenOnGpu(string caseId)
            => Run(caseId,false);

        [TestCase("square-0")] [TestCase("square-1")]
        [TestCase("landscape-0")] [TestCase("landscape-1")]
        [TestCase("portrait-0")] [TestCase("portrait-1")]
        public void InlineAllVisionLayerRmsOriginalEncoderMustMatchGoldenOnGpu(string caseId)
            => Run(caseId,true);

        static void Run(string caseId,bool inlineAllRms)
        {
            Assert.That(SystemInfo.supportsComputeShaders,Is.True,"Actual GPU required; never skip.");
            var source=Environment.GetEnvironmentVariable("EG_ACCEPTANCE_SOURCE");
            var root=Path.GetFullPath(Path.Combine(Application.dataPath,"../../k/artifacts/vision-encoder"));
            var label=inlineAllRms?"vision-inline-all-layer-rms-whole-gpu-":"vision-native-patch-whole-gpu-";
            var output=Path.GetFullPath(Path.Combine(Application.dataPath,"../artifacts/"+label+source+"-"+caseId+".json"));
            Assert.That(File.Exists(output),Is.False);
            var report=new JObject{["success"]=false,["scope"]=inlineAllRms?"one_original_vision_encoder_gpu_all_sixteen_layer_rms_and_native_patch384_candidate":"one_original_vision_encoder_gpu_native_patch384_candidate",["runtime_source"]=source,
                ["case_id"]=caseId,["phase"]="audit",["all_profiles_validated"]=false,["end_to_end_validated"]=false};
            void Save(){Directory.CreateDirectory(Path.GetDirectoryName(output));File.WriteAllText(output,report.ToString());}
            void Check(string path,string sha){using var stream=File.OpenRead(path);Assert.That(PlayerFileHash.Compute(stream,out _),Is.EqualTo(sha));}
            Save();
            try
            {
                var locked=JObject.Parse(File.ReadAllText(Path.Combine(Application.dataPath,"../Packages/packages-lock.json")));
                foreach(var package in new[]{"com.ayutaz.embeddinggemma","com.ayutaz.embeddinggemma.validation"})Assert.That(locked["dependencies"][package].Value<string>("hash"),Is.EqualTo(source));
                Check(Path.Combine(root,"vision-reference.json"),"3dfccd1a487ce04df3f8d640483ca045f7a7a56231bc1096ad0f845c923a29c6");
                Check(Path.Combine(root,"vision-validation.json"),"ab1aa775b553f33a5e2892abed2cdcef7768e35fbad6fa5af9ec3286a7ba90ba");
                var reference=JObject.Parse(File.ReadAllText(Path.Combine(root,"vision-reference.json")));
                var audit=JObject.Parse(File.ReadAllText(Path.Combine(root,"vision-validation.json")));
                var row=((JArray)reference["cases"]).Single(value=>value.Value<string>("id")==caseId);var profile=row.Value<string>("profile");
                var modelSha=profile=="square"?"4ecd0106463e833588c7584a58ea4973fdbab1eb1a56389c2f1597d3e7df34e7":profile=="landscape"?"b9bc2336a34167eb0fdfdd99e49c6486d66b20afc23d72da8df1e10da4a1e632":"41c0ade742b09cf291c0b14343b1cf8bffa6f44c9bb2e8d782bc7ba508371e6b";
                Check(Path.Combine(root,profile+"-fp32.sentis"),modelSha);report["profile"]=profile;report["saved_model_sha256"]=modelSha;
                report["phase"]="prepare";Save();var timer=System.Diagnostics.Stopwatch.StartNew();
                var original=TextModelFile.Load(Path.Combine(root,profile+"-fp32.sentis"));var constants=original.constants.ToDictionary(value=>value.index);
                var working=original;var originalCoefficients=CoefficientsHash(original);
                if(inlineAllRms)
                {
                    Check(Path.Combine(Application.dataPath,"GpuVisionInlineRmsCandidate/inline-rms.onnx"),"15c8ec537c8d40386e61b675c05aaaa65b296667c0d38c39dc7d12fe28a25ccb");
                    var preparationPath=Path.Combine(Application.dataPath,"GpuVisionHeadRmsCandidates/head-rms-candidates.json");
                    Check(preparationPath,"bb1f2d3df3905551b34723badefd7f1dfdc23da08a9ab4c72545d5d1368ce229");
                    var preparation=JObject.Parse(File.ReadAllText(preparationPath));
                    Model Head(int index)
                    {
                        var entry=((JArray)preparation["rows"]).Single(value=>value.Value<int>("index")==index);
                        var asset="GpuVisionHeadRmsCandidates/"+entry["onnx"].Value<string>("file");Check(Path.Combine(Application.dataPath,asset),entry["onnx"].Value<string>("sha256"));
                        return ModelLoader.Load(UnityEditor.AssetDatabase.LoadAssetAtPath<ModelAsset>("Assets/"+asset));
                    }
                    var template=ModelLoader.Load(UnityEditor.AssetDatabase.LoadAssetAtPath<ModelAsset>("Assets/GpuVisionInlineRmsCandidate/inline-rms.onnx"));
                    working=QualifiedVisionRmsOperation.InlineLayerRms(original,template,Head(1),Head(3),16);
                    Assert.That(original.constants.All(value=>working.constants.Any(other=>ReferenceEquals(value,other))),Is.True);
                    report["inline_rms_count"]=112;report["layer_count"]=working.layers.Count;
                    report["original_square_count"]=original.layers.Count(layer=>layer.GetType().Name=="Square");
                    report["remaining_original_rms_count"]=original.layers.Count(layer=>layer.GetType().Name=="Square")-112;
                    report["all_original_coefficient_descriptors_retained"]=true;
                }
                var projection=original.layers.First(layer=>layer.GetType().Name=="MatMul"&&constants.TryGetValue(layer.inputs[1],out var w)&&w.shape==new TensorShape(768,768));
                var raw=(ArraySegment<byte>)typeof(Constant).GetField("array",BindingFlags.NonPublic|BindingFlags.Instance).GetValue(constants[projection.inputs[1]]);
                using(var stream=new MemoryStream(raw.Array,raw.Offset,raw.Count,false))Assert.That(PlayerFileHash.Compute(stream,out _),Is.EqualTo("696c2e091ed4ad117c10faa6bf9707eb808649fb09c84fee457d2327408acc90"));
                var weights=new float[raw.Count/4];Buffer.BlockCopy(raw.Array,raw.Offset,weights,0,raw.Count);
                var before=ModelTensorSlice.Extract(working,projection.inputs[0],"projection_input");
                var after=ModelTensorSlice.Extract(working,working.outputs[0].index,"soft_tokens",new[]{new Model.Input{
                    name="projection_output",index=projection.outputs[0],dataType=DataType.Float,shape=new DynamicTensorShape(2520,768)}});
                Assert.That(before.inputs.Count,Is.EqualTo(1));Assert.That(after.inputs.Count,Is.EqualTo(1));
                using var operation=new GpuBlockedMatmul(768,768,weights,384);
                using var pipeline=new GpuProjectionPipeline(new[]{before,after},new[]{"projection_input"},new[]{"projection_output"},new[]{operation});
                report["prepare_ms"]=timer.Elapsed.TotalMilliseconds;report["cpu_computation_or_input_count"]=0;report["readback_flush_count"]=0;
                var pixelsName=row.Value<string>("pixels");Check(Path.Combine(root,pixelsName),audit["files"][pixelsName].Value<string>("sha256"));
                using var input=new Tensor<float>(new TensorShape(1,2520,768),Floats(Path.Combine(root,pixelsName)));ComputeTensorData.Pin(input);
                using var process=System.Diagnostics.Process.GetCurrentProcess();GpuExecutionProfiler.ReadProcessMemory(process,out var memoryBefore,out _);
                report["host_working_set_before_execute_bytes"]=memoryBefore;report["phase"]="execute";Save();timer.Restart();
                using var encoded=pipeline.Execute(input);Assert.That(encoded.backendType,Is.EqualTo(BackendType.GPUCompute));
                var actual=encoded.DownloadToArray();report["execute_and_final_download_ms"]=timer.Elapsed.TotalMilliseconds;
                GpuExecutionProfiler.ReadProcessMemory(process,out var memoryAfter,out var peak);report["host_working_set_after_download_bytes"]=memoryAfter;report["host_peak_working_set_bytes"]=peak;
                var expected=row["features"].ToObject<float[][]>().SelectMany(values=>values).ToArray();Assert.That(actual.Length,Is.EqualTo(expected.Length));
                Assert.That(encoded.shape,Is.EqualTo(new TensorShape(reference["profiles"][profile].Value<int>("feature_count"),512)));
                int bad=0;double maximum=0,ratio=0;
                for(var i=0;i<actual.Length;i++)
                {
                    Assert.That(float.IsNaN(actual[i])||float.IsInfinity(actual[i]),Is.False);
                    var d=Math.Abs((double)actual[i]-expected[i]);var allowed=1e-5+1e-4*Math.Abs((double)expected[i]);if(d>allowed)bad++;maximum=Math.Max(maximum,d);ratio=Math.Max(ratio,d/allowed);
                }
                report["actual_backend"]="GPUCompute";report["mismatched_elements"]=bad;report["maximum_absolute_error"]=maximum;report["maximum_tolerance_ratio"]=ratio;report["phase"]="completed";Save();
                Assert.That(CoefficientsHash(original),Is.EqualTo(originalCoefficients));report["original_coefficient_bytes_preserved"]=true;report["original_coefficients_sha256"]=originalCoefficients;
                Assert.That(bad,Is.Zero,"Keep the unchanged original FP32 elementwise gate.");report["success"]=true;
            }
            catch(Exception error){report["error"]=error.ToString();throw;}
            finally{Save();}
        }
        static string CoefficientsHash(Model model)
        {
            var field=typeof(Constant).GetField("array",BindingFlags.Instance|BindingFlags.NonPublic);using var hash=IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
            foreach(var constant in model.constants){var raw=(ArraySegment<byte>)field.GetValue(constant);hash.AppendData(raw.Array,raw.Offset,raw.Count);}
            return BitConverter.ToString(hash.GetHashAndReset()).Replace("-","").ToLowerInvariant();
        }
        static float[] Floats(string path){var bytes=File.ReadAllBytes(path);var data=new float[bytes.Length/4];Buffer.BlockCopy(bytes,0,data,0,bytes.Length);return data;}
    }
}
