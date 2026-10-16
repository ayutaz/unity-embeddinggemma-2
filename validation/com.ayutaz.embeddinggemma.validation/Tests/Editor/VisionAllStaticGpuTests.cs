using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using EmbeddingGemma.Tests;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEngine;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class VisionAllStaticGpuTests
    {
        [TestCase("square-0")] [TestCase("square-1")]
        [TestCase("landscape-0")] [TestCase("landscape-1")]
        [TestCase("portrait-0")] [TestCase("portrait-1")]
        public void NativeStaticVisionProjectionsMatchOriginalGoldenOnGpu(string caseId)
            => Run(caseId,false);

        [TestCase("square-0")] [TestCase("square-1")]
        [TestCase("landscape-0")] [TestCase("landscape-1")]
        [TestCase("portrait-0")] [TestCase("portrait-1")]
        public void FirstLayerQualifiedVisionEncoderMatchesOriginalGoldenOnGpu(string caseId)
            => Run(caseId,true);

        static void Run(string caseId,bool firstLayerOnly)
        {
            Assert.That(SystemInfo.supportsComputeShaders,Is.True,"Actual GPU required; never skip.");
            var source=Environment.GetEnvironmentVariable("EG_ACCEPTANCE_SOURCE");
            var root=Path.GetFullPath(Path.Combine(Application.dataPath,"../../k/artifacts/vision-encoder"));
            var label=firstLayerOnly?"vision-first-layer-qualified-whole-gpu-":"vision-all-static-gpu-";
            var output=Path.GetFullPath(Path.Combine(Application.dataPath,"../artifacts/"+label+source+"-"+caseId+".json"));
            Assert.That(File.Exists(output),Is.False);
            var report=new JObject{["success"]=false,["scope"]=firstLayerOnly?"one_original_vision_encoder_qualified_patch_first_rms_and_first_layer_candidate":"one_original_vision_encoder_native_static_gpu_projection_candidate",["runtime_source"]=source,
                ["case_id"]=caseId,["phase"]="audit",["all_profiles_validated"]=false,["end_to_end_validated"]=false};
            void Save(){Directory.CreateDirectory(Path.GetDirectoryName(output));File.WriteAllText(output,report.ToString());}
            void Check(string path,string sha){using var stream=File.OpenRead(path);Assert.That(PlayerFileHash.Compute(stream,out _),Is.EqualTo(sha));}
            var operations=new List<IGpuTensorOperation>();Save();
            try
            {
                var locked=JObject.Parse(File.ReadAllText(Path.Combine(Application.dataPath,"../Packages/packages-lock.json")));
                foreach(var package in new[]{"com.ayutaz.embeddinggemma","com.ayutaz.embeddinggemma.validation"})Assert.That(locked["dependencies"][package].Value<string>("hash"),Is.EqualTo(source));
                Check(Path.Combine(root,"vision-reference.json"),"3dfccd1a487ce04df3f8d640483ca045f7a7a56231bc1096ad0f845c923a29c6");
                Check(Path.Combine(root,"vision-validation.json"),"ab1aa775b553f33a5e2892abed2cdcef7768e35fbad6fa5af9ec3286a7ba90ba");
                var reference=JObject.Parse(File.ReadAllText(Path.Combine(root,"vision-reference.json")));
                var audit=JObject.Parse(File.ReadAllText(Path.Combine(root,"vision-validation.json")));
                var row=((JArray)reference["cases"]).Single(value=>value.Value<string>("id")==caseId);var profile=row.Value<string>("profile");
                var features=reference["profiles"][profile].Value<int>("feature_count");
                var modelSha=profile=="square"?"4ecd0106463e833588c7584a58ea4973fdbab1eb1a56389c2f1597d3e7df34e7":profile=="landscape"?"b9bc2336a34167eb0fdfdd99e49c6486d66b20afc23d72da8df1e10da4a1e632":"41c0ade742b09cf291c0b14343b1cf8bffa6f44c9bb2e8d782bc7ba508371e6b";
                Check(Path.Combine(root,profile+"-fp32.sentis"),modelSha);report["profile"]=profile;report["saved_model_sha256"]=modelSha;
                report["phase"]="prepare";Save();var timer=System.Diagnostics.Stopwatch.StartNew();
                var model=TextModelFile.Load(Path.Combine(root,profile+"-fp32.sentis"));var constants=model.constants.ToDictionary(value=>value.index);
                var selected=model.layers.Where(layer=>layer.GetType().Name=="MatMul"&&!constants.ContainsKey(layer.inputs[0])
                    &&constants.TryGetValue(layer.inputs[1],out var w)&&w.shape.rank==2).Select(layer=>layer.outputs[0]).ToArray();
                Assert.That(selected.Length,Is.GreaterThanOrEqualTo(113));Model rmsTemplate=null;
                if(firstLayerOnly)
                {
                    report["original_projection_count"]=selected.Length;
                    Check(Path.Combine(Application.dataPath,"GpuVisionFirstRmsCandidate/first-rms.onnx"),"724605434efc0b70bb28b59b5dc639f8b0aeb765a4170ec782397506f6549976");
                    rmsTemplate=ModelLoader.Load(UnityEditor.AssetDatabase.LoadAssetAtPath<ModelAsset>("Assets/GpuVisionFirstRmsCandidate/first-rms.onnx"));
                    selected=selected.Take(8).ToArray(); // Patch projection and the seven layer-0 projections only.
                    var marked=QualifiedVisionRmsOperation.MarkFirst(model,selected[1]);model=marked.model;
                    selected=new[]{selected[0],marked.output}.Concat(selected.Skip(1)).ToArray();
                    report["first_rms_replaced"]=true;
                }
                var plan=SavedProjectionPartition.Create(model,selected);
                var projections=new JArray();report["projections"]=projections;
                for(var i=0;i<selected.Length;i++)
                {
                    if(plan.Weights[i]==null)
                    {
                        Assert.That(firstLayerOnly&&i==1,Is.True);operations.Add(new QualifiedVisionRmsOperation(rmsTemplate));
                        projections.Add(new JObject{["kind"]="explicit_division_first_rms",["rows"]=plan.Rows[i],["width"]=plan.Widths[i]});continue;
                    }
                    Assert.That(plan.Rows[i]==2520||plan.Rows[i]==features,Is.True,"Qualify each original token-row profile; never guess dynamic shapes.");
                    var block=plan.Rows[i]==2520?384:192;
                    operations.Add(new GpuBlockedMatmul(plan.Widths[i],plan.Columns[i],plan.Weights[i],block));
                    var bytes=new byte[plan.Weights[i].Length*4];Buffer.BlockCopy(plan.Weights[i],0,bytes,0,bytes.Length);
                    using var stream=new MemoryStream(bytes);projections.Add(new JObject{["width"]=plan.Widths[i],["columns"]=plan.Columns[i],["rows"]=plan.Rows[i],["block"]=block,["weight_sha256"]=PlayerFileHash.Compute(stream,out _)});
                }
                using var pipeline=new GpuProjectionPipeline(plan.Models,plan.Inputs,plan.Outputs,operations.ToArray());
                report["prepare_ms"]=timer.Elapsed.TotalMilliseconds;report["projection_count"]=selected.Length-(firstLayerOnly?1:0);report["gpu_operation_count"]=selected.Length;report["segment_count"]=plan.Models.Length;report["cpu_computation_or_input_count"]=0;report["readback_flush_count"]=0;
                var pixelsName=row.Value<string>("pixels");Check(Path.Combine(root,pixelsName),audit["files"][pixelsName].Value<string>("sha256"));
                using var input=new Tensor<float>(new TensorShape(1,2520,768),Floats(Path.Combine(root,pixelsName)));ComputeTensorData.Pin(input);
                using var process=System.Diagnostics.Process.GetCurrentProcess();GpuExecutionProfiler.ReadProcessMemory(process,out var before,out _);report["host_working_set_before_execute_bytes"]=before;
                report["phase"]="execute";Save();timer.Restart();using var encoded=pipeline.Execute(input);
                Assert.That(encoded.backendType,Is.EqualTo(BackendType.GPUCompute));var actual=encoded.DownloadToArray();report["execute_and_final_download_ms"]=timer.Elapsed.TotalMilliseconds;
                GpuExecutionProfiler.ReadProcessMemory(process,out var after,out var peak);report["host_working_set_after_download_bytes"]=after;report["host_peak_working_set_bytes"]=peak;
                Assert.That(encoded.shape,Is.EqualTo(new TensorShape(features,512)));var expected=row["features"].ToObject<float[][]>().SelectMany(values=>values).ToArray();Assert.That(actual.Length,Is.EqualTo(expected.Length));
                int bad=0;double maximum=0,ratio=0;
                for(var i=0;i<actual.Length;i++)
                {
                    Assert.That(float.IsNaN(actual[i])||float.IsInfinity(actual[i]),Is.False);
                    var d=Math.Abs((double)actual[i]-expected[i]);var allowed=1e-5+1e-4*Math.Abs((double)expected[i]);if(d>allowed)bad++;maximum=Math.Max(maximum,d);ratio=Math.Max(ratio,d/allowed);
                }
                report["actual_backend"]="GPUCompute";report["mismatched_elements"]=bad;report["maximum_absolute_error"]=maximum;report["maximum_tolerance_ratio"]=ratio;report["phase"]="completed";Save();
                Assert.That(bad,Is.Zero,"Keep the unchanged original vision FP32 elementwise gate.");report["success"]=true;
            }
            catch(Exception error){report["error"]=error.ToString();throw;}
            finally{foreach(var operation in operations)operation.Dispose();Save();}
        }
        static float[] Floats(string path){var bytes=File.ReadAllBytes(path);var data=new float[bytes.Length/4];Buffer.BlockCopy(bytes,0,data,0,bytes.Length);return data;}
    }
}
