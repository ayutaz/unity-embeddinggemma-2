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
    public sealed class VisionLayerZeroNativeProjectionsGpuTests
    {
        [Test]
        public void NativeBlock384VisionLayerZeroFromExactPatchMatchesOriginalOnGpu()
        {
            Assert.That(SystemInfo.supportsComputeShaders,Is.True,"Actual GPU required; never skip.");
            var source=Environment.GetEnvironmentVariable("EG_ACCEPTANCE_SOURCE");
            var root=Path.GetFullPath(Path.Combine(Application.dataPath,"../../k/artifacts/vision-diagnostics"));
            var modelPath=Path.GetFullPath(Path.Combine(Application.dataPath,"../artifacts/vision-layer0-original-patch-gpu-e39bdf1b5d0faf2ea424023d83076315736c6fde-captures/layer0.sentis"));
            var output=Path.GetFullPath(Path.Combine(Application.dataPath,"../artifacts/vision-layer0-native-projections-gpu-"+source+".json"));
            Assert.That(File.Exists(output),Is.False);
            var report=new JObject{["success"]=false,["scope"]="original_vision_layer0_seven_native_gpu_projections_candidate",["runtime_source"]=source,
                ["full_encoder_validated"]=false,["end_to_end_validated"]=false,["phase"]="audit"};
            void Save(){Directory.CreateDirectory(Path.GetDirectoryName(output));File.WriteAllText(output,report.ToString());}
            void Check(string path,string sha){using var stream=File.OpenRead(path);Assert.That(PlayerFileHash.Compute(stream,out _),Is.EqualTo(sha));}
            var operations=new List<IGpuTensorOperation>();Save();
            try
            {
                var locked=JObject.Parse(File.ReadAllText(Path.Combine(Application.dataPath,"../Packages/packages-lock.json")));
                foreach(var package in new[]{"com.ayutaz.embeddinggemma","com.ayutaz.embeddinggemma.validation"})Assert.That(locked["dependencies"][package].Value<string>("hash"),Is.EqualTo(source));
                Check(modelPath,"d09b8884fae82ecd0245967774a0c8671efe09156c50367c45b9f133fc53983d");
                Check(Path.Combine(root,"vision-diagnostics.json"),"998b7d0d200b257b0bc19de32f45ef27be4328088d2d51706ea956cc3b4461ce");
                var reference=JObject.Parse(File.ReadAllText(Path.Combine(root,"vision-diagnostics.json")));
                var model=TextModelFile.Load(modelPath);var constants=model.constants.ToDictionary(value=>value.index);
                var selected=model.layers.Where(layer=>layer.GetType().Name=="MatMul"&&!constants.ContainsKey(layer.inputs[0])
                    &&constants.TryGetValue(layer.inputs[1],out var w)&&w.shape.rank==2).Select(layer=>layer.outputs[0]).ToArray();
                Assert.That(selected.Length,Is.EqualTo(7));report["phase"]="prepare";Save();
                var timer=System.Diagnostics.Stopwatch.StartNew();var plan=SavedProjectionPartition.Create(model,selected);
                Assert.That(plan.Rows,Is.All.EqualTo(2520));var rows=new JArray();report["projections"]=rows;
                for(var i=0;i<selected.Length;i++)
                {
                    operations.Add(new GpuBlockedMatmul(plan.Widths[i],plan.Columns[i],plan.Weights[i],384));
                    var bytes=new byte[plan.Weights[i].Length*4];Buffer.BlockCopy(plan.Weights[i],0,bytes,0,bytes.Length);
                    using var stream=new MemoryStream(bytes);rows.Add(new JObject{["width"]=plan.Widths[i],["columns"]=plan.Columns[i],["rows"]=plan.Rows[i],["block"]=384,["weight_sha256"]=PlayerFileHash.Compute(stream,out _)});
                }
                using var pipeline=new GpuProjectionPipeline(plan.Models,plan.Inputs,plan.Outputs,operations.ToArray());
                report["prepare_ms"]=timer.Elapsed.TotalMilliseconds;report["segment_count"]=plan.Models.Length;report["cpu_computation_or_input_count"]=0;report["readback_flush_count"]=0;
                var inputPath=Path.Combine(root,"native-stage-0.bin");var expectedPath=Path.Combine(root,"native-stage-3.bin");
                foreach(var file in new[]{inputPath,expectedPath})Check(file,reference["files"][Path.GetFileName(file)].Value<string>("sha256"));
                using var input=new Tensor<float>(new TensorShape(1,2520,768),Floats(inputPath));ComputeTensorData.Pin(input);
                report["phase"]="execute";Save();timer.Restart();using var encoded=pipeline.Execute(input);
                Assert.That(encoded.backendType,Is.EqualTo(BackendType.GPUCompute));var actual=encoded.DownloadToArray();
                report["execute_and_final_download_ms"]=timer.Elapsed.TotalMilliseconds;var expected=Floats(expectedPath);Assert.That(actual.Length,Is.EqualTo(expected.Length));
                var capture=Path.ChangeExtension(output,"bin");var raw=new byte[actual.Length*4];Buffer.BlockCopy(actual,0,raw,0,raw.Length);File.WriteAllBytes(capture,raw);
                using(var stream=File.OpenRead(capture))report["capture_sha256"]=PlayerFileHash.Compute(stream,out _);
                int bits=0,bad=0;double maximum=0,ratio=0;
                for(var i=0;i<actual.Length;i++)
                {
                    Assert.That(float.IsNaN(actual[i])||float.IsInfinity(actual[i]),Is.False);
                    if(BitConverter.SingleToInt32Bits(actual[i])!=BitConverter.SingleToInt32Bits(expected[i]))bits++;
                    var d=Math.Abs((double)actual[i]-expected[i]);var allowed=1e-5+1e-4*Math.Abs((double)expected[i]);if(d>allowed)bad++;maximum=Math.Max(maximum,d);ratio=Math.Max(ratio,d/allowed);
                }
                report["actual_backend"]="GPUCompute";report["bit_differences"]=bits;report["mismatched_elements"]=bad;report["maximum_absolute_error"]=maximum;report["maximum_tolerance_ratio"]=ratio;report["phase"]="completed";Save();
                Assert.That(bad,Is.Zero,"Keep the unchanged original layer-0 FP32 gate.");report["success"]=true;
            }
            catch(Exception error){report["error"]=error.ToString();throw;}
            finally{foreach(var operation in operations)operation.Dispose();Save();}
        }
        static float[] Floats(string path){var bytes=File.ReadAllBytes(path);var data=new float[bytes.Length/4];Buffer.BlockCopy(bytes,0,data,0,bytes.Length);return data;}
    }
}
