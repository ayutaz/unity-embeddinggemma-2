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
    public sealed class VisionLayerZeroGpuTests
    {
        [Test]
        public void OriginalVisionLayerZeroFromExactPatchMustMatchNativeOnGpu()
        {
            Assert.That(SystemInfo.supportsComputeShaders,Is.True,"Actual GPU required; never skip.");
            var source=Environment.GetEnvironmentVariable("EG_ACCEPTANCE_SOURCE");
            var root=Path.GetFullPath(Path.Combine(Application.dataPath,"../../k/artifacts/vision-diagnostics"));
            var output=Path.GetFullPath(Path.Combine(Application.dataPath,"../artifacts/vision-layer0-original-patch-gpu-"+source+".json"));
            Assert.That(File.Exists(output),Is.False);
            var report=new JObject{["success"]=false,["scope"]="original_vision_layer0_exact_patch_gpu_diagnosis",["runtime_source"]=source,
                ["full_encoder_validated"]=false,["end_to_end_validated"]=false,["phase"]="audit",["diagnostic_readbacks_only"]=true};
            void Save(){Directory.CreateDirectory(Path.GetDirectoryName(output));File.WriteAllText(output,report.ToString());}
            void Check(string path,string sha){using var stream=File.OpenRead(path);Assert.That(PlayerFileHash.Compute(stream,out _),Is.EqualTo(sha));}
            Save();
            try
            {
                var locked=JObject.Parse(File.ReadAllText(Path.Combine(Application.dataPath,"../Packages/packages-lock.json")));
                foreach(var package in new[]{"com.ayutaz.embeddinggemma","com.ayutaz.embeddinggemma.validation"})Assert.That(locked["dependencies"][package].Value<string>("hash"),Is.EqualTo(source));
                var referencePath=Path.Combine(root,"vision-diagnostics.json");Check(referencePath,"998b7d0d200b257b0bc19de32f45ef27be4328088d2d51706ea956cc3b4461ce");
                var reference=JObject.Parse(File.ReadAllText(referencePath));Assert.That(reference.Value<bool>("trace_matches_original_saved"),Is.True);
                var modelPath=Path.Combine(root,"vision-stages-fp32.sentis");Check(modelPath,"63f25d442e6921529e009c4dbf9ca7cec7528499959aaf3a043f072723135508");
                var original=TextModelFile.Load(modelPath);
                var model=ModelTensorSlice.Extract(original,original.outputs[3].index,"layer0",new[]{new Model.Input{
                    name="patch",index=original.outputs[0].index,dataType=DataType.Float,shape=new DynamicTensorShape(1,2520,768)}});
                Assert.That(model.inputs.Count,Is.EqualTo(1));report["layer_count"]=model.layers.Count;
                report["layer_types"]=JObject.FromObject(model.layers.GroupBy(layer=>layer.GetType().Name).ToDictionary(group=>group.Key,group=>group.Count()));
                var constants=model.constants.ToDictionary(value=>value.index);
                var projections=model.layers.Where(layer=>layer.GetType().Name=="MatMul"&&layer.inputs.Length==2
                    &&!constants.ContainsKey(layer.inputs[0])&&constants.TryGetValue(layer.inputs[1],out var w)&&w.shape.rank==2).ToArray();
                var wanted=new HashSet<int>(projections.SelectMany(layer=>new[]{layer.inputs[0],layer.outputs[0]}));
                report["static_projection_count"]=projections.Length;
                var captures=Path.ChangeExtension(output,null)+"-captures";Assert.That(Directory.Exists(captures),Is.False);Directory.CreateDirectory(captures);
                TextModelFile.Save(model,Path.Combine(captures,"layer0.sentis"));
                var files=new JObject();report["captures"]=files;
                void Capture(string name,float[] data,int[] shape)
                {
                    var path=Path.Combine(captures,name+"-fp32.bin");var bytes=new byte[data.Length*4];Buffer.BlockCopy(data,0,bytes,0,bytes.Length);File.WriteAllBytes(path,bytes);
                    using var stream=File.OpenRead(path);files[name]=new JObject{["file"]=path,["sha256"]=PlayerFileHash.Compute(stream,out _),["shape"]=JArray.FromObject(shape)};
                }
                using(var auditWorker=new Worker(model,BackendType.GPUCompute))
                {
                    var cpu=(HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallback",BindingFlags.NonPublic|BindingFlags.Instance).GetValue(auditWorker);
                    var flush=(HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallbackShouldFlushGPU",BindingFlags.NonPublic|BindingFlags.Instance).GetValue(auditWorker);
                    var count=cpu.Intersect(model.layers.SelectMany(layer=>layer.outputs).Concat(model.inputs.Select(value=>value.index))).Count();
                    report["cpu_computation_or_input_count"]=count;report["readback_flush_count"]=flush.Count;Assert.That(count,Is.Zero);Assert.That(flush.Count,Is.Zero);
                }
                var inputPath=Path.Combine(root,"native-stage-0.bin");var expectedPath=Path.Combine(root,"native-stage-3.bin");
                foreach(var path in new[]{inputPath,expectedPath})Check(path,reference["files"][Path.GetFileName(path)].Value<string>("sha256"));
                using var input=new Tensor<float>(new TensorShape(1,2520,768),Floats(inputPath));ComputeTensorData.Pin(input);
                report["phase"]="observe";Save();var observed=WorkerObservation.ObserveIndexes(model,BackendType.GPUCompute,input,wanted);
                Assert.That(observed.FinalMatchesBaselineBits,Is.True);
                for(var i=0;i<projections.Length;i++)
                {
                    var projection=projections[i];
                    foreach(var pair in new[]{("input",projection.inputs[0]),("output",projection.outputs[0])})
                    {
                        var snapshot=observed.Snapshots.Single(value=>value.TensorIndex==pair.Item2);Assert.That(snapshot.Backend,Is.EqualTo(BackendType.GPUCompute));
                        Capture("projection-"+i+"-"+pair.Item1,snapshot.Values,Enumerable.Range(0,snapshot.Shape.rank).Select(axis=>snapshot.Shape[axis]).ToArray());
                    }
                    var weight=constants[projection.inputs[1]];var raw=(ArraySegment<byte>)typeof(Constant).GetField("array",BindingFlags.NonPublic|BindingFlags.Instance).GetValue(weight);
                    var values=new float[raw.Count/4];Buffer.BlockCopy(raw.Array,raw.Offset,values,0,raw.Count);
                    Capture("projection-"+i+"-weight",values,Enumerable.Range(0,weight.shape.rank).Select(axis=>weight.shape[axis]).ToArray());
                }
                Capture("layer0-output",observed.Final,new[]{1,2520,768});var expected=Floats(expectedPath);
                int bits=0,bad=0;double maximum=0,ratio=0;
                Assert.That(observed.Final.Length,Is.EqualTo(expected.Length));
                for(var i=0;i<expected.Length;i++)
                {
                    var value=observed.Final[i];Assert.That(float.IsNaN(value)||float.IsInfinity(value),Is.False);
                    if(BitConverter.SingleToInt32Bits(value)!=BitConverter.SingleToInt32Bits(expected[i]))bits++;
                    var d=Math.Abs((double)value-expected[i]);var allowed=1e-5+1e-4*Math.Abs((double)expected[i]);if(d>allowed)bad++;maximum=Math.Max(maximum,d);ratio=Math.Max(ratio,d/allowed);
                }
                report["normal_final_matches_observed_bits"]=true;report["actual_backend"]="GPUCompute";report["bit_differences"]=bits;report["mismatched_elements"]=bad;report["maximum_absolute_error"]=maximum;report["maximum_tolerance_ratio"]=ratio;report["phase"]="captured";Save();
                Assert.That(bad,Is.Zero,"Keep the unchanged original layer-0 FP32 tolerance.");report["success"]=true;
            }
            catch(Exception error){report["error"]=error.ToString();throw;}
            finally{Save();}
        }
        static float[] Floats(string path){var bytes=File.ReadAllBytes(path);var data=new float[bytes.Length/4];Buffer.BlockCopy(bytes,0,data,0,bytes.Length);return data;}
    }
}
