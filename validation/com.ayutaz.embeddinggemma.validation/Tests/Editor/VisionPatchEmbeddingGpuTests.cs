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
    public sealed class VisionPatchEmbeddingGpuTests
    {
        [Test]
        public void OriginalVisionPatchPrefixMustMatchNativeBitsOnGpu()
        {
            Assert.That(SystemInfo.supportsComputeShaders,Is.True,"Actual GPU required; never skip.");
            var source=Environment.GetEnvironmentVariable("EG_ACCEPTANCE_SOURCE");
            var root=Path.GetFullPath(Path.Combine(Application.dataPath,"../../k/artifacts/vision-diagnostics"));
            var output=Path.GetFullPath(Path.Combine(Application.dataPath,"../artifacts/vision-patch-prefix-gpu-"+source+".json"));
            Assert.That(File.Exists(output),Is.False);
            var report=new JObject{["success"]=false,["scope"]="original_vision_patch_prefix_gpu_with_diagnostic_captures",["runtime_source"]=source,
                ["full_encoder_validated"]=false,["end_to_end_validated"]=false,["phase"]="audit",["diagnostic_readbacks_only"]=true};
            void Save(){Directory.CreateDirectory(Path.GetDirectoryName(output));File.WriteAllText(output,report.ToString());}
            void Check(string path,string sha){using var stream=File.OpenRead(path);Assert.That(PlayerFileHash.Compute(stream,out _),Is.EqualTo(sha));}
            Save();
            try
            {
                var locked=JObject.Parse(File.ReadAllText(Path.Combine(Application.dataPath,"../Packages/packages-lock.json")));
                foreach(var package in new[]{"com.ayutaz.embeddinggemma","com.ayutaz.embeddinggemma.validation"})
                    Assert.That(locked["dependencies"][package].Value<string>("hash"),Is.EqualTo(source));
                Check(Path.Combine(root,"vision-diagnostics.json"),"998b7d0d200b257b0bc19de32f45ef27be4328088d2d51706ea956cc3b4461ce");
                var reference=JObject.Parse(File.ReadAllText(Path.Combine(root,"vision-diagnostics.json")));
                Assert.That(reference.Value<bool>("success")&&reference.Value<bool>("trace_matches_original_saved"),Is.True);
                Assert.That(reference["metadata"].Value<string>("source_commit"),Is.EqualTo("736a4684be754563bbb953a67c2023d688c4a312"));
                var modelPath=Path.Combine(root,"vision-stages-fp32.sentis");Check(modelPath,"63f25d442e6921529e009c4dbf9ca7cec7528499959aaf3a043f072723135508");
                var pixelsPath=Path.Combine(root,"pixels-fp32.bin");var expectedPath=Path.Combine(root,"native-stage-0.bin");
                foreach(var file in new[]{pixelsPath,expectedPath})Check(file,reference["files"][Path.GetFileName(file)].Value<string>("sha256"));
                var original=TextModelFile.Load(modelPath);var prefix=ModelTensorSlice.Extract(original,original.outputs[0].index,"patch_embeddings");
                report["original_layer_count"]=original.layers.Count;report["prefix_layer_count"]=prefix.layers.Count;
                var constants=prefix.constants.ToDictionary(value=>value.index);
                var projection=prefix.layers.Single(layer=>new[]{"MatMul","MatMul2D","Dense"}.Contains(layer.GetType().Name)
                    &&layer.inputs.Length>=2&&constants.TryGetValue(layer.inputs[1],out var w)&&w.shape==new TensorShape(768,768));
                var weight=constants[projection.inputs[1]];
                var raw=(ArraySegment<byte>)typeof(Constant).GetField("array",BindingFlags.NonPublic|BindingFlags.Instance).GetValue(weight);
                var values=new float[raw.Count/4];Buffer.BlockCopy(raw.Array,raw.Offset,values,0,raw.Count);
                var transposed=(bool?)projection.GetType().GetField("transposeB")?.GetValue(projection)??false;
                if(transposed){var ordered=new float[values.Length];for(var row=0;row<768;row++)for(var column=0;column<768;column++)ordered[row*768+column]=values[column*768+row];values=ordered;}
                report["projection_type"]=projection.GetType().Name;report["projection_weight_transposed"]=transposed;
                var captures=Path.ChangeExtension(output,null)+"-captures";Assert.That(Directory.Exists(captures),Is.False);Directory.CreateDirectory(captures);
                var files=new JObject();report["captures"]=files;
                void Capture(string name,float[] data,int[] shape)
                {
                    var path=Path.Combine(captures,name+"-fp32.bin");var bytes=new byte[data.Length*4];Buffer.BlockCopy(data,0,bytes,0,bytes.Length);File.WriteAllBytes(path,bytes);
                    using var stream=File.OpenRead(path);files[name]=new JObject{["file"]=path,["sha256"]=PlayerFileHash.Compute(stream,out _),["shape"]=JArray.FromObject(shape)};
                }
                Capture("projection_weight",values,new[]{768,768});
                TextModelFile.Save(prefix,Path.Combine(captures,"patch-prefix.sentis"));
                using var input=new Tensor<float>(new TensorShape(1,2520,768),Floats(pixelsPath));ComputeTensorData.Pin(input);
                report["phase"]="observe_prefix";Save();
                var observed=WorkerObservation.ObserveIndexes(prefix,BackendType.GPUCompute,input,new HashSet<int>{projection.inputs[0],projection.outputs[0]});
                Assert.That(observed.FinalMatchesBaselineBits,Is.True);Assert.That(observed.UnavailableTensorIndexes,Is.Empty);
                foreach(var pair in new[]{("projection_input",projection.inputs[0]),("projection_output",projection.outputs[0])})
                {
                    var snapshot=observed.Snapshots.Single(value=>value.TensorIndex==pair.Item2);Assert.That(snapshot.Backend,Is.EqualTo(BackendType.GPUCompute));
                    Capture(pair.Item1,snapshot.Values,Enumerable.Range(0,snapshot.Shape.rank).Select(i=>snapshot.Shape[i]).ToArray());
                }
                Capture("patch_output",observed.Final,new[]{1,2520,768});
                var expected=Floats(expectedPath);Assert.That(observed.Final.Length,Is.EqualTo(expected.Length));int bits=0,bad=0;double ratio=0,maximum=0;
                for(var i=0;i<expected.Length;i++)
                {
                    var actual=observed.Final[i];Assert.That(float.IsNaN(actual)||float.IsInfinity(actual),Is.False);
                    if(BitConverter.SingleToInt32Bits(actual)!=BitConverter.SingleToInt32Bits(expected[i]))bits++;
                    var d=Math.Abs((double)actual-expected[i]);var allowed=1e-5+1e-4*Math.Abs((double)expected[i]);if(d>allowed)bad++;ratio=Math.Max(ratio,d/allowed);maximum=Math.Max(maximum,d);
                }
                report["actual_backend"]="GPUCompute";report["normal_prefix_final_matches_observed_bits"]=true;
                report["bit_differences"]=bits;report["mismatched_elements"]=bad;report["maximum_absolute_error"]=maximum;report["maximum_tolerance_ratio"]=ratio;report["phase"]="captured";Save();
                Assert.That(bits,Is.Zero,"Keep the original native patch bits while isolating the first image difference.");report["success"]=true;
            }
            catch(Exception error){report["error"]=error.ToString();throw;}
            finally{Save();}
        }
        static float[] Floats(string path){var bytes=File.ReadAllBytes(path);var data=new float[bytes.Length/4];Buffer.BlockCopy(bytes,0,data,0,bytes.Length);return data;}
    }
}
