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
    public sealed class VisionHeadRmsGpuTests
    {
        [TestCase(1)] [TestCase(2)] [TestCase(3)]
        public void CapturedHeadRmsMatchesIndependentCurrentCpuEquationOnGpu(int index)
        {
            Assert.That(SystemInfo.supportsComputeShaders,Is.True,"Actual GPU required; never skip.");
            var source=Environment.GetEnvironmentVariable("EG_ACCEPTANCE_SOURCE");
            var root=Path.Combine(Application.dataPath,"GpuVisionHeadRmsCandidates");
            var output=Path.GetFullPath(Path.Combine(Application.dataPath,"../artifacts/vision-head-rms-gpu-"+source+"-"+index+".json"));Assert.That(File.Exists(output),Is.False);
            var report=new JObject{["success"]=false,["scope"]="one_vision_head_rms_against_captured_input_current_cpu_equation_only",["runtime_source"]=source,["index"]=index,
                ["original_internal_native_reference_available"]=false,["full_encoder_validated"]=false,["end_to_end_validated"]=false};
            void Save(){Directory.CreateDirectory(Path.GetDirectoryName(output));File.WriteAllText(output,report.ToString());}
            void Check(string path,string expected){using var stream=File.OpenRead(path);Assert.That(PlayerFileHash.Compute(stream,out _),Is.EqualTo(expected));}
            Save();
            try
            {
                var locked=JObject.Parse(File.ReadAllText(Path.Combine(Application.dataPath,"../Packages/packages-lock.json")));
                foreach(var package in new[]{"com.ayutaz.embeddinggemma","com.ayutaz.embeddinggemma.validation"})Assert.That(locked["dependencies"][package].Value<string>("hash"),Is.EqualTo(source));
                var preparation=JObject.Parse(File.ReadAllText(Path.Combine(root,"head-rms-candidates.json")));
                Assert.That(preparation.Value<string>("producer"),Is.EqualTo(source));
                var row=((JArray)preparation["rows"]).Single(value=>value.Value<int>("index")==index);
                Assert.That(row["shape"].ToObject<int[]>(),Is.EqualTo(new[]{1,2520,12,64}));
                var onnx=Path.Combine(root,row["onnx"].Value<string>("file"));Check(onnx,row["onnx"].Value<string>("sha256"));
                var inputPath=row["input"].Value<string>("file");Check(inputPath,row["input"].Value<string>("sha256"));
                var expectedPath=Path.Combine(root,row["expected"].Value<string>("file"));Check(expectedPath,row["expected"].Value<string>("sha256"));
                if(row["scale"] is JObject scale)Check(scale.Value<string>("file"),scale.Value<string>("sha256"));
                var model=ModelLoader.Load(AssetDatabase.LoadAssetAtPath<ModelAsset>("Assets/GpuVisionHeadRmsCandidates/"+Path.GetFileName(onnx)));
                report["layer_count"]=model.layers.Count;using var worker=new Worker(model,BackendType.GPUCompute);
                var cpu=(HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallback",BindingFlags.Instance|BindingFlags.NonPublic).GetValue(worker);
                var flush=(HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallbackShouldFlushGPU",BindingFlags.Instance|BindingFlags.NonPublic).GetValue(worker);
                Assert.That(cpu.Intersect(model.layers.SelectMany(layer=>layer.outputs).Concat(model.inputs.Select(value=>value.index))).Count(),Is.Zero);Assert.That(flush.Count,Is.Zero);
                using var input=new Tensor<float>(new TensorShape(1,2520,12,64),Floats(inputPath));ComputeTensorData.Pin(input);
                worker.Schedule(input);var tensor=(Tensor<float>)worker.PeekOutput();Assert.That(tensor.backendType,Is.EqualTo(BackendType.GPUCompute));
                var actual=tensor.DownloadToArray();var expected=Floats(expectedPath);Assert.That(actual.Length,Is.EqualTo(expected.Length));Assert.That(tensor.shape,Is.EqualTo(input.shape));
                Assert.That(Array.TrueForAll(actual,value=>!float.IsNaN(value)&&!float.IsInfinity(value)),Is.True);
                int bits=0,bad=0;double maximum=0,ratio=0;
                for(var i=0;i<actual.Length;i++)
                {
                    if(BitConverter.SingleToInt32Bits(actual[i])!=BitConverter.SingleToInt32Bits(expected[i]))bits++;
                    var d=Math.Abs((double)actual[i]-expected[i]);var allowed=1e-5+1e-4*Math.Abs((double)expected[i]);if(d>allowed)bad++;maximum=Math.Max(maximum,d);ratio=Math.Max(ratio,d/allowed);
                }
                report["actual_backend"]="GPUCompute";report["cpu_computation_or_input_count"]=0;report["readback_flush_count"]=0;
                report["bit_differences"]=bits;report["mismatched_elements"]=bad;report["maximum_absolute_error"]=maximum;report["maximum_tolerance_ratio"]=ratio;Save();
                Assert.That(bits,Is.Zero,"Keep the independent current CPU equation strict gate.");report["success"]=true;
            }
            catch(Exception error){report["error"]=error.ToString();throw;}
            finally{Save();}
        }
        static float[] Floats(string path){var bytes=File.ReadAllBytes(path);var data=new float[bytes.Length/4];Buffer.BlockCopy(bytes,0,data,0,bytes.Length);return data;}
    }
}
