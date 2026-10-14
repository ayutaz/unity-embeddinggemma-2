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
    public sealed class FrozenBlockConvGpuTests
    {
        [Test]
        public void OriginalSecondConvMustMatchAllSavedNativeBitsOnGpu()
        {
            Assert.That(SystemInfo.supportsComputeShaders,Is.True,"Actual GPU required; no skip.");
            var source=Environment.GetEnvironmentVariable("EG_ACCEPTANCE_SOURCE");
            var root=Path.Combine(Application.dataPath,"FrozenBlockConvProbe");
            var spec=JObject.Parse(File.ReadAllText(Path.Combine(root,"frozen-conv.json")));
            var variant=spec.Value<string>("variant");
            var output=Path.GetFullPath(Path.Combine(Application.dataPath,"../artifacts/frozen-block-conv-gpu-"+source+"-"+variant+".json"));
            Assert.That(File.Exists(output),Is.False);
            var report=new JObject { ["success"]=false,["source"]=source,["variant"]=variant,
                ["scope"]="one_frozen_conv2_original_silence_gpu_bits_only",["phase"]="audit",
                ["gpu"]=SystemInfo.graphicsDeviceName,["graphics_api"]=SystemInfo.graphicsDeviceType.ToString(),
                ["full_encoder_validated"]=false,["end_to_end_validated"]=false };
            void Save(){Directory.CreateDirectory(Path.GetDirectoryName(output));File.WriteAllText(output,report.ToString());}
            Save();
            try
            {
                var locked=JObject.Parse(File.ReadAllText(Path.Combine(Application.dataPath,"../Packages/packages-lock.json")));
                foreach(var p in new[]{"com.ayutaz.embeddinggemma","com.ayutaz.embeddinggemma.validation"})
                    Assert.That(locked["dependencies"][p].Value<string>("hash"),Is.EqualTo(source));
                Assert.That(spec.Value<string>("reference_source"),Is.EqualTo("4edbe17a8105233c5aaa1514faa2862b08a4f7e0"));
                Assert.That(spec.Value<bool>("expected_unchanged"),Is.True);
                foreach(var file in (JObject)spec["files"])
                {
                    var path=Path.Combine(root,file.Key);using var stream=File.OpenRead(path);
                    Assert.That(PlayerFileHash.Compute(stream,out _),Is.EqualTo(file.Value.Value<string>("sha256")));
                    Assert.That(new FileInfo(path).Length,Is.EqualTo(file.Value.Value<long>("bytes")));
                }
                var model=ModelLoader.Load(AssetDatabase.LoadAssetAtPath<ModelAsset>("Assets/FrozenBlockConvProbe/conv.onnx"));
                report["layer_count"]=model.layers.Count;
                using var worker=new Worker(model,BackendType.GPUCompute);
                var cpu=(HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallback",BindingFlags.NonPublic|BindingFlags.Instance).GetValue(worker);
                var flush=(HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallbackShouldFlushGPU",BindingFlags.NonPublic|BindingFlags.Instance).GetValue(worker);
                var count=cpu.Intersect(model.layers.SelectMany(l=>l.outputs).Concat(model.inputs.Select(i=>i.index))).Count();
                report["cpu_computation_or_input_count"]=count;report["readback_flush_count"]=flush.Count;
                Assert.That(count,Is.Zero);Assert.That(flush.Count,Is.Zero);
                using var input=new Tensor<float>(new TensorShape(spec["input_shape"].ToObject<int[]>()),Floats(Path.Combine(root,"input.bin")));
                ComputeTensorData.Pin(input);report["phase"]="execute";Save();worker.Schedule(input);
                var tensor=(Tensor<float>)worker.PeekOutput();Assert.That(tensor.backendType,Is.EqualTo(BackendType.GPUCompute));
                report["actual_backend"]=tensor.backendType.ToString();
                var actual=tensor.DownloadToArray();var expected=Floats(Path.Combine(root,"expected.bin"));
                Assert.That(actual.Length,Is.EqualTo(expected.Length));Assert.That(actual.All(v=>!float.IsNaN(v)&&!float.IsInfinity(v)),Is.True);
                int bits=0,mismatch=0;double maximum=0;
                for(var i=0;i<actual.Length;i++){if(BitConverter.SingleToInt32Bits(actual[i])!=BitConverter.SingleToInt32Bits(expected[i]))bits++;var d=Math.Abs((double)actual[i]-expected[i]);maximum=Math.Max(maximum,d);if(d>1e-5+1e-4*Math.Abs((double)expected[i]))mismatch++;}
                report["phase"]="completed";report["exact_differences"]=bits;report["mismatched_elements"]=mismatch;report["maximum_absolute_error"]=maximum;
                Save();Assert.That(bits,Is.Zero,"Strong local source-order target; original full-encoder tolerance remains unchanged.");
                report["success"]=true;
            }
            catch(Exception error){report["error"]=error.ToString();throw;}
            finally{Save();}
        }
        static float[] Floats(string path){var bytes=File.ReadAllBytes(path);var values=new float[bytes.Length/4];Buffer.BlockCopy(bytes,0,values,0,bytes.Length);return values;}
    }
}
