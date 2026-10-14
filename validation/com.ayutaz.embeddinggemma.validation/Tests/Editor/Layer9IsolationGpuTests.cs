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
    public sealed class Layer9IsolationGpuTests
    {
        [Test]
        public void OriginalInputLayerNineMustPreserveObservedReplayAndNativeNumericGate()
        {
            Assert.That(SystemInfo.supportsComputeShaders,Is.True,"GPU required; no skip.");
            var source=Environment.GetEnvironmentVariable("EG_ACCEPTANCE_SOURCE");
            var root=Path.Combine(Application.dataPath,"Layer9Isolation");
            var specification=JObject.Parse(File.ReadAllText(Path.Combine(root,"layer9-isolation.json")));
            var output=Path.GetFullPath(Path.Combine(Application.dataPath,"../artifacts/layer9-isolation-gpu-"+source+".json"));
            Assert.That(File.Exists(output),Is.False);
            var report=new JObject { ["success"]=false,["scope"]="one_layer9_original_input_gpu_isolation_only",["source"]=source,
                ["requested_backend"]="GPUCompute",["phase"]="audit",["full_encoder_validated"]=false,["end_to_end_validated"]=false };
            void Save(){Directory.CreateDirectory(Path.GetDirectoryName(output));File.WriteAllText(output,report.ToString());}
            void Check(string path,string sha){using var stream=File.OpenRead(path);Assert.That(PlayerFileHash.Compute(stream,out _),Is.EqualTo(sha));}
            Save();
            try
            {
                var locked=JObject.Parse(File.ReadAllText(Path.Combine(Application.dataPath,"../Packages/packages-lock.json")));
                foreach(var p in new[]{"com.ayutaz.embeddinggemma","com.ayutaz.embeddinggemma.validation"})Assert.That(locked["dependencies"][p].Value<string>("hash"),Is.EqualTo(source));
                Assert.That(specification.Value<string>("original_reference_source"),Is.EqualTo("4edbe17a8105233c5aaa1514faa2862b08a4f7e0"));
                Check(Path.Combine(root,"layer9.onnx"),"574942834f4416b257855342a1fe87ef0e1616bb8a1d8952ed41f212f5f689d4");
                foreach(var f in (JObject)specification["files"])Check(f.Value.Value<string>("path"),f.Value.Value<string>("sha256"));
                var model=ModelLoader.Load(AssetDatabase.LoadAssetAtPath<ModelAsset>("Assets/Layer9Isolation/layer9.onnx"));
                model=CompensatedModel.Create(model,64);report["layer_count"]=model.layers.Count;
                using var worker=new Worker(model,BackendType.GPUCompute);
                var cpu=(HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallback",BindingFlags.NonPublic|BindingFlags.Instance).GetValue(worker);
                var flush=(HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallbackShouldFlushGPU",BindingFlags.NonPublic|BindingFlags.Instance).GetValue(worker);
                var count=cpu.Intersect(model.layers.SelectMany(l=>l.outputs).Concat(model.inputs.Select(i=>i.index))).Count();
                report["cpu_computation_or_input_count"]=count;report["readback_flush_count"]=flush.Count;Assert.That(count,Is.Zero);Assert.That(flush.Count,Is.Zero);
                foreach(var phase in new[]{"observed","original"})
                {
                    report["phase"]=phase;Save();
                    var files=specification["files"];
                    using var input=new Tensor<float>(new TensorShape(1,280,1024),Floats(files[phase+"-input"].Value<string>("path")));
                    ComputeTensorData.Pin(input);worker.Schedule(input);
                    var tensor=(Tensor<float>)worker.PeekOutput();Assert.That(tensor.backendType,Is.EqualTo(BackendType.GPUCompute));
                    var actual=tensor.DownloadToArray();var expected=Floats(files[phase+"-expected"].Value<string>("path"));Assert.That(actual.Length,Is.EqualTo(expected.Length));
                    Assert.That(actual.All(v=>!float.IsNaN(v)&&!float.IsInfinity(v)),Is.True);
                    int bits=0,mismatch=0;double maximum=0,ratio=0;
                    for(var i=0;i<actual.Length;i++){if(BitConverter.SingleToInt32Bits(actual[i])!=BitConverter.SingleToInt32Bits(expected[i]))bits++;var d=Math.Abs((double)actual[i]-expected[i]);var allowed=1e-5+1e-4*Math.Abs((double)expected[i]);if(d>allowed)mismatch++;maximum=Math.Max(maximum,d);ratio=Math.Max(ratio,d/allowed);}
                    var bytes=new byte[actual.Length*4];Buffer.BlockCopy(actual,0,bytes,0,bytes.Length);File.WriteAllBytes(Path.ChangeExtension(output,null)+"-"+phase+".bin",bytes);
                    report[phase]=new JObject { ["exact_differences"]=bits,["mismatches"]=mismatch,["maximum_absolute_error"]=maximum,["maximum_tolerance_ratio"]=ratio };Save();
                    if(phase=="observed")Assert.That(bits,Is.Zero,"Extraction must replay the unchanged observed full-layer bits before causal inference.");
                    else Assert.That(mismatch,Is.Zero,"Original input must meet original native layer-nine gate.");
                }
                report["phase"]="completed";report["success"]=true;
            }
            catch(Exception error){report["error"]=error.ToString();throw;}
            finally{Save();}
        }
        static float[] Floats(string path){var bytes=File.ReadAllBytes(path);var a=new float[bytes.Length/4];Buffer.BlockCopy(bytes,0,a,0,bytes.Length);return a;}
    }
}
