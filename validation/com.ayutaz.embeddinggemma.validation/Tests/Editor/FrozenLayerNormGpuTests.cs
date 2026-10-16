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
    public sealed class FrozenLayerNormGpuTests
    {
        [TestCase("normal")]
        [TestCase("mean_inverse")]
        public void FrozenFirstNormRowsMatchIndependentNativeBits(string name) => Execute(name,"FrozenLayerNormProbe");

        [Test]
        public void FrozenSecondNormRowsMatchOriginalNativeBits() => Execute("normal","FrozenSecondNormProbe");

        [TestCase("normal")]
        [TestCase("mean_inverse")]
        public void FrozenRmsRowsMatchIndependentNativeBits(string name) => Execute(name,"FrozenRmsProbe");

        [TestCase("normal")]
        [TestCase("mean_inverse")]
        public void FrozenOuterRmsRowsMatchIndependentNativeBits(string name) => Execute(name,"FrozenOuterRmsProbe");

        void Execute(string name,string folder)
        {
            Assert.That(SystemInfo.supportsComputeShaders,Is.True,"Actual GPU required; never skip.");
            var source=Environment.GetEnvironmentVariable("EG_ACCEPTANCE_SOURCE");
            var root=Path.Combine(Application.dataPath,folder);
            var specification=JObject.Parse(File.ReadAllText(Path.Combine(root,"frozen-probe.json")));
            var probeSha=specification.Value<string>("graph_variant");
            var stem=folder=="FrozenSecondNormProbe"?"frozen-second-norm-gpu-":folder=="FrozenOuterRmsProbe"?"frozen-outer-rms-gpu-":folder=="FrozenRmsProbe"?"frozen-rms-gpu-":"frozen-layernorm-gpu-";
            var output=Path.GetFullPath(Path.Combine(Application.dataPath,"../artifacts/"+stem+source+"-"+probeSha+"-"+name+".json"));
            Assert.That(File.Exists(output),Is.False);
            var report=new JObject { ["success"]=false,["scope"]="first_native_conv_unique_rows_gpu_norm_component_only",
                ["source"]=source,["probe_variant"]=probeSha,["case"]=name,["requested_backend"]="GPUCompute",
                ["full_encoder_validated"]=false,["end_to_end_validated"]=false,["phase"]="audit" };
            void Save(){Directory.CreateDirectory(Path.GetDirectoryName(output));File.WriteAllText(output,report.ToString());}
            Save();
            try
            {
                var locked=JObject.Parse(File.ReadAllText(Path.Combine(Application.dataPath,"../Packages/packages-lock.json")));
                foreach(var package in new[]{"com.ayutaz.embeddinggemma","com.ayutaz.embeddinggemma.validation"})
                    Assert.That(locked["dependencies"][package].Value<string>("hash"),Is.EqualTo(source));
                if(folder=="FrozenSecondNormProbe")
                {
                    Assert.That(specification.Value<string>("reference_source"),Is.EqualTo("4edbe17a8105233c5aaa1514faa2862b08a4f7e0"));
                    Assert.That(specification.Value<int>("full_original_row_count"),Is.EqualTo(8960));
                    report["reference_context"]="original Conv2 and Norm1 bytes, all repeated positions preserved by exact input grouping";
                }
                else if(folder=="FrozenOuterRmsProbe")
                {
                    Assert.That(specification.Value<bool>("original_native_whole_lconv_bits_match"),Is.True);
                    Assert.That(specification["reference_input_strides"].ToObject<int[]>(),Is.EqualTo(new[]{286720,1,280}));
                    report["reference_context"]="derived conv_norm input; full native lconv replay matches saved original bits";
                }
                else Assert.That(specification.Value<bool>("native_normal_matches_original_bits"),Is.True);
                Assert.That(specification.Value<bool>("all_original_rows_preserved"),Is.True);
                foreach(var file in (JObject)specification["files"])
                {
                    var path=Path.Combine(root,file.Key);using var stream=File.OpenRead(path);
                    Assert.That(PlayerFileHash.Compute(stream,out _),Is.EqualTo(file.Value.Value<string>("sha256")));
                    Assert.That(new FileInfo(path).Length,Is.EqualTo(file.Value.Value<long>("bytes")));
                }
                report["scope"]=specification.Value<string>("scope")??report.Value<string>("scope");
                var asset=AssetDatabase.LoadAssetAtPath<ModelAsset>("Assets/"+folder+"/"+name+".onnx");Assert.That(asset,Is.Not.Null);
                var model=ModelLoader.Load(asset);
                report["layer_kinds"]=JObject.FromObject(model.layers.GroupBy(l=>l.GetType().Name).ToDictionary(g=>g.Key,g=>g.Count()));
                using var worker=new Worker(model,BackendType.GPUCompute);
                var cpu=(HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallback",BindingFlags.NonPublic|BindingFlags.Instance).GetValue(worker);
                var flush=(HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallbackShouldFlushGPU",BindingFlags.NonPublic|BindingFlags.Instance).GetValue(worker);
                var count=cpu.Intersect(model.layers.SelectMany(l=>l.outputs).Concat(model.inputs.Select(i=>i.index))).Count();
                report["cpu_computation_or_input_count"]=count;report["readback_flush_count"]=flush.Count;
                Assert.That(count,Is.Zero);Assert.That(flush.Count,Is.Zero);
                using var input=new Tensor<float>(new TensorShape(specification["input_shape"].ToObject<int[]>()),Floats(Path.Combine(root,"input.bin")));
                ComputeTensorData.Pin(input);report["phase"]="execute";Save();worker.Schedule(input);
                var tensor=(Tensor<float>)worker.PeekOutput();Assert.That(tensor.backendType,Is.EqualTo(BackendType.GPUCompute));
                report["actual_backend"]=tensor.backendType.ToString();report["gpu"]=SystemInfo.graphicsDeviceName;
                var actual=tensor.DownloadToArray();var expected=Floats(Path.Combine(root,name+"-expected.bin"));Assert.That(actual.Length,Is.EqualTo(expected.Length));
                Assert.That(actual.All(v=>!float.IsNaN(v)&&!float.IsInfinity(v)),Is.True);
                var bytes=new byte[actual.Length*4];Buffer.BlockCopy(actual,0,bytes,0,bytes.Length);File.WriteAllBytes(Path.ChangeExtension(output,".bin"),bytes);
                var exact=0;var mismatches=0;double maximum=0;
                for(var i=0;i<actual.Length;i++){if(BitConverter.SingleToInt32Bits(actual[i])!=BitConverter.SingleToInt32Bits(expected[i]))exact++;var d=Math.Abs((double)actual[i]-expected[i]);maximum=Math.Max(maximum,d);if(d>1e-5+1e-4*Math.Abs((double)expected[i]))mismatches++;}
                report["phase"]="completed";report["exact_differences"]=exact;report["mismatched_elements"]=mismatches;report["maximum_absolute_error"]=maximum;Save();
                Assert.That(exact,Is.Zero,"Diagnostic stronger local bit target; full encoder original tolerance remains unchanged.");
                report["success"]=true;
            }
            catch(Exception error){report["error"]=error.ToString();throw;}
            finally{Save();}
        }
        static float[] Floats(string path){var bytes=File.ReadAllBytes(path);var result=new float[bytes.Length/4];Buffer.BlockCopy(bytes,0,result,0,bytes.Length);return result;}
    }
}
