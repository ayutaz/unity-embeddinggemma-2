using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEngine;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class VisionOriginalRotaryGpuTests
    {
        [TestCase("Cos","cat","8e7ad14c37f14fb5b11e70b08701181c67ad24f4a874385eee5dd63655b39b75")]
        [TestCase("Sin","cat_1","1eeda3ae4a8b5bf7f09e29b6d6c763054d9b49ec6de551e09d54a080232a331e")]
        public void OriginalSavedRotaryMustMatchFrozenNativeOnGpu(string kind,string node,string expectedSha)
        {
            var source=Environment.GetEnvironmentVariable("EG_ACCEPTANCE_SOURCE");
            var output=Path.GetFullPath(Path.Combine(Application.dataPath,"../artifacts/vision-original-rotary-gpu-"+source+"-"+kind.ToLowerInvariant()+".json"));
            Assert.That(File.Exists(output),Is.False);
            var report=new JObject{["success"]=false,["runtime_source"]=source,["scope"]="one_original_constant_position_rotary_gpu_slice_vs_golden_reproducing_native",
                ["full_encoder_validated"]=false,["end_to_end_validated"]=false};
            void Save(){Directory.CreateDirectory(Path.GetDirectoryName(output));File.WriteAllText(output,report.ToString());}
            void Check(string path,string sha){using var stream=File.OpenRead(path);Assert.That(PlayerFileHash.Compute(stream,out _),Is.EqualTo(sha));}
            Save();
            try
            {
                var locked=JObject.Parse(File.ReadAllText(Path.Combine(Application.dataPath,"../Packages/packages-lock.json")));
                foreach(var package in new[]{"com.ayutaz.embeddinggemma","com.ayutaz.embeddinggemma.validation"})Assert.That(locked["dependencies"][package].Value<string>("hash"),Is.EqualTo(source));
                var path=Path.GetFullPath(Path.Combine(Application.dataPath,"../../k/artifacts/vision-encoder/square-fp32.sentis"));
                Check(path,"4ecd0106463e833588c7584a58ea4973fdbab1eb1a56389c2f1597d3e7df34e7");var original=TextModelFile.Load(path);
                var activation=original.layers.Single(layer=>layer.GetType().Name==kind);
                var analysis=typeof(Worker).Assembly.GetType("Unity.InferenceEngine.Compiler.Analyser.PartialInferenceAnalysis");
                var context=analysis.GetMethod("InferModelPartialTensors").Invoke(null,new object[]{original});
                TensorShape Shape(int index)
                {
                    var partial=context.GetType().GetMethod("GetPartialTensor").Invoke(context,new object[]{index});
                    var shape=(DynamicTensorShape)partial.GetType().GetProperty("shape").GetValue(partial);Assert.That(shape.IsStatic(),Is.True);return shape.ToTensorShape();
                }
                var dependent=new HashSet<int>(activation.outputs);Layer concat=null;
                foreach(var layer in original.layers)
                {
                    if(!layer.inputs.Any(dependent.Contains))continue;
                    dependent.UnionWith(layer.outputs);
                    if(layer.GetType().Name=="Concat"&&Shape(layer.outputs[0]).length==2520*64){concat=layer;break;}
                }
                Assert.That(concat,Is.Not.Null,"Original rotary duplication output is missing.");
                var shape=Shape(concat.outputs[0]);Assert.That(shape==new TensorShape(1,2520,64)||shape==new TensorShape(2520,64),Is.True);
                var slice=ModelTensorSlice.Extract(original,concat.outputs[0],"rotary");Assert.That(slice.inputs.Count,Is.Zero,"Fixed positions must be independent of image pixel values.");
                report["slice_layer_count"]=slice.layers.Count;report["shape"]=JArray.FromObject(shape.ToArray());
                var expectedPath=Path.GetFullPath(Path.Combine(Application.dataPath,"../../numerical-acceptance-state/r38716679261/"+node+"-fp32.bin"));Check(expectedPath,expectedSha);
                using var worker=new Worker(slice,BackendType.GPUCompute);
                const BindingFlags fields=BindingFlags.Instance|BindingFlags.NonPublic;
                var cpu=(HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallback",fields).GetValue(worker);
                var flush=(HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallbackShouldFlushGPU",fields).GetValue(worker);
                var data=new HashSet<int>(slice.layers.SelectMany(layer=>layer.outputs));Assert.That(cpu.Intersect(data).Count(),Is.Zero);Assert.That(flush.Count,Is.Zero);
                worker.Schedule();var tensor=(Tensor<float>)worker.PeekOutput();Assert.That(tensor.backendType,Is.EqualTo(BackendType.GPUCompute));
                var actual=tensor.DownloadToArray();var bytes=File.ReadAllBytes(expectedPath);var expected=new float[bytes.Length/4];Buffer.BlockCopy(bytes,0,expected,0,bytes.Length);
                Assert.That(actual.Length,Is.EqualTo(expected.Length));int bits=0,bad=0;double maximum=0;
                for(var i=0;i<actual.Length;i++)
                {
                    Assert.That(float.IsNaN(actual[i])||float.IsInfinity(actual[i]),Is.False);
                    if(BitConverter.SingleToInt32Bits(actual[i])!=BitConverter.SingleToInt32Bits(expected[i]))bits++;
                    var delta=Math.Abs((double)actual[i]-expected[i]);if(delta>1e-5+1e-4*Math.Abs((double)expected[i]))bad++;maximum=Math.Max(maximum,delta);
                }
                var raw=new byte[actual.Length*4];Buffer.BlockCopy(actual,0,raw,0,raw.Length);File.WriteAllBytes(Path.ChangeExtension(output,"bin"),raw);
                using(var stream=new MemoryStream(raw,false))report["capture_sha256"]=PlayerFileHash.Compute(stream,out _);
                report["bit_differences"]=bits;report["mismatched_elements"]=bad;report["maximum_absolute_error"]=maximum;
                report["actual_backend"]="GPUCompute";report["cpu_computation_or_input_count"]=0;report["readback_flush_count"]=0;Save();
                Assert.That(bits,Is.Zero,"Preserve the frozen original native rotary bits.");report["success"]=true;
            }
            catch(Exception error){report["error"]=error.ToString();throw;}
            finally{Save();}
        }
    }
}
