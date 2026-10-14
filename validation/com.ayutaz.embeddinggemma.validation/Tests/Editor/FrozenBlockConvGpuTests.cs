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
            => Execute("FrozenBlockConvProbe","conv","frozen-block-conv","one_frozen_conv2_original_silence_gpu_bits_only");

        [Test]
        public void OriginalInputProjectionMustMatchAllSavedNativeBitsOnGpu()
            => Execute("FrozenInputProjectionProbe","projection","frozen-input-projection","one_original_280_row_input_projection_gpu_bits_only");

        [TestCase("ffw_layer_1")]
        [TestCase("ffw_layer_2")]
        public void OriginalFirstFeedForwardProjectionMustMatchSavedNativeBitsOnGpu(string part)
            => Execute("FrozenFirstFeedForward-"+part,"projection","frozen-first-ff-"+part,"one_original_first_feed_forward_projection_gpu_bits_only");

        void Execute(string folder,string graph,string reportStem,string scope)
        {
            Assert.That(SystemInfo.supportsComputeShaders,Is.True,"Actual GPU required; no skip.");
            var source=Environment.GetEnvironmentVariable("EG_ACCEPTANCE_SOURCE");
            var root=Path.Combine(Application.dataPath,folder);
            var spec=JObject.Parse(File.ReadAllText(Path.Combine(root,"frozen-conv.json")));
            var variant=spec.Value<string>("variant");
            var output=Path.GetFullPath(Path.Combine(Application.dataPath,"../artifacts/"+reportStem+"-gpu-"+source+"-"+variant+".json"));
            Assert.That(File.Exists(output),Is.False);
            var report=new JObject { ["success"]=false,["source"]=source,["variant"]=variant,
                ["scope"]=scope,["phase"]="audit",
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
                if(folder=="FrozenInputProjectionProbe")
                {
                    Assert.That(spec.Value<bool>("all_original_rows_preserved"),Is.True);
                    Assert.That(spec.Value<string>("original_expected_sha256"),Is.EqualTo("dc54774840fed958a310c6092e5c5c3483de74be52df7914d2bd128a99192969"));
                    Assert.That(spec.Value<string>("weight_sha256"),Is.EqualTo("a59650e614c2746ee18bbb23c475f65677beeb9ffe49350854af2aad873052a1"));
                    Assert.That(spec["input_shape"].ToObject<int[]>(),Is.EqualTo(new[]{1,280,1024}));
                }
                if(folder.StartsWith("FrozenFirstFeedForward-",StringComparison.Ordinal))
                {
                    var first=folder.EndsWith("ffw_layer_1",StringComparison.Ordinal);
                    Assert.That(spec.Value<bool>("all_original_rows_preserved"),Is.True);
                    Assert.That(spec.Value<string>("native_report_sha256"),Is.EqualTo("99529dc0fa6c74825b23fca653d82091797ff1e5267d26c5cdae723fcf86f501"));
                    Assert.That(spec.Value<string>("original_expected_sha256"),Is.EqualTo(first?"e5ac1da2d80e0013714a5be3e3336272bc954176baf71bd932107db62b8318b4":"efafce302f7c558de3f9d62778a6be2d419d841e98a4a797e79ee7222fce1686"));
                    Assert.That(spec.Value<string>("weight_sha256"),Is.EqualTo(first?"d598344ee23d0545a76fa6d95173fbc886e8f2dd46d5907e6bd50b6cfa4891df":"50ff67bc3a9a8704f925183042667884c6a2114de99565c2fa93f79570bd77b6"));
                    Assert.That(spec["input_shape"].ToObject<int[]>(),Is.EqualTo(new[]{1,280,first?1024:4096}));
                }
                foreach(var file in (JObject)spec["files"])
                {
                    var path=Path.Combine(root,file.Key);using var stream=File.OpenRead(path);
                    Assert.That(PlayerFileHash.Compute(stream,out _),Is.EqualTo(file.Value.Value<string>("sha256")));
                    Assert.That(new FileInfo(path).Length,Is.EqualTo(file.Value.Value<long>("bytes")));
                }
                if(variant=="direct_block192")
                {
                    Assert.That(folder.StartsWith("FrozenFirstFeedForward-",StringComparison.Ordinal),Is.True);
                    ExecuteDirect(root,spec,report,Save);
                    report["success"]=true;
                    return;
                }
                var model=ModelLoader.Load(AssetDatabase.LoadAssetAtPath<ModelAsset>("Assets/"+folder+"/"+graph+".onnx"));
                if((folder=="FrozenInputProjectionProbe" || folder.StartsWith("FrozenFirstFeedForward-",StringComparison.Ordinal)) && variant=="static64")
                    model=CompensatedModel.Create(model,64);
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

        static void ExecuteDirect(string root,JObject spec,JObject report,Action save)
        {
            Assert.That(spec["files"]["weight.bin"].Value<string>("sha256"),Is.EqualTo(spec.Value<string>("weight_sha256")));
            var inputShape=new TensorShape(spec["input_shape"].ToObject<int[]>());
            var outputShape=new TensorShape(spec["output_shape"].ToObject<int[]>());
            var inputBounds=(JArray)spec["input_clips"];
            Assert.That(inputBounds.Count,Is.EqualTo(1));
            var outputBounds=(JArray)spec["output_clip"];
            Model Clip(TensorShape shape,float minimum,float maximum)
            {
                var graph=new FunctionalGraph();var input=graph.AddInput<float>(shape);
                return graph.Compile(Functional.Clamp(input,minimum,maximum));
            }
            var prefix=Clip(inputShape,inputBounds[0][0].Value<float>(),inputBounds[0][1].Value<float>());
            var suffix=Clip(outputShape,outputBounds[0].Value<float>(),outputBounds[1].Value<float>());
            using var before=new Worker(prefix,BackendType.GPUCompute);
            using var after=new Worker(suffix,BackendType.GPUCompute);
            var cpuCount=0;var flushCount=0;
            foreach(var pair in new[]{(before,prefix),(after,suffix)})
            {
                var cpu=(HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallback",BindingFlags.NonPublic|BindingFlags.Instance).GetValue(pair.Item1);
                var flush=(HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallbackShouldFlushGPU",BindingFlags.NonPublic|BindingFlags.Instance).GetValue(pair.Item1);
                cpuCount+=cpu.Intersect(pair.Item2.layers.SelectMany(l=>l.outputs).Concat(pair.Item2.inputs.Select(i=>i.index))).Count();
                flushCount+=flush.Count;
            }
            Assert.That(cpuCount,Is.Zero);Assert.That(flushCount,Is.Zero);
            report["cpu_computation_or_input_count"]=cpuCount;report["readback_flush_count"]=flushCount;
            report["implementation"]="direct GPU block192 with GPU input/output clipping";
            report["shader_dispatch_count"]=1;report["clip_layer_count"]=prefix.layers.Count+suffix.layers.Count;
            using var operation=new GpuBlockedMatmul(inputShape[2],outputShape[2],Floats(Path.Combine(root,"weight.bin")));
            using var inputTensor=new Tensor<float>(inputShape,Floats(Path.Combine(root,"input.bin")));
            ComputeTensorData.Pin(inputTensor);
            report["phase"]="execute";save();var timer=System.Diagnostics.Stopwatch.StartNew();
            before.Schedule(inputTensor);
            using var projected=operation.Execute((Tensor<float>)before.PeekOutput());
            after.Schedule(projected);
            var output=(Tensor<float>)after.PeekOutput();
            Assert.That(output.backendType,Is.EqualTo(BackendType.GPUCompute));
            var actual=output.DownloadToArray();report["gpu_execute_and_final_download_ms"]=timer.Elapsed.TotalMilliseconds;
            var expected=Floats(Path.Combine(root,"expected.bin"));Assert.That(actual.Length,Is.EqualTo(expected.Length));
            int bits=0,mismatch=0;double maximum=0;
            for(var i=0;i<actual.Length;i++)
            {
                Assert.That(float.IsNaN(actual[i])||float.IsInfinity(actual[i]),Is.False);
                if(BitConverter.SingleToInt32Bits(actual[i])!=BitConverter.SingleToInt32Bits(expected[i]))bits++;
                var difference=Math.Abs((double)actual[i]-expected[i]);maximum=Math.Max(maximum,difference);
                if(difference>1e-5+1e-4*Math.Abs((double)expected[i]))mismatch++;
            }
            report["actual_backend"]=output.backendType.ToString();report["exact_differences"]=bits;
            report["mismatched_elements"]=mismatch;report["maximum_absolute_error"]=maximum;report["phase"]="completed";save();
            Assert.That(bits,Is.Zero,"Keep the original fixed local GPU bit target.");
        }
        static float[] Floats(string path){var bytes=File.ReadAllBytes(path);var values=new float[bytes.Length/4];Buffer.BlockCopy(bytes,0,values,0,bytes.Length);return values;}
    }
}
