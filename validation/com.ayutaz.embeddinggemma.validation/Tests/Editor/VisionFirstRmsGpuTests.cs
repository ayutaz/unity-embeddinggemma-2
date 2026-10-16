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
    public sealed class VisionFirstRmsGpuTests
    {
        [Test]
        public void IsolatedFirstVisionRmsPreservesObservedGpuBitsAndCoefficientBytes()
        {
            Assert.That(SystemInfo.supportsComputeShaders,Is.True,"Actual GPU required; never skip.");
            var source=Environment.GetEnvironmentVariable("EG_ACCEPTANCE_SOURCE");
            var layerRoot=Path.GetFullPath(Path.Combine(Application.dataPath,"../artifacts/vision-layer0-original-patch-gpu-e39bdf1b5d0faf2ea424023d83076315736c6fde-captures"));
            var root=Path.GetFullPath(Path.Combine(Application.dataPath,"../../k/artifacts/vision-diagnostics"));
            var output=Path.GetFullPath(Path.Combine(Application.dataPath,"../artifacts/vision-first-rms-original-gpu-"+source+".json"));
            Assert.That(File.Exists(output),Is.False);
            var report=new JObject{["success"]=false,["scope"]="isolated_first_vision_rms_observed_gpu_replay_only",["runtime_source"]=source,
                ["native_reference_validated"]=false,["full_encoder_validated"]=false,["end_to_end_validated"]=false,["phase"]="audit"};
            void Save(){Directory.CreateDirectory(Path.GetDirectoryName(output));File.WriteAllText(output,report.ToString());}
            void Check(string path,string sha){using var stream=File.OpenRead(path);Assert.That(PlayerFileHash.Compute(stream,out _),Is.EqualTo(sha));}
            Save();
            try
            {
                var locked=JObject.Parse(File.ReadAllText(Path.Combine(Application.dataPath,"../Packages/packages-lock.json")));
                foreach(var package in new[]{"com.ayutaz.embeddinggemma","com.ayutaz.embeddinggemma.validation"})Assert.That(locked["dependencies"][package].Value<string>("hash"),Is.EqualTo(source));
                var path=Path.Combine(layerRoot,"layer0.sentis");Check(path,"d09b8884fae82ecd0245967774a0c8671efe09156c50367c45b9f133fc53983d");
                var model=TextModelFile.Load(path);var constants=model.constants.ToDictionary(value=>value.index);
                var projection=model.layers.First(layer=>layer.GetType().Name=="MatMul"&&constants.TryGetValue(layer.inputs[1],out var w)&&w.shape.rank==2);
                var prefix=ModelTensorSlice.Extract(model,projection.inputs[0],"normalized");
                var scale=prefix.constants.Single(value=>value.dataType==DataType.Float&&value.shape==new TensorShape(768));
                var scalar=prefix.layers.Single(layer=>layer.GetType().Name=="ScalarMad");
                Assert.That((float)scalar.GetType().GetField("sFloat").GetValue(scalar),Is.EqualTo(1f));
                report["epsilon"]=(float)scalar.GetType().GetField("bFloat").GetValue(scalar);Assert.That(report.Value<float>("epsilon"),Is.EqualTo(1e-6f));
                report["layer_count"]=prefix.layers.Count;
                var captures=Path.ChangeExtension(output,null)+"-captures";Assert.That(Directory.Exists(captures),Is.False);Directory.CreateDirectory(captures);
                var scaleBytes=(ArraySegment<byte>)typeof(Constant).GetField("array",BindingFlags.NonPublic|BindingFlags.Instance).GetValue(scale);
                var scalePath=Path.Combine(captures,"scale-fp32.bin");File.WriteAllBytes(scalePath,scaleBytes.ToArray());
                using(var stream=File.OpenRead(scalePath))report["scale_sha256"]=PlayerFileHash.Compute(stream,out _);
                TextModelFile.Save(prefix,Path.Combine(captures,"first-rms.sentis"));
                var inputPath=Path.Combine(root,"native-stage-0.bin");Check(inputPath,"bc4c19c9aaf9688801c99ed073fdde22f0daf6b822dc2c9c4b7a3f7c9aded636");
                var expectedPath=Path.Combine(layerRoot,"projection-0-input-fp32.bin");Check(expectedPath,"9553e0d41e4aec086ee637d4d0cfff7777586e93cd2641088dbca4edc6e6a7f4");
                using var input=new Tensor<float>(new TensorShape(1,2520,768),Floats(inputPath));ComputeTensorData.Pin(input);
                using var worker=new Worker(prefix,BackendType.GPUCompute);
                var cpu=(HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallback",BindingFlags.NonPublic|BindingFlags.Instance).GetValue(worker);
                var flush=(HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallbackShouldFlushGPU",BindingFlags.NonPublic|BindingFlags.Instance).GetValue(worker);
                var count=cpu.Intersect(prefix.layers.SelectMany(layer=>layer.outputs).Concat(prefix.inputs.Select(value=>value.index))).Count();
                Assert.That(count,Is.Zero);Assert.That(flush.Count,Is.Zero);report["cpu_computation_or_input_count"]=count;report["readback_flush_count"]=flush.Count;
                report["phase"]="execute";Save();worker.Schedule(input);var tensor=(Tensor<float>)worker.PeekOutput();
                Assert.That(tensor.backendType,Is.EqualTo(BackendType.GPUCompute));var actual=tensor.DownloadToArray();var expected=Floats(expectedPath);Assert.That(actual.Length,Is.EqualTo(expected.Length));
                var bits=actual.Where((value,i)=>BitConverter.SingleToInt32Bits(value)!=BitConverter.SingleToInt32Bits(expected[i])).Count();
                report["actual_backend"]="GPUCompute";report["bit_differences_against_observed_gpu"]=bits;report["phase"]="completed";Save();
                Assert.That(bits,Is.Zero,"The isolated RMS must first preserve the unchanged observed GPU calculation.");report["success"]=true;
            }
            catch(Exception error){report["error"]=error.ToString();throw;}
            finally{Save();}
        }
        static float[] Floats(string path){var bytes=File.ReadAllBytes(path);var data=new float[bytes.Length/4];Buffer.BlockCopy(bytes,0,data,0,bytes.Length);return data;}
    }
}
