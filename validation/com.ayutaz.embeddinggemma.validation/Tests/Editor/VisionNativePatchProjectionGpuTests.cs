using System;
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
    public sealed class VisionNativePatchProjectionGpuTests
    {
        [Test]
        public void NativeBlock192VisionPatchProjectionMustMatchOriginalPatchBitsOnGpu()
            =>Assert.That(Run(192),Is.Zero,"Keep the unchanged native patch bits.");

        [Test]
        public void VisionPatchAccumulationBlockSweepFindsAnExactNativeOrder()
        {
            var exact=0;
            foreach(var block in new[]{128,192,256,384,512,768})if(Run(block)==0)exact++;
            Assert.That(exact,Is.GreaterThan(0),"Record every failed candidate; no block may redefine the original golden.");
        }

        int Run(int block)
        {
            Assert.That(SystemInfo.supportsComputeShaders,Is.True,"Actual GPU required; never skip.");
            var source=Environment.GetEnvironmentVariable("EG_ACCEPTANCE_SOURCE");
            var root=Path.GetFullPath(Path.Combine(Application.dataPath,"../../k/artifacts/vision-diagnostics"));
            var output=Path.GetFullPath(Path.Combine(Application.dataPath,"../artifacts/vision-native-patch-projection-gpu-"+source+"-block"+block+".json"));
            Assert.That(File.Exists(output),Is.False);
            var report=new JObject{["success"]=false,["scope"]="one_original_vision_patch_prefix_native_gpu_projection_candidate",["runtime_source"]=source,
                ["full_encoder_validated"]=false,["end_to_end_validated"]=false,["phase"]="audit",["accumulation_block_size"]=block};
            void Save(){Directory.CreateDirectory(Path.GetDirectoryName(output));File.WriteAllText(output,report.ToString());}
            void Check(string path,string sha){using var stream=File.OpenRead(path);Assert.That(PlayerFileHash.Compute(stream,out _),Is.EqualTo(sha));}
            Save();
            try
            {
                var locked=JObject.Parse(File.ReadAllText(Path.Combine(Application.dataPath,"../Packages/packages-lock.json")));
                foreach(var package in new[]{"com.ayutaz.embeddinggemma","com.ayutaz.embeddinggemma.validation"})Assert.That(locked["dependencies"][package].Value<string>("hash"),Is.EqualTo(source));
                Check(Path.Combine(root,"vision-diagnostics.json"),"998b7d0d200b257b0bc19de32f45ef27be4328088d2d51706ea956cc3b4461ce");
                var reference=JObject.Parse(File.ReadAllText(Path.Combine(root,"vision-diagnostics.json")));
                var prefixPath=Path.GetFullPath(Path.Combine(Application.dataPath,"../artifacts/vision-patch-prefix-gpu-28aeb460008ed5b1013b5534df3642217b1cea0a-captures/patch-prefix.sentis"));
                Check(prefixPath,"02b81eeb7339fade9fb5682272998ee284be46bcd67b77104fe12df7ba28b1c5");
                var prefix=TextModelFile.Load(prefixPath);var constants=prefix.constants.ToDictionary(value=>value.index);
                var projection=prefix.layers.Single(layer=>layer.GetType().Name=="MatMul"&&constants.TryGetValue(layer.inputs[1],out var w)&&w.shape==new TensorShape(768,768));
                var raw=(ArraySegment<byte>)typeof(Constant).GetField("array",BindingFlags.NonPublic|BindingFlags.Instance).GetValue(constants[projection.inputs[1]]);
                var coefficients=new float[raw.Count/4];Buffer.BlockCopy(raw.Array,raw.Offset,coefficients,0,raw.Count);
                using(var stream=new MemoryStream(raw.Array,raw.Offset,raw.Count,false))Assert.That(PlayerFileHash.Compute(stream,out _),Is.EqualTo("696c2e091ed4ad117c10faa6bf9707eb808649fb09c84fee457d2327408acc90"));
                var before=ModelTensorSlice.Extract(prefix,projection.inputs[0],"projection_input");
                var after=ModelTensorSlice.Extract(prefix,prefix.outputs[0].index,"patch_output",new[]{new Model.Input{
                    name="projection_output",index=projection.outputs[0],dataType=DataType.Float,shape=new DynamicTensorShape(2520,768)}});
                Assert.That(before.inputs.Count,Is.EqualTo(1));Assert.That(after.inputs.Count,Is.EqualTo(1));
                using var operation=(IGpuTensorOperation)Activator.CreateInstance(typeof(GpuBlockedMatmul),768,768,coefficients,block);
                using var pipeline=new GpuProjectionPipeline(new[]{before,after},new[]{"projection_input"},new[]{"projection_output"},new IGpuTensorOperation[]{operation});
                report["cpu_computation_or_input_count"]=0;report["readback_flush_count"]=0;report["prefix_layers_before"]=before.layers.Count;report["prefix_layers_after"]=after.layers.Count;
                var pixelsPath=Path.Combine(root,"pixels-fp32.bin");var expectedPath=Path.Combine(root,"native-stage-0.bin");
                foreach(var file in new[]{pixelsPath,expectedPath})Check(file,reference["files"][Path.GetFileName(file)].Value<string>("sha256"));
                using var input=new Tensor<float>(new TensorShape(1,2520,768),Floats(pixelsPath));ComputeTensorData.Pin(input);
                report["phase"]="execute";Save();using var encoded=pipeline.Execute(input);Assert.That(encoded.backendType,Is.EqualTo(BackendType.GPUCompute));
                var actual=encoded.DownloadToArray();var expected=Floats(expectedPath);Assert.That(actual.Length,Is.EqualTo(expected.Length));
                var capture=Path.ChangeExtension(output,"bin");var bytes=new byte[actual.Length*4];Buffer.BlockCopy(actual,0,bytes,0,bytes.Length);File.WriteAllBytes(capture,bytes);
                using(var stream=File.OpenRead(capture))report["capture_sha256"]=PlayerFileHash.Compute(stream,out _);
                int bits=0,bad=0;double maximum=0,ratio=0;
                for(var i=0;i<actual.Length;i++)
                {
                    Assert.That(float.IsNaN(actual[i])||float.IsInfinity(actual[i]),Is.False);
                    if(BitConverter.SingleToInt32Bits(actual[i])!=BitConverter.SingleToInt32Bits(expected[i]))bits++;
                    var d=Math.Abs((double)actual[i]-expected[i]);var allowed=1e-5+1e-4*Math.Abs((double)expected[i]);if(d>allowed)bad++;ratio=Math.Max(ratio,d/allowed);maximum=Math.Max(maximum,d);
                }
                report["actual_backend"]="GPUCompute";report["bit_differences"]=bits;report["mismatched_elements"]=bad;report["maximum_absolute_error"]=maximum;report["maximum_tolerance_ratio"]=ratio;report["phase"]="completed";Save();
                report["success"]=bits==0;return bits;
            }
            catch(Exception error){report["error"]=error.ToString();throw;}
            finally{Save();}
        }
        static float[] Floats(string path){var bytes=File.ReadAllBytes(path);var data=new float[bytes.Length/4];Buffer.BlockCopy(bytes,0,data,0,bytes.Length);return data;}
    }
}
