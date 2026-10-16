using System;
using System.IO;
using System.Linq;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEditor;
using UnityEngine;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class BoundedRstdGpuTests
    {
        [Test]
        public void BoundedInverseStandardDeviationMustMatchIndependentRoundedOracleOnGpu()
        {
            Assert.That(SystemInfo.supportsComputeShaders, Is.True, "Actual GPU required; never skip.");
            var source=Environment.GetEnvironmentVariable("EG_ACCEPTANCE_SOURCE");
            var root=Path.Combine(Application.dataPath,"GpuBoundedRstdReference");
            var output=Path.GetFullPath(Path.Combine(Application.dataPath,"../artifacts/bounded-rstd-gpu-"+source+".json"));
            Assert.That(File.Exists(output),Is.False);
            var report=new JObject{["success"]=false,["runtime_source"]=source,["scope"]="bounded_scalar_rstd_standard_fp32_graph_only_no_model",
                ["count"]=65536,["seed"]=41015,["domain"]=new JArray(1e-6,2e6),
                ["full_encoder_validated"]=false,["end_to_end_validated"]=false};
            void Save(){Directory.CreateDirectory(Path.GetDirectoryName(output));File.WriteAllText(output,report.ToString());}
            void Check(string path,string expected){using var stream=File.OpenRead(path);Assert.That(PlayerFileHash.Compute(stream,out _),Is.EqualTo(expected));}
            Save();
            try
            {
                var packages=JObject.Parse(File.ReadAllText(Path.Combine(Application.dataPath,"../Packages/packages-lock.json")));
                foreach(var package in new[]{"com.ayutaz.embeddinggemma","com.ayutaz.embeddinggemma.validation"})
                    Assert.That(packages["dependencies"][package].Value<string>("hash"),Is.EqualTo(source));
                Check(Path.Combine(root,"rstd-reference.json"),"0e061576c0b4f377b60f88d54f7516087766e56db2647bc24ca3a3c4f41ed7b5");
                var reference=JObject.Parse(File.ReadAllText(Path.Combine(root,"rstd-reference.json")));
                foreach(var key in new[]{"input","expected","onnx"})Check(Path.Combine(root,reference["files"][key].Value<string>("file")),reference["files"][key].Value<string>("sha256"));
                var values=Floats(Path.Combine(root,"input-fp32.bin"));var expected=Floats(Path.Combine(root,"expected-fp32.bin"));
                var model=ModelLoader.Load(AssetDatabase.LoadAssetAtPath<ModelAsset>("Assets/GpuBoundedRstdReference/rstd.onnx"));
                using var pipeline=new GpuProjectionPipeline(new[]{model},Array.Empty<string>(),Array.Empty<string>(),Array.Empty<IGpuTensorOperation>());
                using var input=new Tensor<float>(new TensorShape(1,65536),values);ComputeTensorData.Pin(input);
                using var tensor=pipeline.Execute(input);Assert.That(tensor.backendType,Is.EqualTo(BackendType.GPUCompute));
                var actual=tensor.DownloadToArray();Assert.That(actual.Length,Is.EqualTo(expected.Length));
                var raw=new byte[actual.Length*4];Buffer.BlockCopy(actual,0,raw,0,raw.Length);File.WriteAllBytes(Path.ChangeExtension(output,"bin"),raw);
                using(var stream=new MemoryStream(raw,false))report["capture_sha256"]=PlayerFileHash.Compute(stream,out _);
                int bits=0;double maximum=0;var witnesses=new JArray();
                for(var i=0;i<actual.Length;i++)
                {
                    Assert.That(float.IsNaN(actual[i])||float.IsInfinity(actual[i]),Is.False);
                    if(BitConverter.SingleToInt32Bits(actual[i])!=BitConverter.SingleToInt32Bits(expected[i]))
                    {
                        bits++;if(witnesses.Count<16)witnesses.Add(new JObject{["index"]=i,["statistic"]=values[i],["actual"]=actual[i],["expected"]=expected[i]});
                    }
                    maximum=Math.Max(maximum,Math.Abs((double)actual[i]-expected[i]));
                }
                Assert.That(input.DownloadToArray(),Is.EqualTo(values));report["borrowed_input_preserved"]=true;
                report["actual_backend"]="GPUCompute";report["cpu_fallback_count"]=0;report["intermediate_readback_count"]=0;
                report["bit_differences"]=bits;report["maximum_absolute_error"]=maximum;report["witnesses"]=witnesses;report["phase"]="completed";Save();
                Assert.That(bits,Is.Zero,"Keep independent two-step FP32 sqrt / reciprocal rounding.");report["success"]=true;
            }
            catch(Exception error){report["error"]=error.ToString();throw;}
            finally{Save();}
        }
        static float[] Floats(string path){var bytes=File.ReadAllBytes(path);var values=new float[bytes.Length/4];Buffer.BlockCopy(bytes,0,values,0,bytes.Length);return values;}
    }
}
