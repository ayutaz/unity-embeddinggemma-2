using System;
using System.IO;
using EmbeddingGemma.Tests;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEngine;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class NativeAttentionBmmGpuTests
    {
        [TestCase("matrix_ac")]
        [TestCase("matrix_bd")]
        [TestCase("weighted_values")]
        public void FrozenNativeAttentionProductMustMatchAllBitsOnGpu(string name)
        {
            Assert.That(SystemInfo.supportsComputeShaders,Is.True,"Actual GPU required; no skip.");
            var source=Environment.GetEnvironmentVariable("EG_ACCEPTANCE_SOURCE");
            var root=Path.GetFullPath(Path.Combine(Application.dataPath,"../../numerical-acceptance-state/native-attention-operations-5a51cff-20261015"));
            var output=Path.GetFullPath(Path.Combine(Application.dataPath,"../artifacts/frozen-native-attention-bmm-gpu-"+source+"-"+name+".json"));Assert.That(File.Exists(output),Is.False);
            var report=new JObject{["success"]=false,["scope"]="one_frozen_native_attention_bmm_gpu_bit_gate",["runtime_source"]=source,["operation"]=name,["full_encoder_validated"]=false,["end_to_end_validated"]=false};
            void Save(){Directory.CreateDirectory(Path.GetDirectoryName(output));File.WriteAllText(output,report.ToString());}
            void Check(string path,string sha){using var stream=File.OpenRead(path);Assert.That(PlayerFileHash.Compute(stream,out _),Is.EqualTo(sha));}
            Save();
            try
            {
                var locked=JObject.Parse(File.ReadAllText(Path.Combine(Application.dataPath,"../Packages/packages-lock.json")));
                foreach(var package in new[]{"com.ayutaz.embeddinggemma","com.ayutaz.embeddinggemma.validation"})Assert.That(locked["dependencies"][package].Value<string>("hash"),Is.EqualTo(source));
                var path=Path.Combine(root,"audio-attention-replay-report.json");Check(path,"d16f82c1c343769577106a39203da32f00fb797468cc656ab1b12b685b8e691e");
                var reference=JObject.Parse(File.ReadAllText(path));Assert.That(reference.Value<bool>("success")&&reference.Value<bool>("native_final_exact_match"),Is.True);
                Assert.That(reference.Value<string>("reference_source"),Is.EqualTo("4edbe17a8105233c5aaa1514faa2862b08a4f7e0"));
                float[] Read(string kind,out int[] shape)
                {
                    var part=reference["parts"]["self_attn."+name+"."+kind];var file=Path.Combine(root,"audio-attention-replay",part.Value<string>("file"));Check(file,part.Value<string>("sha256"));
                    shape=part["shape"].ToObject<int[]>();var bytes=File.ReadAllBytes(file);var values=new float[bytes.Length/4];Buffer.BlockCopy(bytes,0,values,0,bytes.Length);return values;
                }
                var a=Read("left",out var shapeA);var b=Read("right",out var shapeB);var expected=Read("output",out var shapeOutput);
                var type=typeof(GpuSilu).Assembly.GetType("EmbeddingGemma.GpuBatchedMatmul");Assert.That(type,Is.Not.Null,"Native-order GPU batched matmul is missing.");
                using var operation=(IDisposable)Activator.CreateInstance(type);using var left=new Tensor<float>(new TensorShape(shapeA),a);using var right=new Tensor<float>(new TensorShape(shapeB),b);
                ComputeTensorData.Pin(left);ComputeTensorData.Pin(right);
                using var result=(Tensor<float>)type.GetMethod("Execute",new[]{typeof(Tensor<float>),typeof(Tensor<float>)}).Invoke(operation,new object[]{left,right});
                Assert.That(result.backendType,Is.EqualTo(BackendType.GPUCompute));Assert.That(result.shape,Is.EqualTo(new TensorShape(shapeOutput)));var actual=result.DownloadToArray();int bits=0;
                for(var index=0;index<actual.Length;index++)if(BitConverter.SingleToInt32Bits(actual[index])!=BitConverter.SingleToInt32Bits(expected[index]))bits++;
                report["actual_backend"]=result.backendType.ToString();report["count"]=actual.Length;report["bit_differences"]=bits;Save();
                Assert.That(bits,Is.Zero,"Keep the original native product bits.");report["success"]=true;
            }
            catch(Exception error){report["error"]=error.ToString();throw;}
            finally{Save();}
        }
    }
}
