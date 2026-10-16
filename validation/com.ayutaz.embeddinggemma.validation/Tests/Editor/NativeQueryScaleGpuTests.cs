using System;
using System.IO;
using EmbeddingGemma.Tests;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEngine;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class NativeQueryScaleGpuTests
    {
        static Type Required()
        {
            var type=typeof(GpuSilu).Assembly.GetType("EmbeddingGemma.GpuSequentialScale");
            Assert.That(type,Is.Not.Null,"Sequential FP32 GPU scale is missing.");return type;
        }
        static float[] Floats(string path){var bytes=File.ReadAllBytes(path);var result=new float[bytes.Length/4];Buffer.BlockCopy(bytes,0,result,0,bytes.Length);return result;}
        static void Check(string path,string sha){using var stream=File.OpenRead(path);Assert.That(PlayerFileHash.Compute(stream,out _),Is.EqualTo(sha));}

        [Test]
        public void SequentialScaleBorrowsGpuInputCopiesCoefficientsAndSurvivesRejectedInput()
        {
            Assert.That(SystemInfo.supportsComputeShaders,Is.True,"Actual GPU required; no skip.");
            var type=Required();var vector=new[]{1f,2f,3f};var operation=(IDisposable)Activator.CreateInstance(type,2f,vector);vector[0]=99f;
            var execute=type.GetMethod("Execute");using var cpu=new Tensor<float>(new TensorShape(2,3),new[]{1f,2f,3f,4f,5f,6f});
            var error=Assert.Throws<System.Reflection.TargetInvocationException>(()=>execute.Invoke(operation,new object[]{cpu}));Assert.That(error.InnerException,Is.InstanceOf<ArgumentException>());
            using var wrong=new Tensor<float>(new TensorShape(1,2),new[]{1f,2f});ComputeTensorData.Pin(wrong);
            error=Assert.Throws<System.Reflection.TargetInvocationException>(()=>execute.Invoke(operation,new object[]{wrong}));Assert.That(error.InnerException,Is.InstanceOf<ArgumentException>());
            using var input=new Tensor<float>(new TensorShape(2,3),new[]{1f,2f,3f,4f,5f,6f});ComputeTensorData.Pin(input);
            using var result=(Tensor<float>)execute.Invoke(operation,new object[]{input});using var other=(Tensor<float>)execute.Invoke(operation,new object[]{input});
            operation.Dispose();operation.Dispose();Assert.That(result.backendType,Is.EqualTo(BackendType.GPUCompute));Assert.That(result.shape,Is.EqualTo(input.shape));
            Assert.That(result.DownloadToArray(),Is.EqualTo(new[]{2f,8f,18f,8f,20f,36f}));Assert.That(other.DownloadToArray(),Is.EqualTo(result.DownloadToArray()));
            Assert.That(input.DownloadToArray(),Is.EqualTo(new[]{1f,2f,3f,4f,5f,6f}));
            error=Assert.Throws<System.Reflection.TargetInvocationException>(()=>execute.Invoke(operation,new object[]{input}));Assert.That(error.InnerException,Is.InstanceOf<ObjectDisposedException>());
        }

        [Test]
        public void FrozenNativeQueryScalingMustMatchAllBitsOnGpu()
        {
            Assert.That(SystemInfo.supportsComputeShaders,Is.True,"Actual GPU required; no skip.");
            var source=Environment.GetEnvironmentVariable("EG_ACCEPTANCE_SOURCE");
            var root=Path.GetFullPath(Path.Combine(Application.dataPath,"../../numerical-acceptance-state"));
            var output=Path.GetFullPath(Path.Combine(Application.dataPath,"../artifacts/frozen-native-query-scale-gpu-"+source+".json"));Assert.That(File.Exists(output),Is.False);
            var report=new JObject{["success"]=false,["scope"]="one_frozen_native_query_scale_gpu_bit_gate",["runtime_source"]=source,["full_encoder_validated"]=false,["end_to_end_validated"]=false};
            void Save(){Directory.CreateDirectory(Path.GetDirectoryName(output));File.WriteAllText(output,report.ToString());}Save();
            try
            {
                var locked=JObject.Parse(File.ReadAllText(Path.Combine(Application.dataPath,"../Packages/packages-lock.json")));
                foreach(var package in new[]{"com.ayutaz.embeddinggemma","com.ayutaz.embeddinggemma.validation"})Assert.That(locked["dependencies"][package].Value<string>("hash"),Is.EqualTo(source));
                var nativeRoot=Path.Combine(root,"native-attention-operations-5a51cff-20261015");var referenceFile=Path.Combine(nativeRoot,"audio-attention-replay-report.json");
                Check(referenceFile,"d16f82c1c343769577106a39203da32f00fb797468cc656ab1b12b685b8e691e");var reference=JObject.Parse(File.ReadAllText(referenceFile));
                Assert.That(reference.Value<bool>("success")&&reference.Value<bool>("native_final_exact_match"),Is.True);
                Assert.That(reference.Value<string>("reference_source"),Is.EqualTo("4edbe17a8105233c5aaa1514faa2862b08a4f7e0"));
                float[] Read(string name){var part=reference["parts"][name];var path=Path.Combine(nativeRoot,"audio-attention-replay",part.Value<string>("file"));Check(path,part.Value<string>("sha256"));return Floats(path);}
                var query=Read("self_attn.q_proj");var blocked=Read("self_attn.matrix_ac.left");var vectorFile=Path.Combine(root,"query-scale-fixed-reference/scale-fp32.bin");
                Check(vectorFile,"082757fba75572192e470011329c38d52f58d5524f41eacbb5aa9ebc9e3582e0");var type=Required();
                using var operation=(IDisposable)Activator.CreateInstance(type,0.127517431974411f,Floats(vectorFile));
                using var input=new Tensor<float>(new TensorShape(1,280,8,128),query);ComputeTensorData.Pin(input);
                using var result=(Tensor<float>)type.GetMethod("Execute").Invoke(operation,new object[]{input});Assert.That(result.backendType,Is.EqualTo(BackendType.GPUCompute));
                var actual=result.DownloadToArray();Assert.That(actual.Length,Is.EqualTo(280*8*128));int differences=0;
                for(var frame=0;frame<280;frame++)for(var head=0;head<8;head++)for(var dim=0;dim<128;dim++)
                {
                    var expected=blocked[((head*24+frame/12)*12+frame%12)*128+dim];
                    if(BitConverter.SingleToInt32Bits(actual[(frame*8+head)*128+dim])!=BitConverter.SingleToInt32Bits(expected))differences++;
                }
                report["actual_backend"]=result.backendType.ToString();report["count"]=actual.Length;report["bit_differences"]=differences;Save();
                Assert.That(differences,Is.Zero,"Keep original sequential FP32 query scaling bits.");report["success"]=true;
            }
            catch(Exception error){report["error"]=error.ToString();throw;}
            finally{Save();}
        }
    }
}
