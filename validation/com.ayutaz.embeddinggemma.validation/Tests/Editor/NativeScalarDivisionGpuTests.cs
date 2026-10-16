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
    public sealed class NativeScalarDivisionGpuTests
    {
        static Type Required()
        {
            var type=typeof(GpuTanh).Assembly.GetType("EmbeddingGemma.GpuScalarDivide");Assert.That(type,Is.Not.Null,"Rounded FP32 GPU division is missing.");return type;
        }
        static float[] Floats(string path){var bytes=File.ReadAllBytes(path);var result=new float[bytes.Length/4];Buffer.BlockCopy(bytes,0,result,0,bytes.Length);return result;}
        static void Check(string path,string sha){using var stream=File.OpenRead(path);Assert.That(PlayerFileHash.Compute(stream,out _),Is.EqualTo(sha));}

        [Test]
        public void ScalarDivisionBorrowsGpuInputAndReusesAfterRejectedCpuInput()
        {
            Assert.That(SystemInfo.supportsComputeShaders,Is.True,"Actual GPU required; no skip.");var type=Required();
            foreach(var divisor in new[]{0f,float.NaN,float.PositiveInfinity})
            {
                var error=Assert.Throws<TargetInvocationException>(()=>Activator.CreateInstance(type,divisor));Assert.That(error.InnerException,Is.InstanceOf<ArgumentException>());
            }
            using var operation=(IDisposable)Activator.CreateInstance(type,50f);var execute=type.GetMethod("Execute");
            var values=new[]{BitConverter.Int32BitsToSingle(unchecked((int)0x80000000)),0f,50f,-100f};
            using var cpu=new Tensor<float>(new TensorShape(4),values);
            var rejected=Assert.Throws<TargetInvocationException>(()=>execute.Invoke(operation,new object[]{cpu}));Assert.That(rejected.InnerException,Is.InstanceOf<ArgumentException>());
            using var input=new Tensor<float>(new TensorShape(4),values);ComputeTensorData.Pin(input);
            using var result=(Tensor<float>)execute.Invoke(operation,new object[]{input});using var other=(Tensor<float>)execute.Invoke(operation,new object[]{input});operation.Dispose();operation.Dispose();
            Assert.That(result.backendType,Is.EqualTo(BackendType.GPUCompute));Assert.That(result.shape,Is.EqualTo(input.shape));
            var expected=new[]{values[0],0f,1f,-2f};Assert.That(result.DownloadToArray().Select(BitConverter.SingleToInt32Bits),Is.EqualTo(expected.Select(BitConverter.SingleToInt32Bits)));
            Assert.That(other.DownloadToArray().Select(BitConverter.SingleToInt32Bits),Is.EqualTo(expected.Select(BitConverter.SingleToInt32Bits)));
            Assert.That(input.DownloadToArray().Select(BitConverter.SingleToInt32Bits),Is.EqualTo(values.Select(BitConverter.SingleToInt32Bits)));
            rejected=Assert.Throws<TargetInvocationException>(()=>execute.Invoke(operation,new object[]{input}));Assert.That(rejected.InnerException,Is.InstanceOf<ObjectDisposedException>());
        }

        [Test]
        public void FrozenNativeSoftcapDivisionMustMatchAllBitsOnGpu()
        {
            Assert.That(SystemInfo.supportsComputeShaders,Is.True,"Actual GPU required; no skip.");var source=Environment.GetEnvironmentVariable("EG_ACCEPTANCE_SOURCE");
            var root=Path.GetFullPath(Path.Combine(Application.dataPath,"../../numerical-acceptance-state"));
            var output=Path.GetFullPath(Path.Combine(Application.dataPath,"../artifacts/frozen-native-scalar-division-gpu-"+source+".json"));Assert.That(File.Exists(output),Is.False);
            var report=new JObject{["success"]=false,["scope"]="one_frozen_original_softcap_division_gpu_bit_gate",["runtime_source"]=source,["full_encoder_validated"]=false,["end_to_end_validated"]=false};
            void Save(){Directory.CreateDirectory(Path.GetDirectoryName(output));File.WriteAllText(output,report.ToString());}Save();
            try
            {
                var locked=JObject.Parse(File.ReadAllText(Path.Combine(Application.dataPath,"../Packages/packages-lock.json")));
                foreach(var package in new[]{"com.ayutaz.embeddinggemma","com.ayutaz.embeddinggemma.validation"})Assert.That(locked["dependencies"][package].Value<string>("hash"),Is.EqualTo(source));
                var referenceRoot=Path.Combine(root,"native-attention-operations-5a51cff-20261015");var referenceFile=Path.Combine(referenceRoot,"audio-attention-replay-report.json");
                Check(referenceFile,"d16f82c1c343769577106a39203da32f00fb797468cc656ab1b12b685b8e691e");var reference=JObject.Parse(File.ReadAllText(referenceFile));
                Assert.That(reference.Value<bool>("success")&&reference.Value<bool>("native_final_exact_match"),Is.True);
                Assert.That(reference.Value<string>("reference_source"),Is.EqualTo("4edbe17a8105233c5aaa1514faa2862b08a4f7e0"));
                var expectedPart=reference["parts"]["self_attn.tanh.input"];var expectedFile=Path.Combine(referenceRoot,"audio-attention-replay",expectedPart.Value<string>("file"));Check(expectedFile,expectedPart.Value<string>("sha256"));
                var inputFile=Path.Combine(root,"native-scalar-division-fixture/sum-fp32.bin");Check(inputFile,"1a02730524edd92ad8f741413ea2d4e3ded954c8f558a07b0b592d9f6318e6d5");
                using var operation=(IGpuTensorOperation)Activator.CreateInstance(Required(),50f);
                using var input=new Tensor<float>(new TensorShape(1,8,24,12,24),Floats(inputFile));ComputeTensorData.Pin(input);using var result=operation.Execute(input);
                Assert.That(result.backendType,Is.EqualTo(BackendType.GPUCompute));var actual=result.DownloadToArray();var expected=Floats(expectedFile);Assert.That(actual.Length,Is.EqualTo(expected.Length));int bits=0;
                for(var i=0;i<actual.Length;i++)if(BitConverter.SingleToInt32Bits(actual[i])!=BitConverter.SingleToInt32Bits(expected[i]))bits++;
                report["actual_backend"]=result.backendType.ToString();report["count"]=actual.Length;report["bit_differences"]=bits;Save();Assert.That(bits,Is.Zero,"Keep original rounded division bits.");report["success"]=true;
            }
            catch(Exception error){report["error"]=error.ToString();throw;}
            finally{Save();}
        }
    }
}
