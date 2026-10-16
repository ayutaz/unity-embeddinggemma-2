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
    public sealed class NativeGluGpuTests
    {
        static Type Required(){var type=typeof(GpuTanh).Assembly.GetType("EmbeddingGemma.GpuGlu");Assert.That(type,Is.Not.Null,"Native-order GPU GLU is missing.");return type;}
        [Test]
        public void GluBorrowsGpuInputHalvesLastAxisAndReusesAfterRejectedInput()
        {
            Assert.That(SystemInfo.supportsComputeShaders,Is.True,"Actual GPU required; no skip.");var type=Required();using var operation=(IDisposable)Activator.CreateInstance(type);var execute=type.GetMethod("Execute");
            var values=new[]{BitConverter.Int32BitsToSingle(unchecked((int)0x80000000)),2f,0f,0f};using var cpu=new Tensor<float>(new TensorShape(1,4),values);
            var error=Assert.Throws<TargetInvocationException>(()=>execute.Invoke(operation,new object[]{cpu}));Assert.That(error.InnerException,Is.InstanceOf<ArgumentException>());
            using var odd=new Tensor<float>(new TensorShape(1,3),new[]{1f,2f,3f});ComputeTensorData.Pin(odd);
            error=Assert.Throws<TargetInvocationException>(()=>execute.Invoke(operation,new object[]{odd}));Assert.That(error.InnerException,Is.InstanceOf<ArgumentException>());
            using var input=new Tensor<float>(new TensorShape(1,4),values);ComputeTensorData.Pin(input);
            using var result=(Tensor<float>)execute.Invoke(operation,new object[]{input});using var other=(Tensor<float>)execute.Invoke(operation,new object[]{input});operation.Dispose();operation.Dispose();
            var expected=new[]{values[0],1f};Assert.That(result.backendType,Is.EqualTo(BackendType.GPUCompute));Assert.That(result.shape,Is.EqualTo(new TensorShape(1,2)));
            Assert.That(result.DownloadToArray().Select(BitConverter.SingleToInt32Bits),Is.EqualTo(expected.Select(BitConverter.SingleToInt32Bits)));
            Assert.That(other.DownloadToArray().Select(BitConverter.SingleToInt32Bits),Is.EqualTo(expected.Select(BitConverter.SingleToInt32Bits)));
            Assert.That(input.DownloadToArray().Select(BitConverter.SingleToInt32Bits),Is.EqualTo(values.Select(BitConverter.SingleToInt32Bits)));
            error=Assert.Throws<TargetInvocationException>(()=>execute.Invoke(operation,new object[]{input}));Assert.That(error.InnerException,Is.InstanceOf<ObjectDisposedException>());
        }
        [Test]
        public void FrozenNativeGluMustMatchAllBitsOnGpu()
        {
            Assert.That(SystemInfo.supportsComputeShaders,Is.True,"Actual GPU required; no skip.");var source=Environment.GetEnvironmentVariable("EG_ACCEPTANCE_SOURCE");
            var root=Path.GetFullPath(Path.Combine(Application.dataPath,"../../numerical-acceptance-state/native-convolution-reference-5d5c5eb-20261015/audio-layer-replay-5d5c5eb8a58b4d84f65caea9f7cddf5b5ae0a4ab"));
            var output=Path.GetFullPath(Path.Combine(Application.dataPath,"../artifacts/frozen-native-glu-gpu-"+source+".json"));Assert.That(File.Exists(output),Is.False);
            var report=new JObject{["success"]=false,["scope"]="one_frozen_original_glu_gpu_bit_gate",["runtime_source"]=source,["full_encoder_validated"]=false,["end_to_end_validated"]=false};
            void Save(){Directory.CreateDirectory(Path.GetDirectoryName(output));File.WriteAllText(output,report.ToString());}
            void Check(string path,string sha){using var stream=File.OpenRead(path);Assert.That(PlayerFileHash.Compute(stream,out _),Is.EqualTo(sha));}Save();
            try
            {
                var locked=JObject.Parse(File.ReadAllText(Path.Combine(Application.dataPath,"../Packages/packages-lock.json")));
                foreach(var package in new[]{"com.ayutaz.embeddinggemma","com.ayutaz.embeddinggemma.validation"})Assert.That(locked["dependencies"][package].Value<string>("hash"),Is.EqualTo(source));
                var referenceFile=Path.Combine(root,"audio-attention-replay-report.json");Check(referenceFile,"906d186ec19f25e7397ddc3e826d0fa0f6ab07ecd988c1ec314cfeaab95d7c78");var reference=JObject.Parse(File.ReadAllText(referenceFile));
                Assert.That(reference.Value<bool>("success")&&reference.Value<bool>("native_final_exact_match")&&reference.Value<bool>("context_validated"),Is.True);
                Assert.That(reference.Value<string>("reference_source"),Is.EqualTo("4edbe17a8105233c5aaa1514faa2862b08a4f7e0"));
                float[] Read(string kind,out int[] shape){var part=reference["parts"]["lconv1d.glu."+kind];var path=Path.Combine(root,"audio-attention-replay",part.Value<string>("file"));Check(path,part.Value<string>("sha256"));shape=part["shape"].ToObject<int[]>();var bytes=File.ReadAllBytes(path);var values=new float[bytes.Length/4];Buffer.BlockCopy(bytes,0,values,0,bytes.Length);return values;}
                var x=Read("input",out var shape);var expected=Read("output",out var outputShape);using var operation=(IGpuTensorOperation)Activator.CreateInstance(Required());
                using var input=new Tensor<float>(new TensorShape(shape),x);ComputeTensorData.Pin(input);using var result=operation.Execute(input);
                Assert.That(result.backendType,Is.EqualTo(BackendType.GPUCompute));Assert.That(result.shape,Is.EqualTo(new TensorShape(outputShape)));var actual=result.DownloadToArray();int bits=0;
                for(var i=0;i<actual.Length;i++)if(BitConverter.SingleToInt32Bits(actual[i])!=BitConverter.SingleToInt32Bits(expected[i]))bits++;
                report["actual_backend"]=result.backendType.ToString();report["count"]=actual.Length;report["bit_differences"]=bits;Save();Assert.That(bits,Is.Zero,"Keep original native GLU bits.");report["success"]=true;
            }
            catch(Exception error){report["error"]=error.ToString();throw;}
            finally{Save();}
        }
    }
}
