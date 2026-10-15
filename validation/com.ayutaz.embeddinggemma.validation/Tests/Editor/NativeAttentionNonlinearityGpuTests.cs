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
    public sealed class NativeAttentionNonlinearityGpuTests
    {
        [TestCase("tanh")]
        [TestCase("softmax")]
        public void FrozenNativeAttentionNonlinearityMustMatchAllBitsOnGpu(string name)
            =>Execute(name,false);

        [Test]
        public void FrozenNativeTanhCandidateMustMatchAllBitsOnGpu()
            =>Execute("tanh",true);

        [Test]
        public void FrozenNativeSoftmaxCandidateMustMatchAllBitsOnGpu()
            =>Execute("softmax",true);

        static Type RequiredSoftmax()
        {
            var type=typeof(GpuSilu).Assembly.GetType("EmbeddingGemma.GpuSoftmax");Assert.That(type,Is.Not.Null,"Native-order GPU softmax is missing.");return type;
        }

        [Test]
        public void NativeSoftmaxBorrowsInputAndReusesAfterRejectedCpuInput()
        {
            Assert.That(SystemInfo.supportsComputeShaders,Is.True,"Actual GPU required; no skip.");
            var type=RequiredSoftmax();using var operation=(IDisposable)Activator.CreateInstance(type);var execute=type.GetMethod("Execute");var values=new float[48];
            using var cpu=new Tensor<float>(new TensorShape(2,24),values);
            var error=Assert.Throws<TargetInvocationException>(()=>execute.Invoke(operation,new object[]{cpu}));Assert.That(error.InnerException,Is.InstanceOf<ArgumentException>());
            using var input=new Tensor<float>(new TensorShape(2,24),values);ComputeTensorData.Pin(input);
            using var result=(Tensor<float>)execute.Invoke(operation,new object[]{input});using var other=(Tensor<float>)execute.Invoke(operation,new object[]{input});
            operation.Dispose();operation.Dispose();Assert.That(result.backendType,Is.EqualTo(BackendType.GPUCompute));Assert.That(result.shape,Is.EqualTo(input.shape));
            Assert.That(result.DownloadToArray(),Is.EqualTo(Enumerable.Repeat(1f/24f,48).ToArray()));Assert.That(other.DownloadToArray(),Is.EqualTo(result.DownloadToArray()));
            Assert.That(input.DownloadToArray(),Is.EqualTo(values));error=Assert.Throws<TargetInvocationException>(()=>execute.Invoke(operation,new object[]{input}));Assert.That(error.InnerException,Is.InstanceOf<ObjectDisposedException>());
        }

        static Type RequiredTanh()
        {
            var type=typeof(GpuSilu).Assembly.GetType("EmbeddingGemma.GpuTanh");Assert.That(type,Is.Not.Null,"Native-rounding GPU tanh is missing.");return type;
        }

        [Test]
        public void NativeTanhPreservesSignedZeroAndBorrowedGpuResourcesAfterRejectedInput()
        {
            Assert.That(SystemInfo.supportsComputeShaders,Is.True,"Actual GPU required; no skip.");
            var type=RequiredTanh();using var operation=(IDisposable)Activator.CreateInstance(type);var execute=type.GetMethod("Execute");
            var values=new[]{BitConverter.Int32BitsToSingle(unchecked((int)0x80000000)),0f};
            using var cpu=new Tensor<float>(new TensorShape(2),values);
            var error=Assert.Throws<TargetInvocationException>(()=>execute.Invoke(operation,new object[]{cpu}));Assert.That(error.InnerException,Is.InstanceOf<ArgumentException>());
            using var input=new Tensor<float>(new TensorShape(2),values);ComputeTensorData.Pin(input);
            using var output=(Tensor<float>)execute.Invoke(operation,new object[]{input});using var other=(Tensor<float>)execute.Invoke(operation,new object[]{input});
            operation.Dispose();operation.Dispose();Assert.That(output.backendType,Is.EqualTo(BackendType.GPUCompute));Assert.That(output.shape,Is.EqualTo(input.shape));
            Assert.That(output.DownloadToArray().Select(BitConverter.SingleToInt32Bits),Is.EqualTo(values.Select(BitConverter.SingleToInt32Bits)));
            Assert.That(other.DownloadToArray().Select(BitConverter.SingleToInt32Bits),Is.EqualTo(values.Select(BitConverter.SingleToInt32Bits)));
            Assert.That(input.DownloadToArray().Select(BitConverter.SingleToInt32Bits),Is.EqualTo(values.Select(BitConverter.SingleToInt32Bits)));
            error=Assert.Throws<TargetInvocationException>(()=>execute.Invoke(operation,new object[]{input}));Assert.That(error.InnerException,Is.InstanceOf<ObjectDisposedException>());
        }

        void Execute(string name,bool nativeCandidate)
        {
            Assert.That(SystemInfo.supportsComputeShaders,Is.True,"Actual GPU required; no skip.");
            var source=Environment.GetEnvironmentVariable("EG_ACCEPTANCE_SOURCE");
            var root=Path.GetFullPath(Path.Combine(Application.dataPath,"../../numerical-acceptance-state/native-attention-operations-5a51cff-20261015"));
            var output=Path.GetFullPath(Path.Combine(Application.dataPath,"../artifacts/"+(nativeCandidate?"frozen-native-"+name+"-candidate-gpu-":"frozen-native-attention-nonlinearity-gpu-")+source+"-"+name+".json"));Assert.That(File.Exists(output),Is.False);
            var report=new JObject{["success"]=false,["scope"]="one_frozen_native_attention_nonlinearity_gpu_bit_gate",["runtime_source"]=source,["operation"]=name,["full_encoder_validated"]=false,["end_to_end_validated"]=false};
            void Save(){Directory.CreateDirectory(Path.GetDirectoryName(output));File.WriteAllText(output,report.ToString());}
            void Check(string path,string sha){using var stream=File.OpenRead(path);Assert.That(PlayerFileHash.Compute(stream,out _),Is.EqualTo(sha));}Save();
            try
            {
                var locked=JObject.Parse(File.ReadAllText(Path.Combine(Application.dataPath,"../Packages/packages-lock.json")));
                foreach(var package in new[]{"com.ayutaz.embeddinggemma","com.ayutaz.embeddinggemma.validation"})Assert.That(locked["dependencies"][package].Value<string>("hash"),Is.EqualTo(source));
                var referenceFile=Path.Combine(root,"audio-attention-replay-report.json");Check(referenceFile,"d16f82c1c343769577106a39203da32f00fb797468cc656ab1b12b685b8e691e");
                var reference=JObject.Parse(File.ReadAllText(referenceFile));Assert.That(reference.Value<bool>("success")&&reference.Value<bool>("native_final_exact_match"),Is.True);
                Assert.That(reference.Value<string>("reference_source"),Is.EqualTo("4edbe17a8105233c5aaa1514faa2862b08a4f7e0"));
                float[] Read(string kind,out int[] shape)
                {
                    var part=reference["parts"]["self_attn."+name+"."+kind];var path=Path.Combine(root,"audio-attention-replay",part.Value<string>("file"));Check(path,part.Value<string>("sha256"));
                    shape=part["shape"].ToObject<int[]>();var bytes=File.ReadAllBytes(path);var result=new float[bytes.Length/4];Buffer.BlockCopy(bytes,0,result,0,bytes.Length);return result;
                }
                var values=Read("input",out var shape);var expected=Read("output",out var outputShape);Assert.That(outputShape,Is.EqualTo(shape));
                using var input=new Tensor<float>(new TensorShape(shape),values);ComputeTensorData.Pin(input);
                using var result=Run(input,name,nativeCandidate);
                Assert.That(result.backendType,Is.EqualTo(BackendType.GPUCompute));var actual=result.DownloadToArray();Assert.That(actual.Length,Is.EqualTo(expected.Length));int bits=0,bad=0;double maxRatio=0;
                for(var i=0;i<actual.Length;i++)
                {
                    Assert.That(float.IsNaN(actual[i])||float.IsInfinity(actual[i]),Is.False);
                    if(BitConverter.SingleToInt32Bits(actual[i])!=BitConverter.SingleToInt32Bits(expected[i]))bits++;
                    var delta=Math.Abs((double)actual[i]-expected[i]);var allowed=1e-5+1e-4*Math.Abs((double)expected[i]);if(delta>allowed)bad++;maxRatio=Math.Max(maxRatio,delta/allowed);
                }
                report["actual_backend"]=result.backendType.ToString();report["count"]=actual.Length;report["bit_differences"]=bits;report["mismatched_elements"]=bad;report["maximum_tolerance_ratio"]=maxRatio;Save();
                Assert.That(bits,Is.Zero,"Keep original native nonlinearity bits.");report["success"]=true;
            }
            catch(Exception error){report["error"]=error.ToString();throw;}
            finally{Save();}
        }

        static Tensor<float> Run(Tensor<float> input,string name,bool nativeCandidate)
        {
            if(nativeCandidate)
            {
                using var operation=(IGpuTensorOperation)Activator.CreateInstance(name=="tanh"?RequiredTanh():RequiredSoftmax());return operation.Execute(input);
            }
            var graph=new FunctionalGraph();var x=graph.AddInput<float>(input.shape,"input");var model=graph.Compile(name=="tanh"?Functional.Tanh(x):Functional.Softmax(x,-1));
            using var worker=new Worker(model,BackendType.GPUCompute);
            var cpu=(HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallback",BindingFlags.NonPublic|BindingFlags.Instance).GetValue(worker);
            var flush=(HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallbackShouldFlushGPU",BindingFlags.NonPublic|BindingFlags.Instance).GetValue(worker);
            Assert.That(cpu.Intersect(model.layers.SelectMany(l=>l.outputs).Concat(model.inputs.Select(i=>i.index))).Count(),Is.Zero);Assert.That(flush.Count,Is.Zero);
            worker.Schedule(input);Tensor output=null;worker.CopyOutput(0,ref output);return (Tensor<float>)output;
        }
    }
}
