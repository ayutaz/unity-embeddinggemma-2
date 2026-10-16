using System;
using System.Linq;
using System.Reflection;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEngine;

namespace EmbeddingGemma.Tests
{
    // Small GPU contracts only; real audio numerics require separate fixed-reference gates.
    public sealed class PartitionedAudioFeatureEncoderTests
    {
        static Type Required()
        {
            var type=typeof(AudioFeatureEncoder).Assembly.GetType("EmbeddingGemma.PartitionedAudioFeatureEncoder");
            Assert.That(type,Is.Not.Null,"The accepted GPU segment path cannot yet implement IAudioFeatureEncoder.");
            return type;
        }
        static IAudioFeatureEncoder Create(Model[] models,IGpuTensorOperation[] operations,AudioEncoderProfile profile)
            =>(IAudioFeatureEncoder)Activator.CreateInstance(Required(),models,new[]{new[]{"reduced"}},new[]{"projected"},operations,profile);
        static Model Rename(Model model,string name)
        {var output=model.outputs[0];output.name=name;model.outputs[0]=output;return model;}
        static Model[] Models(int width=512)
        {
            var first=new FunctionalGraph();var input=first.AddInput<float>(new TensorShape(1,1119,128),"input");
            var prefix=Rename(first.Compile(Functional.ReduceMean(input,new[]{1},false)),"reduced");
            var last=new FunctionalGraph();var projected=last.AddInput<float>(new TensorShape(1,width),"projected");
            return new[]{prefix,Rename(last.Compile(projected),"tokens")};
        }
        static AudioPreprocessingResult Input(float value=0.25f,int frames=2,BackendType backend=BackendType.GPUCompute)
        {
            var features=new Tensor<float>(new TensorShape(1,1119,128),Enumerable.Repeat(value,1119*128).ToArray());
            var mask=new Tensor<int>(new TensorShape(1,1119),Enumerable.Range(0,1119).Select(i=>i<frames?1:0).ToArray());
            if(backend==BackendType.GPUCompute){ComputeTensorData.Pin(features);ComputeTensorData.Pin(mask);}
            return (AudioPreprocessingResult)Activator.CreateInstance(typeof(AudioPreprocessingResult),BindingFlags.Instance|BindingFlags.NonPublic,null,new object[]{features,mask,frames},null);
        }

        [Test]
        public void GpuAdapterOwnsWorkersAndResultsWhileBorrowingInputsAndOperations()
        {
            Assert.That(SystemInfo.supportsComputeShaders,Is.True,"Actual GPU required; never skip.");
            using var operation=new GpuBlockedMatmul(128,512,Enumerable.Repeat(1f,128*512).ToArray());
            var encoder=Create(Models(),new IGpuTensorOperation[]{operation},new AudioEncoderProfile(2));
            using var owner=(IDisposable)encoder;
            Assert.That(encoder.Backend,Is.EqualTo(BackendType.GPUCompute));
            using var input=Input();using var changed=Input(0.5f);
            using var first=encoder.Encode(input);using var second=encoder.Encode(changed);
            owner.Dispose();owner.Dispose();
            Assert.That(first.SoftTokens.backendType,Is.EqualTo(BackendType.GPUCompute));
            Assert.That(first.SoftTokens.shape,Is.EqualTo(new TensorShape(1,512)));
            Assert.That(first.SoftTokens.DownloadToArray(),Is.All.EqualTo(32f));
            Assert.That(second.SoftTokens.DownloadToArray(),Is.All.EqualTo(64f));
            Assert.That(input.Features.dataOnBackend,Is.Not.Null);
            Assert.Throws<ObjectDisposedException>(()=>encoder.Encode(input));
            using var probe=new Tensor<float>(new TensorShape(1,128),Enumerable.Repeat(1f,128).ToArray());ComputeTensorData.Pin(probe);
            using var reused=operation.Execute(probe);Assert.That(reused.DownloadToArray(),Is.All.EqualTo(128f));
        }

        [Test]
        public void WrongPrefixDeviceNullAndDisposedInputAreRejectedWithoutPoisoningReuse()
        {
            using var operation=new GpuBlockedMatmul(128,512,Enumerable.Repeat(1f,128*512).ToArray());
            var encoder=Create(Models(),new IGpuTensorOperation[]{operation},new AudioEncoderProfile(2));using var owner=(IDisposable)encoder;
            using var wrong=Input(frames:3);using var cpu=Input(backend:BackendType.CPU);using var valid=Input();
            var disposed=Input();disposed.Dispose();
            Assert.Throws<ArgumentException>(()=>encoder.Encode(wrong));
            Assert.Throws<ArgumentException>(()=>encoder.Encode(cpu));
            Assert.Throws<ArgumentNullException>(()=>encoder.Encode(null));
            Assert.Throws<ObjectDisposedException>(()=>encoder.Encode(disposed));
            using var result=encoder.Encode(valid);Assert.That(result.SoftTokens.DownloadToArray(),Is.All.EqualTo(32f));
        }

        [Test]
        public void WrongScheduledOutputShapeIsRejectedAndCanBeRetried()
        {
            using var operation=new GpuBlockedMatmul(128,511,Enumerable.Repeat(1f,128*511).ToArray());
            var encoder=Create(Models(511),new IGpuTensorOperation[]{operation},new AudioEncoderProfile(2));using var owner=(IDisposable)encoder;
            using var input=Input();
            Assert.Throws<InvalidOperationException>(()=>encoder.Encode(input));
            Assert.Throws<InvalidOperationException>(()=>encoder.Encode(input));
            Assert.That(input.Features.dataOnBackend,Is.Not.Null);
        }
    }
}
