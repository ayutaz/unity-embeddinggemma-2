using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Runtime.ExceptionServices;
using System.Threading;
using System.Threading.Tasks;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.TestTools;

namespace EmbeddingGemma.Tests
{
    // Small adapter contracts; original image numerical acceptance is separate.
    public sealed class PartitionedImageFeatureEncoderTests
    {
        static Type Required()
        {
            var type=typeof(ImageFeatureEncoder).Assembly.GetType("EmbeddingGemma.PartitionedImageFeatureEncoder");
            Assert.That(type,Is.Not.Null,"The qualified GPU segment path must implement the image feature encoder contract.");return type;
        }
        static IImageFeatureEncoder Create(IGpuTensorOperation operation,ImageEncoderProfile profile,int outputCount=0)
        {
            var first=new FunctionalGraph();var pixels=first.AddInput<float>(new TensorShape(1,2520,768),"pixels");
            var before=Rename(first.Compile(Functional.Unsqueeze(Functional.ReduceMean(pixels,new[]{0,1},false),0)),"reduced");
            var last=new FunctionalGraph();var projected=last.AddInput<float>(new TensorShape(1,512),"projected");
            var after=Rename(last.Compile(projected*Functional.Constant(new TensorShape(outputCount==0?profile.FeatureCount:outputCount,512),Enumerable.Repeat(1f,(outputCount==0?profile.FeatureCount:outputCount)*512).ToArray())),"tokens");
            try{return (IImageFeatureEncoder)Activator.CreateInstance(Required(),new[]{before,after},new[]{new[]{"reduced"}},new[]{"projected"},new[]{operation},profile);}
            catch(TargetInvocationException error){ExceptionDispatchInfo.Capture(error.InnerException).Throw();throw;}
        }
        static Model Rename(Model model,string name){var output=model.outputs[0];output.name=name;model.outputs[0]=output;return model;}
        static ImagePreprocessingResult Input(ImageEncoderProfile profile,float value=.25f,BackendType backend=BackendType.GPUCompute,bool wrongPosition=false)
        {
            var pixels=new Tensor<float>(new TensorShape(1,2520,768),Enumerable.Repeat(value,2520*768).ToArray());
            if(backend==BackendType.GPUCompute)ComputeTensorData.Pin(pixels);
            var positions=Enumerable.Repeat(-1,5040).ToArray();var columns=profile.ResizedWidth/16;var count=columns*(profile.ResizedHeight/16);
            for(var i=0;i<count;i++){positions[i*2]=i%columns;positions[i*2+1]=i/columns;}if(wrongPosition)positions[0]=1;
            return (ImagePreprocessingResult)Activator.CreateInstance(typeof(ImagePreprocessingResult),BindingFlags.Instance|BindingFlags.NonPublic,null,
                new object[]{pixels,profile.ResizedWidth,profile.ResizedHeight,profile.FeatureCount,positions,new List<Tensor>{pixels},new List<ComputeBuffer>()},null);
        }
        static GpuBlockedMatmul Operation()=>new GpuBlockedMatmul(768,512,Enumerable.Repeat(1f,768*512).ToArray());

        [TestCase(768,768)] [TestCase(1008,576)] [TestCase(576,1008)]
        public void AllImageProfilesRetainIndependentGpuResultsAndBorrowedInputsAndOperations(int width,int height)
        {
            Assert.That(SystemInfo.supportsComputeShaders,Is.True,"Actual GPU required; never skip.");
            var profile=new ImageEncoderProfile(width,height);using var operation=Operation();var encoder=Create(operation,profile);using var owner=(IDisposable)encoder;
            using var input=Input(profile);using var changed=Input(profile,.5f);
            using var first=encoder.Encode(input);using var second=encoder.Encode(changed);owner.Dispose();owner.Dispose();
            Assert.That(encoder.Backend,Is.EqualTo(BackendType.GPUCompute));Assert.That(encoder.Profile,Is.SameAs(profile));
            Assert.That(first.SoftTokens.shape,Is.EqualTo(new TensorShape(profile.FeatureCount,512)));Assert.That(first.SoftTokens.backendType,Is.EqualTo(BackendType.GPUCompute));
            Assert.That(first.SoftTokens.DownloadToArray(),Is.All.EqualTo(192f));Assert.That(second.SoftTokens.DownloadToArray(),Is.All.EqualTo(384f));
            Assert.That(input.Pixels.dataOnBackend,Is.Not.Null);Assert.Throws<ObjectDisposedException>(()=>encoder.Encode(input));
            using var probe=new Tensor<float>(new TensorShape(1,768),Enumerable.Repeat(1f,768).ToArray());ComputeTensorData.Pin(probe);
            using var reused=operation.Execute(probe);Assert.That(reused.DownloadToArray(),Is.All.EqualTo(768f));
        }
        [Test]
        public void WrongImageGeometryCoordinatesDeviceAndDisposedInputDoNotPoisonReuse()
        {
            var profile=new ImageEncoderProfile(768,768);using var operation=Operation();var encoder=Create(operation,profile);using var owner=(IDisposable)encoder;
            using var valid=Input(profile);using var wrongGeometry=Input(new ImageEncoderProfile(1008,576));using var wrongPosition=Input(profile,wrongPosition:true);using var cpu=Input(profile,backend:BackendType.CPU);
            var disposed=Input(profile);disposed.Dispose();
            Assert.Throws<ArgumentNullException>(()=>encoder.Encode(null));Assert.Throws<ArgumentException>(()=>encoder.Encode(wrongGeometry));
            Assert.Throws<ArgumentException>(()=>encoder.Encode(wrongPosition));Assert.Throws<ArgumentException>(()=>encoder.Encode(cpu));Assert.Throws<ObjectDisposedException>(()=>encoder.Encode(disposed));
            using var result=encoder.Encode(valid);Assert.That(result.SoftTokens.DownloadToArray(),Is.All.EqualTo(192f));
        }
        [Test]
        public void WrongProjectedImageOutputIsRejectedAndCanBeRetried()
        {
            var profile=new ImageEncoderProfile(768,768);using var operation=Operation();var encoder=Create(operation,profile,255);using var owner=(IDisposable)encoder;using var input=Input(profile);
            Assert.Throws<InvalidOperationException>(()=>encoder.Encode(input));Assert.Throws<InvalidOperationException>(()=>encoder.Encode(input));Assert.That(input.Pixels.dataOnBackend,Is.Not.Null);
        }
        [UnityTest]
        public IEnumerator StagedImageEncodingCancelsBeforeWorkAndRemainsReusable()
        {
            var profile=new ImageEncoderProfile(768,768);using var operation=Operation();var encoder=Create(operation,profile);using var owner=(IDisposable)encoder;using var input=Input(profile);
            var method=Required().GetMethod("EncodeAsync",new[]{typeof(ImagePreprocessingResult),typeof(CancellationToken)});Assert.That(method,Is.Not.Null,"The large image graph needs staged GPU execution.");
            using var cancellation=new CancellationTokenSource();cancellation.Cancel();
            Task<ImageEncodingResult> Call(CancellationToken token)=>(Task<ImageEncodingResult>)method.Invoke(encoder,new object[]{input,token});
            var canceled=Call(cancellation.Token);while(!canceled.IsCompleted)yield return null;Assert.That(canceled.IsCanceled,Is.True);
            var pending=Call(default);while(!pending.IsCompleted){UnityEditor.EditorApplication.QueuePlayerLoopUpdate();yield return null;}
            using var result=pending.GetAwaiter().GetResult();owner.Dispose();Assert.That(result.SoftTokens.DownloadToArray(),Is.All.EqualTo(192f));Assert.That(input.Pixels.dataOnBackend,Is.Not.Null);
        }
    }
}
