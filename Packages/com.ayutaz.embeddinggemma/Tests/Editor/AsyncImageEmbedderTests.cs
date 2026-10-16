using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.TestTools;

namespace EmbeddingGemma.Tests
{
    // Controlled pending stages test facade lifetimes. Real models have separate gates.
    public sealed class AsyncImageEmbedderTests
    {
        static Task<float[]> Call(ImageEmbedder api,Texture texture,TextRole role=TextRole.Raw,string title=null,CancellationToken token=default)
        {
            var method=typeof(ImageEmbedder).GetMethod("EmbedAsync",new[]{typeof(Texture),typeof(TextRole),typeof(string),typeof(CancellationToken)});
            Assert.That(method,Is.Not.Null,"The final image API must await staged GPU encoding.");
            return (Task<float[]>)method.Invoke(api,new object[]{texture,role,title,token});
        }
        static IEnumerator Wait(Task pending)
        {
            while(!pending.IsCompleted){UnityEditor.EditorApplication.QueuePlayerLoopUpdate();yield return null;}
        }
        static MultimodalPromptBuilder Builder(MultimodalBackboneProfile profile)
        {
            var fixture=typeof(MultimodalPromptBuilderTests);const BindingFlags flags=BindingFlags.Static|BindingFlags.NonPublic;
            return new MultimodalPromptBuilder((string)fixture.GetField("Json",flags).GetRawConstantValue(),
                (string)fixture.GetField("Configuration",flags).GetRawConstantValue(),profile);
        }
        sealed class Pre : IImagePreprocessor,IDisposable
        {
            public BackendType Backend=>BackendType.GPUCompute;
            public ImagePreprocessingResult Last;public int Calls;public bool Disposed;
            public ImagePreprocessingResult Process(Texture input)
            {
                if(Disposed)throw new ObjectDisposedException("fixture");Calls++;
                var pixels=new Tensor<float>(new TensorShape(1,2520,768));ComputeTensorData.Pin(pixels);
                var positions=Enumerable.Repeat(-1,5040).ToArray();for(int i=0;i<2304;i++){positions[2*i]=i%48;positions[2*i+1]=i/48;}
                return Last=(ImagePreprocessingResult)Activator.CreateInstance(typeof(ImagePreprocessingResult),BindingFlags.Instance|BindingFlags.NonPublic,null,
                    new object[]{pixels,768,768,256,positions,new List<Tensor>{pixels},new List<ComputeBuffer>()},null);
            }
            public void Dispose(){Disposed=true;}
        }
        sealed class Enc : IAsyncImageFeatureEncoder,IDisposable
        {
            public BackendType Backend=>BackendType.GPUCompute;
            public ImageEncoderProfile Profile{get;}=new ImageEncoderProfile(768,768);
            public TaskCompletionSource<ImageEncodingResult> Pending;public ImageEncodingResult Last;public ImagePreprocessingResult Held;
            public bool Disposed;public int SyncCalls;
            public ImageEncodingResult Encode(ImagePreprocessingResult input){SyncCalls++;throw new InvalidOperationException("The async facade must await the staged encoder.");}
            public Task<ImageEncodingResult> EncodeAsync(ImagePreprocessingResult input,CancellationToken token=default)
            {
                if(Disposed)throw new ObjectDisposedException("fixture");Held=input;Pending=new TaskCompletionSource<ImageEncodingResult>();
                var captured=Pending;token.Register(()=>captured.TrySetCanceled());return captured.Task;
            }
            public void Release()
            {
                Assert.That(Held.Pixels.dataOnBackend,Is.Not.Null,"Preprocessing must live until encoding completes.");
                var tensor=new Tensor<float>(new TensorShape(256,512));ComputeTensorData.Pin(tensor);
                Last=(ImageEncodingResult)Activator.CreateInstance(typeof(ImageEncodingResult),BindingFlags.Instance|BindingFlags.NonPublic,null,new object[]{tensor,768,768},null);
                Assert.That(Pending.TrySetResult(Last),Is.True);
            }
            public void Dispose(){Disposed=true;}
        }
        sealed class Back : IMultimodalBackbone,IDisposable
        {
            public BackendType Backend=>BackendType.GPUCompute;
            public MultimodalBackboneProfile Profile{get;}=new MultimodalBackboneProfile(280,10,11,256,0,32,0);
            public int Calls;public int[] Ids,Mask;public float[] Returned;public string Error;public bool Disposed;
            public float[] Embed(int[] ids,int[] mask,Tensor<float> image,Tensor<float> audio)
            {
                if(Disposed)throw new ObjectDisposedException("fixture");Calls++;Ids=ids;Mask=mask;
                Assert.That(image.dataOnBackend,Is.Not.Null);Assert.That(image.backendType,Is.EqualTo(BackendType.GPUCompute));Assert.That(image.shape,Is.EqualTo(new TensorShape(256,512)));
                Assert.That(audio.shape,Is.EqualTo(new TensorShape(0,512)));
                if(Error=="throw")throw new InvalidOperationException("controlled backbone failure");if(Error=="null")return null;
                Returned=new float[Error=="dimension"?767:768];Returned[0]=Error=="nonfinite"?float.NaN:Error=="norm"?0f:1f;return Returned;
            }
            public void Dispose(){Disposed=true;}
        }
        sealed class Parts : IDisposable
        {
            public readonly Pre Pre=new Pre();public readonly Enc Enc=new Enc();public readonly Back Back=new Back();
            public readonly Texture2D Texture=new Texture2D(2,2,TextureFormat.RGBA32,false,true);
            public readonly MultimodalPromptBuilder Prompts;
            public Parts(){Assert.That(SystemInfo.supportsComputeShaders,Is.True,"Actual GPU required; never skip.");Prompts=Builder(Back.Profile);}
            public ImageEmbedder Create()=>new ImageEmbedder(Pre,Enc,Back,Prompts);
            public void AssertReleased()
            {
                Assert.Throws<ObjectDisposedException>(()=>{var ignored=Pre.Last.Pixels;});
                if(Enc.Last!=null)Assert.Throws<ObjectDisposedException>(()=>{var ignored=Enc.Last.SoftTokens;});
            }
            public void Dispose(){Pre.Dispose();Enc.Dispose();Back.Dispose();UnityEngine.Object.DestroyImmediate(Texture);}
        }

        [UnityTest]
        public IEnumerator AllRolesAwaitGpuTokensAndReturnIndependentNormalizedVectors()
        {
            using var p=new Parts();using var api=p.Create();
            foreach(var role in new[]{TextRole.Query,TextRole.Document,TextRole.Raw})
            {
                var pending=Call(api,p.Texture,role);Assert.That(pending.IsCompleted,Is.False);Assert.That(p.Pre.Last.Pixels.dataOnBackend,Is.Not.Null);
                p.Enc.Release();yield return Wait(pending);var result=pending.GetAwaiter().GetResult();
                var expected=p.Prompts.BuildImage(256,role);Assert.That(p.Back.Ids,Is.EqualTo(expected.InputIds));Assert.That(p.Back.Mask,Is.EqualTo(expected.AttentionMask));
                Assert.That(result.Length,Is.EqualTo(768));Assert.That(result[0],Is.EqualTo(1));Assert.That(result,Is.Not.SameAs(p.Back.Returned));p.AssertReleased();
                result[0]=42;Assert.That(p.Back.Returned[0],Is.EqualTo(1));Assert.That(p.Texture==null,Is.False);Assert.That(p.Enc.SyncCalls,Is.Zero);
            }
        }
        [Test]
        public void InvalidInputRoleTitleAndPreCancellationRejectBeforeGpuWork()
        {
            using var p=new Parts();using var api=p.Create();
            Assert.Throws<ArgumentNullException>(()=>Call(api,null).GetAwaiter().GetResult());
            Assert.Throws<ArgumentOutOfRangeException>(()=>Call(api,p.Texture,(TextRole)99).GetAwaiter().GetResult());
            Assert.Throws<ArgumentException>(()=>Call(api,p.Texture,TextRole.Document,"<image>").GetAwaiter().GetResult());
            using var canceled=new CancellationTokenSource();canceled.Cancel();var pending=Call(api,p.Texture,token:canceled.Token);Assert.That(pending.IsCanceled,Is.True);
            Assert.That(p.Pre.Calls,Is.Zero);
        }
        [UnityTest]
        public IEnumerator CancellationDuringEncoderReleasesInputsAndAllowsAnotherCall()
        {
            using var p=new Parts();using var api=p.Create();using var cancellation=new CancellationTokenSource();
            var pending=Call(api,p.Texture,token:cancellation.Token);cancellation.Cancel();yield return Wait(pending);
            Assert.That(pending.IsCanceled,Is.True);p.AssertReleased();Assert.That(p.Back.Calls,Is.Zero);
            var next=Call(api,p.Texture);p.Enc.Release();yield return Wait(next);Assert.That(next.GetAwaiter().GetResult().Length,Is.EqualTo(768));p.AssertReleased();
        }
        [UnityTest]
        public IEnumerator DisposeDuringEncoderCancelsWithoutDisposingBorrowedStages()
        {
            using var p=new Parts();var api=p.Create();var pending=Call(api,p.Texture);api.Dispose();api.Dispose();yield return Wait(pending);
            Assert.That(pending.IsCanceled,Is.True);p.AssertReleased();Assert.That(p.Pre.Disposed||p.Enc.Disposed||p.Back.Disposed,Is.False);Assert.That(p.Texture==null,Is.False);
            Assert.Throws<ObjectDisposedException>(()=>Call(api,p.Texture).GetAwaiter().GetResult());
            using var processed=p.Pre.Process(p.Texture);var borrowed=p.Enc.EncodeAsync(processed);p.Enc.Release();using var result=borrowed.GetAwaiter().GetResult();Assert.That(result.SoftTokens.dataOnBackend,Is.Not.Null);
        }
        [UnityTest]
        public IEnumerator ConcurrentAsyncAndSynchronousCallsAreRejectedWithoutPoisoningPendingCall()
        {
            using var p=new Parts();using var api=p.Create();var pending=Call(api,p.Texture);
            Assert.Throws<InvalidOperationException>(()=>Call(api,p.Texture).GetAwaiter().GetResult());Assert.Throws<InvalidOperationException>(()=>api.Embed(p.Texture));
            p.Enc.Release();yield return Wait(pending);Assert.That(pending.GetAwaiter().GetResult().Length,Is.EqualTo(768));p.AssertReleased();
        }
        [UnityTest]
        public IEnumerator EncoderFailureReleasesOwnedInputAndAllowsReuse()
        {
            using var p=new Parts();using var api=p.Create();var pending=Call(api,p.Texture);p.Enc.Pending.SetException(new InvalidOperationException("controlled encoder failure"));yield return Wait(pending);
            Assert.Throws<InvalidOperationException>(()=>pending.GetAwaiter().GetResult());p.AssertReleased();Assert.That(p.Back.Calls,Is.Zero);
            var next=Call(api,p.Texture);p.Enc.Release();yield return Wait(next);Assert.That(next.GetAwaiter().GetResult().Length,Is.EqualTo(768));
        }
        [UnityTest]
        public IEnumerator InvalidOrFailedFinalOutputReleasesAllOwnedResultsAndAllowsReuse()
        {
            using var p=new Parts();using var api=p.Create();
            foreach(var error in new[]{"throw","null","dimension","norm","nonfinite"})
            {
                p.Back.Error=error;var pending=Call(api,p.Texture);p.Enc.Release();yield return Wait(pending);Assert.Throws<InvalidOperationException>(()=>pending.GetAwaiter().GetResult());p.AssertReleased();
            }
            p.Back.Error=null;var next=Call(api,p.Texture);p.Enc.Release();yield return Wait(next);Assert.That(next.GetAwaiter().GetResult().Length,Is.EqualTo(768));
        }
        sealed class SyncEncoder : IImageFeatureEncoder
        {
            readonly Enc inner;public SyncEncoder(Enc inner){this.inner=inner;}public BackendType Backend=>inner.Backend;public ImageEncoderProfile Profile=>inner.Profile;
            public ImageEncodingResult Encode(ImagePreprocessingResult input)=>inner.Encode(input);
        }
        [Test]
        public void MissingStagedEncoderIsRejectedBeforePreprocessing()
        {
            using var p=new Parts();using var api=new ImageEmbedder(p.Pre,new SyncEncoder(p.Enc),p.Back,p.Prompts);
            Assert.Throws<NotSupportedException>(()=>Call(api,p.Texture).GetAwaiter().GetResult());Assert.That(p.Pre.Calls,Is.Zero);
        }
    }
}
