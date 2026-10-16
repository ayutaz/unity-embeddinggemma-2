using System;
using System.Collections;
using System.Linq;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEngine.TestTools;

namespace EmbeddingGemma.Tests
{
    // Controlled pending GPU stages: ownership and control flow, not real-model numerics.
    public sealed class AsyncAudioEmbedderTests
    {
        static Task<float[]> Call(AudioEmbedder api,AudioPcmInput input,TextRole role=TextRole.Raw,string title=null,CancellationToken token=default)
        {
            var method=typeof(AudioEmbedder).GetMethod("EmbedAsync",new[]{typeof(AudioPcmInput),typeof(TextRole),typeof(string),typeof(CancellationToken)});
            Assert.That(method,Is.Not.Null,"The final audio API must await bounded staged GPU encoding.");
            return (Task<float[]>)method.Invoke(api,new object[]{input,role,title,token});
        }
        static IEnumerator Wait(Task task){while(!task.IsCompleted){UnityEditor.EditorApplication.QueuePlayerLoopUpdate();yield return null;}}
        sealed class Pre : IAudioPreprocessor,IDisposable
        {
            public BackendType Backend=>BackendType.GPUCompute;
            public AudioPreprocessingResult Last;public int Calls;public bool Disposed;
            public AudioPreprocessingResult Process(AudioPcmInput input)
            {
                if(Disposed)throw new ObjectDisposedException("fixture");Calls++;
                var features=new Tensor<float>(new TensorShape(1,1119,128));var mask=new Tensor<int>(new TensorShape(1,1119));
                ComputeTensorData.Pin(features);ComputeTensorData.Pin(mask);
                return Last=(AudioPreprocessingResult)Activator.CreateInstance(typeof(AudioPreprocessingResult),BindingFlags.Instance|BindingFlags.NonPublic,null,new object[]{features,mask,2},null);
            }
            public void Dispose(){Disposed=true;}
        }
        sealed class Enc : IAsyncAudioFeatureEncoder,IDisposable
        {
            public BackendType Backend=>BackendType.GPUCompute;
            public AudioEncoderProfile Profile{get;}=new AudioEncoderProfile(2);
            public TaskCompletionSource<AudioEncodingResult> Pending;public AudioEncodingResult Last;public AudioPreprocessingResult Held;
            public bool Disposed;public int SyncCalls;
            public AudioEncodingResult Encode(AudioPreprocessingResult input){SyncCalls++;throw new InvalidOperationException("Async must not fall back to synchronous encoding.");}
            public Task<AudioEncodingResult> EncodeAsync(AudioPreprocessingResult input,CancellationToken token=default)
            {
                if(Disposed)throw new ObjectDisposedException("fixture");Held=input;Pending=new TaskCompletionSource<AudioEncodingResult>();
                var captured=Pending;token.Register(()=>captured.TrySetCanceled());return captured.Task;
            }
            public void Release()
            {
                Assert.That(Held.Features.dataOnBackend,Is.Not.Null);
                var tensor=new Tensor<float>(new TensorShape(1,512));ComputeTensorData.Pin(tensor);
                Last=(AudioEncodingResult)Activator.CreateInstance(typeof(AudioEncodingResult),BindingFlags.Instance|BindingFlags.NonPublic,null,new object[]{tensor,2},null);
                Assert.That(Pending.TrySetResult(Last),Is.True);
            }
            public void Dispose(){Disposed=true;}
        }
        sealed class Back : IMultimodalBackbone,IDisposable
        {
            public BackendType Backend=>BackendType.GPUCompute;
            public MultimodalBackboneProfile Profile{get;}=new MultimodalBackboneProfile(24,10,11,0,1,32,0);
            public int Calls;public int[] Ids,Mask;public float[] Returned;public string Error;public bool Disposed;
            public float[] Embed(int[] ids,int[] mask,Tensor<float> image,Tensor<float> audio)
            {
                if(Disposed)throw new ObjectDisposedException("fixture");Calls++;Ids=ids;Mask=mask;
                Assert.That(audio.dataOnBackend,Is.Not.Null);Assert.That(audio.backendType,Is.EqualTo(BackendType.GPUCompute));Assert.That(audio.shape,Is.EqualTo(new TensorShape(1,512)));
                Assert.That(image.shape,Is.EqualTo(new TensorShape(0,512)));
                if(Error=="throw")throw new InvalidOperationException("controlled backbone failure");if(Error=="null")return null;
                Returned=new float[Error=="dimension"?767:768];Returned[0]=Error=="nonfinite"?float.NaN:Error=="norm"?0f:1f;return Returned;
            }
            public void Dispose(){Disposed=true;}
        }
        sealed class Parts : IDisposable
        {
            public readonly Pre Pre=new Pre();public readonly Enc Enc=new Enc();public readonly Back Back=new Back();
            public readonly AudioPcmInput Input=new AudioPcmInput(Enumerable.Repeat(0.25f,321).ToArray());
            public readonly MultimodalPromptBuilder Prompts;
            public Parts()
            {
                const BindingFlags flags=BindingFlags.Static|BindingFlags.NonPublic;var fixture=typeof(MultimodalPromptBuilderTests);
                Prompts=new MultimodalPromptBuilder((string)fixture.GetField("Json",flags).GetRawConstantValue(),(string)fixture.GetField("Configuration",flags).GetRawConstantValue(),Back.Profile);
            }
            public AudioEmbedder Create()=>new AudioEmbedder(Pre,Enc,Back,Prompts);
            public void Released(){Assert.Throws<ObjectDisposedException>(()=>{var ignored=Pre.Last.Features;});if(Enc.Last!=null)Assert.Throws<ObjectDisposedException>(()=>{var ignored=Enc.Last.SoftTokens;});}
            public void Dispose(){Pre.Dispose();Enc.Dispose();Back.Dispose();}
        }
        [UnityTest] public IEnumerator AllRolesKeepGpuInputsAliveAndReturnOwnedVectors()
        {
            using var p=new Parts();using var api=p.Create();
            foreach(var role in new[]{TextRole.Query,TextRole.Document,TextRole.Raw})
            {
                var pending=Call(api,p.Input,role);Assert.That(pending.IsCompleted,Is.False);p.Enc.Release();yield return Wait(pending);
                var vector=pending.GetAwaiter().GetResult();var prompt=p.Prompts.BuildAudio(2,role);
                Assert.That(p.Back.Ids,Is.EqualTo(prompt.InputIds));Assert.That(p.Back.Mask,Is.EqualTo(prompt.AttentionMask));
                Assert.That(vector.Length,Is.EqualTo(768));Assert.That(vector,Is.Not.SameAs(p.Back.Returned));vector[0]=42;Assert.That(p.Back.Returned[0],Is.EqualTo(1));p.Released();Assert.That(p.Enc.SyncCalls,Is.Zero);
            }
        }
        [Test] public void InvalidPcmRoleTitleAndPreCancellationRejectBeforeGpuWork()
        {
            using var p=new Parts();using var api=p.Create();
            Assert.Throws<ArgumentNullException>(()=>Call(api,null).GetAwaiter().GetResult());
            Assert.Throws<ArgumentException>(()=>Call(api,new AudioPcmInput(new float[161])).GetAwaiter().GetResult());
            Assert.Throws<ArgumentOutOfRangeException>(()=>Call(api,p.Input,(TextRole)99).GetAwaiter().GetResult());
            Assert.Throws<ArgumentException>(()=>Call(api,p.Input,TextRole.Document,"<audio>").GetAwaiter().GetResult());
            using var canceled=new CancellationTokenSource();canceled.Cancel();Assert.That(Call(api,p.Input,token:canceled.Token).IsCanceled,Is.True);Assert.That(p.Pre.Calls,Is.Zero);
        }
        [UnityTest] public IEnumerator CancellationReleasesInputsAndAllowsReuse()
        {
            using var p=new Parts();using var api=p.Create();using var cancellation=new CancellationTokenSource();
            var pending=Call(api,p.Input,token:cancellation.Token);cancellation.Cancel();yield return Wait(pending);Assert.That(pending.IsCanceled,Is.True);p.Released();Assert.That(p.Back.Calls,Is.Zero);
            var next=Call(api,p.Input);p.Enc.Release();yield return Wait(next);Assert.That(next.GetAwaiter().GetResult().Length,Is.EqualTo(768));p.Released();
        }
        [UnityTest] public IEnumerator DisposeCancelsPendingCallWithoutDisposingBorrowedStages()
        {
            using var p=new Parts();var api=p.Create();var pending=Call(api,p.Input);api.Dispose();api.Dispose();yield return Wait(pending);
            Assert.That(pending.IsCanceled,Is.True);p.Released();Assert.That(p.Pre.Disposed||p.Enc.Disposed||p.Back.Disposed,Is.False);
            Assert.Throws<ObjectDisposedException>(()=>Call(api,p.Input).GetAwaiter().GetResult());
            using var processed=p.Pre.Process(p.Input);var borrowed=p.Enc.EncodeAsync(processed);p.Enc.Release();using var encoded=borrowed.GetAwaiter().GetResult();Assert.That(encoded.SoftTokens.dataOnBackend,Is.Not.Null);
        }
        [UnityTest] public IEnumerator OverlappingAsyncAndSyncCallsCannotCorruptPendingWork()
        {
            using var p=new Parts();using var api=p.Create();var pending=Call(api,p.Input);
            Assert.Throws<InvalidOperationException>(()=>Call(api,p.Input).GetAwaiter().GetResult());Assert.Throws<InvalidOperationException>(()=>api.Embed(p.Input));
            p.Enc.Release();yield return Wait(pending);Assert.That(pending.GetAwaiter().GetResult().Length,Is.EqualTo(768));p.Released();
        }
        [UnityTest] public IEnumerator EncoderAndBackboneFailuresReleaseResultsAndAllowReuse()
        {
            using var p=new Parts();using var api=p.Create();var failed=Call(api,p.Input);p.Enc.Pending.SetException(new InvalidOperationException("encoder fixture"));yield return Wait(failed);
            Assert.Throws<InvalidOperationException>(()=>failed.GetAwaiter().GetResult());p.Released();
            foreach(var error in new[]{"throw","null","dimension","norm","nonfinite"})
            {
                p.Back.Error=error;var pending=Call(api,p.Input);p.Enc.Release();yield return Wait(pending);Assert.Throws<InvalidOperationException>(()=>pending.GetAwaiter().GetResult());p.Released();
            }
            p.Back.Error=null;var next=Call(api,p.Input);p.Enc.Release();yield return Wait(next);Assert.That(next.GetAwaiter().GetResult().Length,Is.EqualTo(768));p.Released();
        }
        sealed class SyncEncoder : IAudioFeatureEncoder
        {
            readonly Enc inner;public SyncEncoder(Enc inner){this.inner=inner;}public BackendType Backend=>inner.Backend;public AudioEncoderProfile Profile=>inner.Profile;
            public AudioEncodingResult Encode(AudioPreprocessingResult input)=>inner.Encode(input);
        }
        sealed class PendingPre : IAsyncAudioPreprocessor
        {
            readonly Pre inner; public PendingPre(Pre inner) { this.inner = inner; }
            public BackendType Backend => BackendType.GPUCompute;
            public int AsyncCalls, SyncCalls; public TaskCompletionSource<AudioPreprocessingResult> Pending;
            public AudioPreprocessingResult Process(AudioPcmInput input) { SyncCalls++; throw new InvalidOperationException("Await the preprocessing fence before encoding."); }
            public Task<AudioPreprocessingResult> ProcessAsync(AudioPcmInput input, CancellationToken token = default)
            {
                AsyncCalls++; Pending = new TaskCompletionSource<AudioPreprocessingResult>(); var captured = Pending;
                token.Register(() => captured.TrySetCanceled()); return Pending.Task;
            }
            public void Release(AudioPcmInput input) => Pending.SetResult(inner.Process(input));
        }
        [UnityTest] public IEnumerator AsyncPreprocessingIsAwaitedBeforeEncodingAndItsOwnedOutputsAreReleased()
        {
            using var p = new Parts(); var pre = new PendingPre(p.Pre); using var api = new AudioEmbedder(pre, p.Enc, p.Back, p.Prompts);
            var pending = Call(api, p.Input); Assert.That(pre.AsyncCalls, Is.EqualTo(1)); Assert.That(pre.SyncCalls, Is.Zero); Assert.That(p.Enc.Pending, Is.Null);
            pre.Release(p.Input); while (p.Enc.Pending == null) yield return null;
            p.Enc.Release(); yield return Wait(pending); Assert.That(pending.GetAwaiter().GetResult().Length, Is.EqualTo(768)); p.Released();
        }
        [UnityTest] public IEnumerator CancellationDuringPreprocessingCannotStartTheEncoder()
        {
            using var p = new Parts(); var pre = new PendingPre(p.Pre); using var api = new AudioEmbedder(pre, p.Enc, p.Back, p.Prompts);
            using var cancellation = new CancellationTokenSource(); var pending = Call(api, p.Input, token: cancellation.Token);
            Assert.That(pre.AsyncCalls, Is.EqualTo(1)); cancellation.Cancel(); yield return Wait(pending);
            Assert.That(pending.IsCanceled, Is.True); Assert.That(p.Enc.Pending, Is.Null); Assert.That(p.Pre.Calls, Is.Zero);
        }
        [UnityTest] public IEnumerator DisposingFacadeCancelsPreprocessingWithoutDisposingBorrowedStage()
        {
            using var p = new Parts(); var pre = new PendingPre(p.Pre); using var api = new AudioEmbedder(pre, p.Enc, p.Back, p.Prompts);
            var pending = Call(api, p.Input); Assert.That(pre.AsyncCalls, Is.EqualTo(1)); api.Dispose(); yield return Wait(pending);
            Assert.That(pending.IsCanceled, Is.True); Assert.That(p.Pre.Disposed, Is.False); Assert.That(p.Enc.Pending, Is.Null);
        }
        [Test] public void MissingStagedEncoderIsRejectedBeforePreprocessing()
        {
            using var p=new Parts();using var api=new AudioEmbedder(p.Pre,new SyncEncoder(p.Enc),p.Back,p.Prompts);
            Assert.Throws<NotSupportedException>(()=>Call(api,p.Input).GetAwaiter().GetResult());Assert.That(p.Pre.Calls,Is.Zero);
        }
    }
}
