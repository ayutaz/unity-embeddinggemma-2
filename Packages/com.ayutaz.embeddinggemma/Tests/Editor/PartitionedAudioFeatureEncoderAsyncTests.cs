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
    public sealed class PartitionedAudioFeatureEncoderAsyncTests
    {
        const BindingFlags Hidden=BindingFlags.NonPublic|BindingFlags.Static;
        static AudioPreprocessingResult Input(int frames=2)=>(AudioPreprocessingResult)typeof(PartitionedAudioFeatureEncoderTests).GetMethod("Input",Hidden).Invoke(null,new object[]{0.25f,frames,BackendType.GPUCompute});
        static PartitionedAudioFeatureEncoder Encoder(IGpuTensorOperation operation)
        {
            var models=(Model[])typeof(PartitionedAudioFeatureEncoderTests).GetMethod("Models",Hidden).Invoke(null,new object[]{512});
            return new PartitionedAudioFeatureEncoder(models,new[]{new[]{"reduced"}},new[]{"projected"},new[]{operation},new AudioEncoderProfile(2));
        }
        static Task<AudioEncodingResult> Call(PartitionedAudioFeatureEncoder encoder,AudioPreprocessingResult input,CancellationToken token=default)
        {
            var method=typeof(PartitionedAudioFeatureEncoder).GetMethod("EncodeAsync",new[]{typeof(AudioPreprocessingResult),typeof(CancellationToken)});
            Assert.That(method,Is.Not.Null,"Audio encoder must retire GPU-complete caches between segments.");return (Task<AudioEncodingResult>)method.Invoke(encoder,new object[]{input,token});
        }
        static IEnumerator Wait(Task task){while(!task.IsCompleted){UnityEditor.EditorApplication.QueuePlayerLoopUpdate();yield return null;}}
        [UnityTest] public IEnumerator StagedExecutionKeepsBitsInputsAndIndependentOwnedOutputs()
        {
            using var operation=new GpuBlockedMatmul(128,512,Enumerable.Repeat(1f,128*512).ToArray());using var encoder=Encoder(operation);using var input=Input();
            using var baseline=encoder.Encode(input);var pending=Call(encoder,input);Assert.That(pending.IsCompleted,Is.False);yield return Wait(pending);using var encoded=pending.GetAwaiter().GetResult();
            var repeat=Call(encoder,input);yield return Wait(repeat);using var again=repeat.GetAwaiter().GetResult();encoder.Dispose();
            Assert.That(encoded.SoftTokens.backendType,Is.EqualTo(BackendType.GPUCompute));Assert.That(encoded.SoftTokens.DownloadToArray(),Is.EqualTo(baseline.SoftTokens.DownloadToArray()));
            Assert.That(again.SoftTokens.DownloadToArray(),Is.All.EqualTo(32f));Assert.That(input.Features.dataOnBackend,Is.Not.Null);
        }
        [UnityTest] public IEnumerator CancellationAtGpuFencePreservesBorrowedInputAndAllowsReuse()
        {
            using var operation=new GpuBlockedMatmul(128,512,Enumerable.Repeat(1f,128*512).ToArray());using var encoder=Encoder(operation);using var input=Input();using var cancellation=new CancellationTokenSource();
            var pending=Call(encoder,input,cancellation.Token);Assert.That(pending.IsCompleted,Is.False);cancellation.Cancel();yield return Wait(pending);Assert.That(pending.IsCanceled,Is.True);
            var next=Call(encoder,input);yield return Wait(next);using var result=next.GetAwaiter().GetResult();Assert.That(result.SoftTokens.DownloadToArray(),Is.All.EqualTo(32f));Assert.That(input.FrameMask.dataOnBackend,Is.Not.Null);
        }
        [UnityTest] public IEnumerator DisposePendingAndOverlapRejectWithoutTakingBorrowedOwnership()
        {
            using var operation=new GpuBlockedMatmul(128,512,Enumerable.Repeat(1f,128*512).ToArray());using var encoder=Encoder(operation);using var input=Input();
            var pending=Call(encoder,input);Assert.Throws<InvalidOperationException>(()=>encoder.Encode(input));encoder.Dispose();yield return Wait(pending);
            Assert.Throws<ObjectDisposedException>(()=>pending.GetAwaiter().GetResult());Assert.Throws<ObjectDisposedException>(()=>Call(encoder,input).GetAwaiter().GetResult());Assert.That(input.Features.dataOnBackend,Is.Not.Null);
        }
        [UnityTest] public IEnumerator InvalidPrefixAndPreCancellationDoNotPoisonReuse()
        {
            using var operation=new GpuBlockedMatmul(128,512,Enumerable.Repeat(1f,128*512).ToArray());using var encoder=Encoder(operation);using var wrong=Input(3);using var input=Input();
            Assert.Throws<ArgumentException>(()=>Call(encoder,wrong).GetAwaiter().GetResult());using var canceled=new CancellationTokenSource();canceled.Cancel();Assert.That(Call(encoder,input,canceled.Token).IsCanceled,Is.True);
            var pending=Call(encoder,input);yield return Wait(pending);using var result=pending.GetAwaiter().GetResult();Assert.That(result.SoftTokens.shape,Is.EqualTo(new TensorShape(1,512)));
        }
    }
}
