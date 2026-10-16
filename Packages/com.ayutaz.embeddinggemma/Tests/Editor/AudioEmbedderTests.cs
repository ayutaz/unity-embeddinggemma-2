using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEngine;

namespace EmbeddingGemma.Tests
{
    // Independent small graphs prove composition, cleanup and the real GPU handoff only.
    public sealed class AudioEmbedderTests
    {
        static AudioEmbedder Create(Parts parts, string omit = null) => new AudioEmbedder(
            omit == "preprocessor" ? null : parts.Pre, omit == "encoder" ? null : parts.Enc,
            omit == "backbone" ? null : parts.Back, omit == "prompts" ? null : parts.Prompts);
        static float[] Embed(AudioEmbedder api, AudioPcmInput input) => api.Embed(input);
        static MultimodalPrompt Prompt(MultimodalPromptBuilder builder, int frames) => builder.BuildAudio(frames);
        static MultimodalPromptBuilder Builder(MultimodalBackboneProfile profile)
        {
            var fixture = typeof(AudioEmbedderTests).Assembly.GetType("EmbeddingGemma.Tests.MultimodalPromptBuilderTests");
            const BindingFlags flags = BindingFlags.Static | BindingFlags.NonPublic;
            return new MultimodalPromptBuilder((string)fixture.GetField("Json", flags).GetRawConstantValue(),
                (string)fixture.GetField("Configuration", flags).GetRawConstantValue(), profile);
        }
        static AudioPcmInput Pcm(float value = 0.25f, int samples = 321) => new AudioPcmInput(Enumerable.Repeat(value, samples).ToArray());
        static Model PreModel()
        {
            var graph = new FunctionalGraph();
            var pcm = graph.AddInput<float>(new TensorShape(1, 179200), "waveform");
            var mask = graph.AddInput<int>(new TensorShape(1, 179200), "sample_mask");
            var seed = Functional.ReduceSum(pcm, new[] {0,1}, true) / 179200f;
            return graph.Compile(seed * Functional.Constant(new TensorShape(1,1119,128), Enumerable.Repeat(1f,1119*128).ToArray()),
                Functional.IndexSelect(mask,1,Functional.Constant(new TensorShape(1119), Enumerable.Range(0,1119).Select(i => 160+i*160).ToArray())));
        }
        static Model EncModel()
        {
            var graph = new FunctionalGraph(); var input = graph.AddInput<float>(new TensorShape(1,1119,128), "input");
            return graph.Compile(Functional.ReduceSum(input,new[] {0,1,2},false) / (1119*128f)
                * Functional.Constant(new TensorShape(1,512),Enumerable.Repeat(1f,512).ToArray()));
        }
        static Model BackModel()
        {
            var graph = new FunctionalGraph(); var ids = graph.AddInput<int>(new TensorShape(1,16),"input_ids");
            var mask = graph.AddInput<int>(new TensorShape(1,16),"attention_mask");
            var image = graph.AddInput<float>(new TensorShape(0,512),"image_features");
            var audio = graph.AddInput<float>(new TensorShape(1,512),"audio_features");
            var seed = Functional.ReduceSum((ids+mask).Float(),new[] {0,1},true)
                + Functional.ReduceSum(audio,new[] {0,1},true)/512f + Functional.ReduceSum(image,new[] {0,1},true);
            var first = new float[768]; first[0]=1; var second = new float[768]; second[1]=1;
            var vector = seed*Functional.Constant(new TensorShape(1,768),first)+Functional.Constant(new TensorShape(1,768),second);
            return graph.Compile(vector/Functional.Sqrt(Functional.ReduceSumSquare(vector,1,true)));
        }
        sealed class TrackingPre : IAudioPreprocessor
        {
            public AudioPreprocessor Inner; public AudioPreprocessingResult Last; public int Calls;
            public BackendType Backend {get;set;}
            public AudioPreprocessingResult Process(AudioPcmInput input) {Calls++; return Last=Inner.Process(input);}
        }
        sealed class TrackingEnc : IAudioFeatureEncoder
        {
            public AudioFeatureEncoder Inner; public AudioEncodingResult Last; public bool Fail;
            public BackendType Backend {get;set;} public AudioEncoderProfile Profile => Inner.Profile;
            public AudioEncodingResult Encode(AudioPreprocessingResult input)
            { if(Fail) throw new InvalidOperationException("encoder failure fixture"); return Last=Inner.Encode(input); }
        }
        sealed class TrackingBack : IMultimodalBackbone
        {
            public MultimodalBackbone Inner; public Tensor<float> LastAudio, LastImage; public bool Fail;
            public BackendType Backend {get;set;} public MultimodalBackboneProfile Profile {get;set;}
            public float[] Embed(int[] ids,int[] mask,Tensor<float> image,Tensor<float> audio)
            {LastAudio=audio; LastImage=image; if(Fail) throw new InvalidOperationException("backbone failure fixture"); return Inner.Embed(ids,mask,image,audio);}
        }
        sealed class Parts : IDisposable
        {
            public readonly TrackingPre Pre; public readonly TrackingEnc Enc; public readonly TrackingBack Back;
            public readonly MultimodalPromptBuilder Prompts; public readonly Model BackboneModel;
            public Parts(BackendType backend = BackendType.CPU)
            {
                var profile = new MultimodalBackboneProfile(16,10,11,0,1,32,0);
                Pre=new TrackingPre {Inner=new AudioPreprocessor(PreModel(),backend),Backend=backend};
                Enc=new TrackingEnc {Inner=new AudioFeatureEncoder(EncModel(),backend,new AudioEncoderProfile(2)),Backend=backend};
                BackboneModel=BackModel(); Back=new TrackingBack {Inner=new MultimodalBackbone(BackboneModel,profile,backend),Backend=backend,Profile=profile};
                Prompts=Builder(profile);
            }
            public void Dispose() {Pre.Inner.Dispose(); Enc.Inner.Dispose(); Back.Inner.Dispose();}
        }
        [TestCase(1,1)] [TestCase(4,1)] [TestCase(5,2)] [TestCase(839,210)] [TestCase(1119,280)]
        public void AudioPromptUsesCpuPrefixMetadataAndOfficialMarkers(int frames,int features)
        {
            var prompt=Prompt(Builder(new MultimodalBackboneProfile(features+8,10,11,0,features,32,0)),frames);
            Assert.That(prompt.InputIds.Take(features+4),Is.EqualTo(new[] {1,14}.Concat(Enumerable.Repeat(11,features)).Concat(new[] {15,2})));
        }
        [TestCase(0)] [TestCase(1120)]
        public void InvalidPrefixIsRejectedBeforeAudioPrompt(int frames) =>
            Assert.Throws<ArgumentOutOfRangeException>(() => Prompt(Builder(new MultimodalBackboneProfile(16,10,11,0,1,32,0)),frames));
        [TestCase("preprocessor")] [TestCase("encoder")] [TestCase("backbone")] [TestCase("prompts")]
        public void NullStagesAreRejected(string omit)
        {using var parts=new Parts(); Assert.Throws<ArgumentNullException>(() => Create(parts,omit));}
        [Test]
        public void MixedBackendsAreRejected()
        {using var parts=new Parts(); parts.Enc.Backend=BackendType.GPUCompute; Assert.Throws<ArgumentException>(()=>Create(parts));}
        [Test]
        public void MismatchedPromptAndBackboneProfileIsRejected()
        {using var parts=new Parts(); parts.Back.Profile=new MultimodalBackboneProfile(15,10,11,0,1,32,0); Assert.Throws<ArgumentException>(()=>Create(parts));}
        [Test]
        public void ImageOrWrongAudioProfileIsRejected()
        {using var parts=new Parts(); parts.Back.Profile=new MultimodalBackboneProfile(16,10,11,1,1,32,0); Assert.Throws<ArgumentException>(()=>Create(parts));}
        [TestCase(BackendType.CPU)] [TestCase(BackendType.GPUCompute)]
        public void PipelineReturnsOwnedNormalized768VectorAndReleasesIntermediateCopies(BackendType backend)
        {
            if(backend==BackendType.GPUCompute) Assert.That(SystemInfo.supportsComputeShaders,Is.True,"Actual GPU required; never skip.");
            using var parts=new Parts(backend); using var api=Create(parts);
            var result=Embed(api,Pcm()); Assert.That(result.Length,Is.EqualTo(768));
            var seed=48+321*0.25/179200; var norm=Math.Sqrt(seed*seed+1);
            Assert.That(result[0],Is.EqualTo(seed/norm).Within(1e-6)); Assert.That(result[1],Is.EqualTo(1/norm).Within(1e-6));
            Assert.That(result.Skip(2),Is.All.EqualTo(0)); Assert.That(result.Sum(x=>(double)x*x),Is.EqualTo(1).Within(1e-6));
            Assert.Throws<ObjectDisposedException>(()=>{var ignored=parts.Pre.Last.Features;});
            Assert.Throws<ObjectDisposedException>(()=>{var ignored=parts.Enc.Last.SoftTokens;});
            Assert.That(parts.Back.LastAudio.dataOnBackend,Is.Null);
            var repeated=Embed(api,Pcm(0.5f)); Assert.That(repeated,Is.Not.SameAs(result));
            Assert.That(Math.Abs(repeated[1]-result[1]),Is.GreaterThan(1e-8));
            if(backend==BackendType.GPUCompute)
            {
                var worker=(Worker)typeof(MultimodalBackbone).GetField("worker",BindingFlags.Instance|BindingFlags.NonPublic).GetValue(parts.Back.Inner);
                var cpu=(HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallback",BindingFlags.Instance|BindingFlags.NonPublic).GetValue(worker);
                Assert.That(cpu.Intersect(parts.BackboneModel.layers.SelectMany(layer=>layer.outputs)),Is.Empty);
            }
        }
        [TestCase("encoder")] [TestCase("backbone")]
        public void StageFailureReleasesOwnedResultsAndAllowsRecovery(string stage)
        {
            using var parts=new Parts(); using var api=Create(parts);
            parts.Enc.Fail=stage=="encoder"; parts.Back.Fail=stage=="backbone";
            Assert.Throws<InvalidOperationException>(()=>Embed(api,Pcm()));
            Assert.Throws<ObjectDisposedException>(()=>{var ignored=parts.Pre.Last.Features;});
            if(parts.Enc.Last!=null) Assert.Throws<ObjectDisposedException>(()=>{var ignored=parts.Enc.Last.SoftTokens;});
            parts.Enc.Fail=false; parts.Back.Fail=false; Assert.That(Embed(api,Pcm()).Length,Is.EqualTo(768));
        }
        [Test]
        public void InvalidInputIsRejectedBeforePreprocessing()
        {
            using var parts=new Parts(); using var api=Create(parts);
            Assert.Throws<ArgumentNullException>(()=>Embed(api,null)); Assert.Throws<ArgumentException>(()=>Embed(api,Pcm(samples:481)));
            Assert.That(parts.Pre.Calls,Is.Zero);
        }
        [Test]
        public void FacadeDisposalDoesNotDisposeBorrowedWorkers()
        {
            using var parts=new Parts(); var api=Create(parts); api.Dispose(); api.Dispose();
            Assert.Throws<ObjectDisposedException>(()=>Embed(api,Pcm()));
            using var processed=parts.Pre.Inner.Process(Pcm()); using var encoded=parts.Enc.Inner.Encode(processed);
            Assert.That(encoded.SoftTokens.shape,Is.EqualTo(new TensorShape(1,512)));
        }
    }
}
