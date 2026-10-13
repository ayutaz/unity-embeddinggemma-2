using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Runtime.ExceptionServices;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEngine;

namespace EmbeddingGemma.Tests
{
    // Independent graphs and tracking stages test composition; real image inference is a separate gate.
    public sealed class ImageEmbedderTests
    {
        static Type ApiType()
        {
            var type = typeof(ImageFeatureEncoder).Assembly.GetType("EmbeddingGemma.ImageEmbedder");
            Assert.That(type, Is.Not.Null, "Final image API is missing.");
            return type;
        }

        static object Invoke(Func<object> action)
        {
            try { return action(); }
            catch (TargetInvocationException error) { ExceptionDispatchInfo.Capture(error.InnerException).Throw(); throw; }
        }

        static IDisposable Create(Parts parts, string omit = null) => (IDisposable)Invoke(() => Activator.CreateInstance(ApiType(),
            omit == "preprocessor" ? null : parts.Pre, omit == "encoder" ? null : parts.Enc,
            omit == "backbone" ? null : parts.Back, omit == "prompts" ? null : parts.Prompts));

        static float[] Embed(IDisposable api, Texture texture, TextRole role = TextRole.Raw, string title = null) =>
            (float[])Invoke(() => api.GetType().GetMethod("Embed").Invoke(api, new object[] {texture, role, title}));

        static MultimodalPromptBuilder Builder(MultimodalBackboneProfile profile)
        {
            var fixture = typeof(ImageEmbedderTests).Assembly.GetType("EmbeddingGemma.Tests.MultimodalPromptBuilderTests");
            const BindingFlags flags = BindingFlags.Static | BindingFlags.NonPublic;
            return new MultimodalPromptBuilder((string)fixture.GetField("Json", flags).GetRawConstantValue(),
                (string)fixture.GetField("Configuration", flags).GetRawConstantValue(), profile);
        }

        static MultimodalPrompt Prompt(MultimodalPromptBuilder builder, int count)
        {
            var method = typeof(MultimodalPromptBuilder).GetMethod("BuildImage");
            Assert.That(method, Is.Not.Null, "Image-only prompt entry point is missing.");
            return (MultimodalPrompt)Invoke(() => method.Invoke(builder, new object[] {count, TextRole.Raw, null}));
        }

        static Model EncoderModel()
        {
            var graph = new FunctionalGraph(); var input = graph.AddInput<float>(new TensorShape(1,2520,768), "pixels");
            return graph.Compile(Functional.ReduceSum(input, new[] {0,1,2}, false) / (2520*768f) *
                Functional.Constant(new TensorShape(256,512), Enumerable.Repeat(1f,256*512).ToArray()));
        }

        static Model BackboneModel()
        {
            var graph = new FunctionalGraph(); var ids = graph.AddInput<int>(new TensorShape(1,280), "input_ids");
            var mask = graph.AddInput<int>(new TensorShape(1,280), "attention_mask");
            var image = graph.AddInput<float>(new TensorShape(256,512), "image_features");
            var audio = graph.AddInput<float>(new TensorShape(0,512), "audio_features");
            var seed = Functional.ReduceSum((ids+mask).Float(), new[] {0,1}, true) / 10000f +
                Functional.ReduceSum(image, new[] {0,1}, true) / (256*512f) + Functional.ReduceSum(audio, new[] {0,1}, true);
            var first = new float[768]; first[0] = 1; var second = new float[768]; second[1] = 1;
            var vector = seed*Functional.Constant(new TensorShape(1,768), first) + Functional.Constant(new TensorShape(1,768), second);
            return graph.Compile(vector / Functional.Sqrt(Functional.ReduceSumSquare(vector, 1, true)));
        }

        sealed class TrackingPre : IImagePreprocessor
        {
            public BackendType Backend {get; set;}
            public ImagePreprocessingResult Last;
            public int Calls;
            public float Value = 0.25f;
            public bool Fail;
            public ImagePreprocessingResult Process(Texture input)
            {
                Calls++;
                if (Fail) throw new InvalidOperationException("preprocessing failure fixture");
                var pixels = new Tensor<float>(new TensorShape(1,2520,768), Enumerable.Repeat(Value,2520*768).ToArray());
                if (Backend == BackendType.GPUCompute) ComputeTensorData.Pin(pixels);
                var positions = Enumerable.Repeat(-1,5040).ToArray();
                for (var patch = 0; patch < 48*48; patch++) {positions[2*patch]=patch%48; positions[2*patch+1]=patch/48;}
                return Last = (ImagePreprocessingResult)typeof(ImagePreprocessingResult).GetConstructors(BindingFlags.NonPublic|BindingFlags.Instance).Single().Invoke(
                    new object[] {pixels,768,768,256,positions,new List<Tensor> {pixels},new List<ComputeBuffer>()});
            }
        }

        sealed class TrackingEnc : IImageFeatureEncoder
        {
            public ImageFeatureEncoder Inner;
            public ImageEncodingResult Last;
            public bool Fail;
            public BackendType Backend {get; set;}
            public ImageEncoderProfile Profile => Inner.Profile;
            public ImageEncodingResult Encode(ImagePreprocessingResult input)
            {
                if (Fail) throw new InvalidOperationException("encoder failure fixture");
                return Last = Inner.Encode(input);
            }
        }

        sealed class TrackingBack : IMultimodalBackbone
        {
            public MultimodalBackbone Inner;
            public Tensor<float> LastImage, LastAudio;
            public float[] Returned;
            public bool Fail;
            public string Invalid;
            public BackendType Backend {get; set;}
            public MultimodalBackboneProfile Profile {get; set;}
            public float[] Embed(int[] ids, int[] mask, Tensor<float> image, Tensor<float> audio)
            {
                LastImage=image; LastAudio=audio;
                Assert.That(image.backendType, Is.EqualTo(Backend));
                Assert.That(image.shape, Is.EqualTo(new TensorShape(256,512)));
                Assert.That(audio.shape, Is.EqualTo(new TensorShape(0,512)));
                if (Fail) throw new InvalidOperationException("backbone failure fixture");
                if (Invalid == "dimension") return new float[767];
                if (Invalid == "norm") return new float[768];
                if (Invalid == "nonfinite") {var bad=new float[768]; bad[0]=float.NaN; return bad;}
                return Returned = Inner.Embed(ids,mask,image,audio);
            }
        }

        sealed class Parts : IDisposable
        {
            public readonly TrackingPre Pre;
            public readonly TrackingEnc Enc;
            public readonly TrackingBack Back;
            public readonly MultimodalPromptBuilder Prompts;
            public readonly Texture2D Texture = new Texture2D(2,2,TextureFormat.RGBA32,false);
            public Parts(BackendType backend = BackendType.CPU)
            {
                var profile = new MultimodalBackboneProfile(280,10,11,256,0,32,0);
                Pre = new TrackingPre {Backend=backend};
                Enc = new TrackingEnc {Inner=new ImageFeatureEncoder(EncoderModel(),backend,new ImageEncoderProfile(768,768)),Backend=backend};
                Back = new TrackingBack {Inner=new MultimodalBackbone(BackboneModel(),profile,backend),Backend=backend,Profile=profile};
                Prompts=Builder(profile);
            }
            public void Dispose() {Enc.Inner.Dispose(); Back.Inner.Dispose(); UnityEngine.Object.DestroyImmediate(Texture);}
        }

        [TestCase(256)] [TestCase(252)]
        public void ImagePromptUsesConfiguredMarkersAndExactFeatureCount(int count)
        {
            var prompt=Prompt(Builder(new MultimodalBackboneProfile(count+8,10,11,count,0,32,0)),count);
            Assert.That(prompt.InputIds.Take(count+4), Is.EqualTo(new[] {1,12}.Concat(Enumerable.Repeat(10,count)).Concat(new[] {13,2})));
            Assert.That(prompt.AudioFeatureCount, Is.Zero);
        }
        [TestCase(0)] [TestCase(252)]
        public void WrongCountCannotConstructAnImagePrompt(int count)
        {Assert.Throws<ArgumentException>(() => Prompt(Builder(new MultimodalBackboneProfile(280,10,11,256,0,32,0)),count));}
        [TestCase("preprocessor")] [TestCase("encoder")] [TestCase("backbone")] [TestCase("prompts")]
        public void NullStagesAreRejected(string omit)
        {using var parts=new Parts(); Assert.Throws<ArgumentNullException>(() => Create(parts,omit));}
        [Test] public void MixedBackendsAreRejected()
        {using var parts=new Parts(); parts.Enc.Backend=BackendType.GPUCompute; Assert.Throws<ArgumentException>(() => Create(parts));}
        [TestCase("audio")] [TestCase("count")] [TestCase("prompt")]
        public void MismatchedOrMixedModalityProfilesAreRejected(string problem)
        {
            using var parts=new Parts();
            parts.Back.Profile=new MultimodalBackboneProfile(problem=="prompt"?279:280,10,11,problem=="count"?252:256,problem=="audio"?1:0,32,0);
            Assert.Throws<ArgumentException>(() => Create(parts));
        }
        [TestCase(BackendType.CPU)] [TestCase(BackendType.GPUCompute)]
        public void OwnedFinalVectorAndDeviceTokensSurviveOnlyTheirIntendedLifetimes(BackendType backend)
        {
            if(backend==BackendType.GPUCompute) Assert.That(SystemInfo.supportsComputeShaders, Is.True, "Actual GPU required; never skip.");
            using var parts=new Parts(backend); using var api=Create(parts);
            var vector=Embed(api,parts.Texture); var seed=(2588+260)/10000.0+0.25; var norm=Math.Sqrt(seed*seed+1);
            Assert.That(vector.Length, Is.EqualTo(768));
            Assert.That(vector[0], Is.EqualTo(seed/norm).Within(1e-6)); Assert.That(vector[1], Is.EqualTo(1/norm).Within(1e-6));
            Assert.That(vector.Skip(2), Is.All.EqualTo(0)); Assert.That(vector, Is.Not.SameAs(parts.Back.Returned));
            Assert.Throws<ObjectDisposedException>(() => {var ignored=parts.Pre.Last.Pixels;});
            Assert.Throws<ObjectDisposedException>(() => {var ignored=parts.Enc.Last.SoftTokens;});
            Assert.That(parts.Back.LastImage.dataOnBackend, Is.Null);
            Assert.That(parts.Texture == null, Is.False, "Caller still owns the texture.");
            parts.Pre.Value=0.5f; var next=Embed(api,parts.Texture);
            Assert.That(next, Is.Not.SameAs(vector)); Assert.That(Math.Abs(next[1]-vector[1]), Is.GreaterThan(0.01));
        }
        [TestCase("preprocessor")] [TestCase("encoder")] [TestCase("backbone")]
        public void FailureReleasesOwnedResultsAndAllowsRecovery(string stage)
        {
            using var parts=new Parts(); using var api=Create(parts);
            parts.Pre.Fail=stage=="preprocessor"; parts.Enc.Fail=stage=="encoder"; parts.Back.Fail=stage=="backbone";
            Assert.Throws<InvalidOperationException>(() => Embed(api,parts.Texture));
            if(parts.Pre.Last!=null) Assert.Throws<ObjectDisposedException>(() => {var ignored=parts.Pre.Last.Pixels;});
            if(parts.Enc.Last!=null) Assert.Throws<ObjectDisposedException>(() => {var ignored=parts.Enc.Last.SoftTokens;});
            parts.Pre.Fail=false; parts.Enc.Fail=false; parts.Back.Fail=false;
            Assert.That(Embed(api,parts.Texture).Length, Is.EqualTo(768));
        }
        [TestCase("dimension")] [TestCase("norm")] [TestCase("nonfinite")]
        public void InvalidFinalVectorCannotBeReturned(string problem)
        {
            using var parts=new Parts(); using var api=Create(parts); parts.Back.Invalid=problem;
            Assert.Throws<InvalidOperationException>(() => Embed(api,parts.Texture));
            Assert.Throws<ObjectDisposedException>(() => {var ignored=parts.Pre.Last.Pixels;});
            Assert.Throws<ObjectDisposedException>(() => {var ignored=parts.Enc.Last.SoftTokens;});
        }
        [Test] public void InvalidInputAndRoleAreRejectedBeforePreprocessing()
        {
            using var parts=new Parts(); using var api=Create(parts);
            Assert.Throws<ArgumentNullException>(() => Embed(api,null));
            Assert.Throws<ArgumentOutOfRangeException>(() => Embed(api,parts.Texture,(TextRole)99));
            Assert.Throws<ArgumentException>(() => Embed(api,parts.Texture,TextRole.Document,"<image>"));
            Assert.That(parts.Pre.Calls, Is.Zero);
        }
        [Test] public void DisposingFacadeLeavesBorrowedStagesUsable()
        {
            using var parts=new Parts(); var api=Create(parts); api.Dispose(); api.Dispose();
            Assert.Throws<ObjectDisposedException>(() => Embed(api,parts.Texture));
            using var processed=parts.Pre.Process(parts.Texture); using var encoded=parts.Enc.Encode(processed);
            Assert.That(encoded.SoftTokens.shape, Is.EqualTo(new TensorShape(256,512)));
        }
    }
}
