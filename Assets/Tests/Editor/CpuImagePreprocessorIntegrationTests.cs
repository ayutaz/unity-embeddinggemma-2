using System;
using System.Linq;
using System.Reflection;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEngine;

namespace EmbeddingGemma.Tests
{
    public sealed class CpuImagePreprocessorIntegrationTests
    {
        [TestCase(TextureFormat.RGB24,true)] [TestCase(TextureFormat.RGB24,false)]
        [TestCase(TextureFormat.BGRA32,true)] [TestCase(TextureFormat.BGRA32,false)]
        public void OtherAcceptedByteFormatsKeepStoredRgb(TextureFormat format,bool readable)
        {
            using var processor=new ImagePreprocessor(BackendType.CPU); var texture=new Texture2D(1,1,format,false,true);
            try
            {
                texture.SetPixels32(new[] {new Color32(64,128,192,7)}); texture.Apply(false,!readable);
                using var result=processor.Process(texture); var values=result.Pixels.DownloadToArray();
                Assert.That(result.Pixels.backendType,Is.EqualTo(BackendType.CPU));
                Assert.That(values[0],Is.EqualTo(64/255f).Within(1e-6));
                Assert.That(values[1],Is.EqualTo(128/255f).Within(1e-6));
                Assert.That(values[2],Is.EqualTo(192/255f).Within(1e-6));
            }
            finally {UnityEngine.Object.DestroyImmediate(texture);}
        }

        [Test] public void RealTextureCpuPreprocessingConnectsToSyntheticFinalApi()
        {
            using var preprocessor=new ImagePreprocessor(BackendType.CPU);
            var encGraph=new FunctionalGraph(); var pixels=encGraph.AddInput<float>(new TensorShape(1,2520,768),"pixels");
            var encModel=encGraph.Compile(Functional.ReduceSum(pixels,new[] {0,1,2},false)/(2520*768f)*
                Functional.Constant(new TensorShape(256,512),Enumerable.Repeat(1f,256*512).ToArray()));
            using var encoder=new ImageFeatureEncoder(encModel,BackendType.CPU,new ImageEncoderProfile(768,768));
            var graph=new FunctionalGraph(); var ids=graph.AddInput<int>(new TensorShape(1,280),"input_ids");
            var mask=graph.AddInput<int>(new TensorShape(1,280),"attention_mask");
            var image=graph.AddInput<float>(new TensorShape(256,512),"image_features");
            var audio=graph.AddInput<float>(new TensorShape(0,512),"audio_features");
            var seed=Functional.ReduceSum((ids+mask).Float(),new[] {0,1},true)/10000f+
                Functional.ReduceSum(image,new[] {0,1},true)/(256*512f)+Functional.ReduceSum(audio,new[] {0,1},true);
            var first=new float[768]; first[0]=1; var second=new float[768]; second[1]=1;
            var vector=seed*Functional.Constant(new TensorShape(1,768),first)+Functional.Constant(new TensorShape(1,768),second);
            var model=graph.Compile(vector/Functional.Sqrt(Functional.ReduceSumSquare(vector,1,true)));
            var profile=new MultimodalBackboneProfile(280,10,11,256,0,32,0);
            using var backbone=new MultimodalBackbone(model,profile,BackendType.CPU);
            var tests=AppDomain.CurrentDomain.GetAssemblies().Single(assembly=>assembly.GetName().Name=="EmbeddingGemma.Package.Editor.Tests");
            var fixture=tests.GetType("EmbeddingGemma.Tests.MultimodalPromptBuilderTests");
            const BindingFlags flags=BindingFlags.Static|BindingFlags.NonPublic;
            var builder=new MultimodalPromptBuilder((string)fixture.GetField("Json",flags).GetRawConstantValue(),
                (string)fixture.GetField("Configuration",flags).GetRawConstantValue(),profile);
            var texture=new Texture2D(2,2,TextureFormat.RGBA32,false,true);
            try
            {
                texture.SetPixels32(Enumerable.Repeat(new Color32(64,64,64,7),4).ToArray()); texture.Apply(false,false);
                using(var api=new ImageEmbedder(preprocessor,encoder,backbone,builder))
                {
                    var actual=api.Embed(texture);
                    var expectedSeed=(2588+260)/10000.0+64/255.0*2304/2520;
                    var norm=Math.Sqrt(expectedSeed*expectedSeed+1);
                    Assert.That(actual.Length,Is.EqualTo(768));
                    Assert.That(actual[0],Is.EqualTo(expectedSeed/norm).Within(1e-6));
                    Assert.That(actual[1],Is.EqualTo(1/norm).Within(1e-6)); Assert.That(actual.Skip(2),Is.All.EqualTo(0));
                }
                // Disposal of the facade must leave the borrowed real preprocessor and encoder usable.
                using var processed=preprocessor.Process(texture); using var encoded=encoder.Encode(processed);
                Assert.That(encoded.SoftTokens.backendType,Is.EqualTo(BackendType.CPU)); Assert.That(texture.isReadable,Is.True);
            }
            finally {UnityEngine.Object.DestroyImmediate(texture);}
        }
    }
}
