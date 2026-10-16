using System;
using System.Linq;
using System.Reflection;
using System.Runtime.ExceptionServices;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.Experimental.Rendering;

namespace EmbeddingGemma.Tests
{
    public sealed class CpuImagePreprocessorTests
    {
        internal static ImagePreprocessor Create(BackendType backend = BackendType.CPU)
        {
            var constructor=typeof(ImagePreprocessor).GetConstructor(new[] {typeof(BackendType)});
            Assert.That(constructor, Is.Not.Null, "Explicit CPU image preprocessing is missing.");
            try {return (ImagePreprocessor)constructor.Invoke(new object[] {backend});}
            catch(TargetInvocationException error) {ExceptionDispatchInfo.Capture(error.InnerException).Throw(); throw;}
        }

        [Test] public void UnsupportedBackendIsRejectedBeforeAllocation()
        {Assert.Throws<ArgumentOutOfRangeException>(() => Create((BackendType)99));}

        [TestCase(96,96,768,768,256)] [TestCase(160,96,1008,576,252)] [TestCase(96,160,576,1008,252)]
        public void CpuProducesOfficialGeometryAndIndependentOwnedPixels(int width,int height,int resizedWidth,int resizedHeight,int count)
        {
            using var processor=Create(); var texture=new Texture2D(width,height,TextureFormat.RGBA32,false,true);
            try
            {
                texture.SetPixels32(Enumerable.Repeat(new Color32(51,102,153,7),width*height).ToArray()); texture.Apply(false,false);
                using var result=processor.Process(texture);
                Assert.That(processor.Backend, Is.EqualTo(BackendType.CPU)); Assert.That(result.Pixels.backendType, Is.EqualTo(BackendType.CPU));
                Assert.That(result.ResizedWidth, Is.EqualTo(resizedWidth)); Assert.That(result.ResizedHeight, Is.EqualTo(resizedHeight));
                Assert.That(result.SoftTokenCount, Is.EqualTo(count)); Assert.That(result.Pixels.shape, Is.EqualTo(new TensorShape(1,2520,768)));
                var values=result.Pixels.DownloadToArray();
                for(var index=0;index<values.Length;index++)
                    if(Math.Abs(values[index]-(index<count*9*768?(index%3+1)*0.2f:0))>1e-6) Assert.Fail("Wrong RGB or padded value at "+index);
                Assert.That(texture.isReadable, Is.True); Assert.That(texture.GetPixels32()[0].a, Is.EqualTo(7));
            }
            finally {UnityEngine.Object.DestroyImmediate(texture);}
        }

        [TestCase("readable_linear")] [TestCase("readable_srgb")]
        [TestCase("texture_linear")] [TestCase("texture_srgb")]
        [TestCase("render_linear")] [TestCase("render_srgb")]
        public void IdentityImageKeepsStoredRgbOrientationAndIgnoresAlpha(string path)
        {
            using var processor=Create(); var linear=path.EndsWith("linear",StringComparison.Ordinal);
            var texture=new Texture2D(768,768,TextureFormat.RGBA32,false,linear); RenderTexture render=null;
            var colors=new Color32[768*768];
            for(var y=0;y<768;y++) for(var x=0;x<768;x++)
                colors[(767-y)*768+x]=new Color32((byte)(x*7%256),(byte)(y*11%256),(byte)((x+y)*5%256),(byte)(x%256));
            try
            {
                texture.SetPixels32(colors); texture.Apply(false,!path.StartsWith("readable",StringComparison.Ordinal)); Texture input=texture;
                if(path.StartsWith("render",StringComparison.Ordinal))
                {
                    render=new RenderTexture(new RenderTextureDescriptor(768,768) {graphicsFormat=linear?GraphicsFormat.R8G8B8A8_UNorm:GraphicsFormat.R8G8B8A8_SRGB,depthBufferBits=0});
                    Assert.That(render.Create(), Is.True); Graphics.CopyTexture(texture,render); input=render;
                }
                var originalActive=RenderTexture.active; using var result=processor.Process(input);
                Assert.That(RenderTexture.active, Is.SameAs(originalActive)); Assert.That(result.Pixels.backendType, Is.EqualTo(BackendType.CPU));
                var values=result.Pixels.DownloadToArray(); var positions=result.Positions.ToArray();
                for(var patch=0;patch<2520;patch++)
                {
                    Assert.That(positions[2*patch], Is.EqualTo(patch<2304?patch%48:-1));
                    Assert.That(positions[2*patch+1], Is.EqualTo(patch<2304?patch/48:-1));
                    for(var local=0;local<768;local++)
                    {
                        var x=(patch%48)*16+(local/3)%16; var y=(patch/48)*16+local/48;
                        var expected=patch>=2304?0:local%3==0?x*7%256:local%3==1?y*11%256:(x+y)*5%256;
                        if(Math.Abs(values[patch*768+local]-expected/255f)>1e-6) Assert.Fail(path+" wrong identity patch at "+(patch*768+local));
                    }
                }
                if(render!=null) Assert.That(render.IsCreated(), Is.True);
            }
            finally {if(render!=null){render.Release(); UnityEngine.Object.DestroyImmediate(render);} UnityEngine.Object.DestroyImmediate(texture);}
        }

        [Test] public void ResultSurvivesReuseAndProcessorDisposal()
        {
            var processor=Create(); var texture=new Texture2D(1,1,TextureFormat.RGBA32,false,true);
            ImagePreprocessingResult first=null,second=null;
            try
            {
                texture.SetPixels32(new[] {new Color32(255,0,0,7)}); texture.Apply(false,false); first=processor.Process(texture);
                texture.SetPixels32(new[] {new Color32(0,255,0,255)}); texture.Apply(false,false); second=processor.Process(texture); processor.Dispose();
                Assert.That(first.Pixels.DownloadToArray()[0], Is.EqualTo(1)); Assert.That(second.Pixels.DownloadToArray()[1], Is.EqualTo(1));
                first.Dispose(); first.Dispose(); Assert.Throws<ObjectDisposedException>(() => {var ignored=first.Pixels;});
                Assert.Throws<ObjectDisposedException>(() => processor.Process(texture));
            }
            finally {first?.Dispose(); second?.Dispose(); processor.Dispose(); UnityEngine.Object.DestroyImmediate(texture);}
        }

        [Test] public void CpuNullDestroyedAndFloatInputsAreRejected()
        {
            using var processor=Create(); Assert.Throws<ArgumentNullException>(() => processor.Process(null));
            var texture=new Texture2D(1,1,TextureFormat.RGBAFloat,false,true);
            try {Assert.Throws<ArgumentException>(() => processor.Process(texture));}
            finally {UnityEngine.Object.DestroyImmediate(texture);}
            Assert.Throws<ArgumentException>(() => processor.Process(texture));
        }
    }
}
