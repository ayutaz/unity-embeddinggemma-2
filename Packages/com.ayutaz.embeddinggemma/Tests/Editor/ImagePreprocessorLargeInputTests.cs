using System;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.TestTools;

namespace EmbeddingGemma.Tests
{
    public sealed class ImagePreprocessorLargeInputTests
    {
        [Test]
        public void LargestSquareInputDoesNotExceedOneDispatchDimension()
        {
            Assert.That(SystemInfo.supportsComputeShaders, Is.True, "Real GPU required.");
            Assert.That(SystemInfo.maxTextureSize, Is.GreaterThanOrEqualTo(8192));
            var texture = new RenderTexture(new RenderTextureDescriptor(8192, 8192)
                { graphicsFormat = GraphicsFormat.R8G8B8A8_UNorm, depthBufferBits = 0 });
            var previous = RenderTexture.active;
            try
            {
                Assert.That(texture.Create(), Is.True);
                RenderTexture.active = texture;
                GL.Clear(false, true, new Color(0.2f, 0.4f, 0.6f, 1));
                RenderTexture.active = previous;
                using var processor = new ImagePreprocessor();
                using var result = processor.Process(texture);
                var pixels = result.Pixels.DownloadToArray();
                Assert.That(result.ResizedWidth, Is.EqualTo(768));
                Assert.That(result.ResizedHeight, Is.EqualTo(768));
                for (var index = 0; index < pixels.Length; index++)
                {
                    var expected = index < 2304 * 768 ? (index % 3 + 1) * 0.2f : 0;
                    if (Math.Abs(pixels[index] - expected) > 1e-6 || float.IsNaN(pixels[index]))
                        Assert.Fail($"Largest-input pixel {index}: expected {expected}, got {pixels[index]}.");
                }
            }
            finally
            {
                RenderTexture.active = previous;
                texture.Release(); UnityEngine.Object.DestroyImmediate(texture);
            }
            LogAssert.NoUnexpectedReceived();
        }
    }
}
