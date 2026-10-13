using System;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEngine;

namespace EmbeddingGemma.Tests
{
    public sealed class ImagePreprocessorTests
    {
        [Test]
        public void NullInputIsRejected()
        {
            using var processor = new ImagePreprocessor();
            Assert.Throws<ArgumentNullException>(() => processor.Process(null));
        }

        [Test]
        public void DisposedProcessorRejectsInputAndDoubleDisposeIsSafe()
        {
            var processor = new ImagePreprocessor();
            processor.Dispose();
            processor.Dispose();
            Assert.Throws<ObjectDisposedException>(() => processor.Process(null));
        }

        [Test]
        public void DestroyedInputIsRejected()
        {
            using var processor = new ImagePreprocessor();
            var texture = new Texture2D(4, 4);
            UnityEngine.Object.DestroyImmediate(texture);
            Assert.Throws<ArgumentException>(() => processor.Process(texture));
        }

        [Test]
        public void UnsupportedFloatTextureIsRejectedBeforeAllocation()
        {
            using var processor = new ImagePreprocessor();
            var texture = new Texture2D(4, 4, TextureFormat.RGBAFloat, false, true);
            try { Assert.Throws<ArgumentException>(() => processor.Process(texture)); }
            finally { UnityEngine.Object.DestroyImmediate(texture); }
        }

        [Test]
        public void UncreatedRenderTextureIsRejectedWithoutCreatingBorrowedInput()
        {
            using var processor = new ImagePreprocessor();
            var texture = new RenderTexture(4, 4, 0, RenderTextureFormat.ARGB32);
            try
            {
                Assert.Throws<ArgumentException>(() => processor.Process(texture));
                Assert.That(texture.IsCreated(), Is.False);
            }
            finally { UnityEngine.Object.DestroyImmediate(texture); }
        }

        [TestCase(96, 96, 768, 768, 256)]
        [TestCase(160, 96, 1008, 576, 252)]
        [TestCase(96, 160, 576, 1008, 252)]
        public void GPUResultUsesOfficialGeometryPositionsAndZeroPadding(int width, int height,
            int resizedWidth, int resizedHeight, int softTokens)
        {
            Assert.That(SystemInfo.supportsComputeShaders, Is.True, "A real compute GPU is required.");
            using var processor = new ImagePreprocessor();
            var texture = new Texture2D(width, height, TextureFormat.RGBA32, false, true);
            var colors = new Color32[width * height];
            Array.Fill(colors, new Color32(51, 102, 153, 7));
            texture.SetPixels32(colors);
            texture.Apply(false, true); // GPU-only input: CPU pixel access is no longer available.
            try
            {
                using var result = processor.Process(texture);
                Assert.That(result, Is.Not.Null, "GPU preprocessing is missing.");
                Assert.That(result.ResizedWidth, Is.EqualTo(resizedWidth));
                Assert.That(result.ResizedHeight, Is.EqualTo(resizedHeight));
                Assert.That(result.SoftTokenCount, Is.EqualTo(softTokens));
                Assert.That(result.Pixels.shape, Is.EqualTo(new TensorShape(1, 2520, 768)));
                Assert.That(result.Pixels.dataOnBackend.backendType, Is.EqualTo(BackendType.GPUCompute));
                var positions = result.Positions.ToArray();
                Assert.That(positions.Length, Is.EqualTo(5040));
                var columns = resizedWidth / 16;
                for (var patch = 0; patch < 2520; patch++)
                {
                    Assert.That(positions[2 * patch], Is.EqualTo(patch < softTokens * 9 ? patch % columns : -1));
                    Assert.That(positions[2 * patch + 1], Is.EqualTo(patch < softTokens * 9 ? patch / columns : -1));
                }
                var pixels = result.Pixels.DownloadToArray(); // Final diagnostic boundary only.
                for (var index = 0; index < pixels.Length; index++)
                    if (Math.Abs(pixels[index] - (index < softTokens * 9 * 768 ? (index % 3 + 1) * 0.2f : 0)) > 1e-6)
                        Assert.Fail($"Pixel {index}: {pixels[index]} is not official RGB / padding.");
                Assert.That(texture, Is.Not.Null, "The input is borrowed.");
            }
            finally { UnityEngine.Object.DestroyImmediate(texture); }
        }

        [Test]
        public void ResultOwnsIndependentGPUStorageAcrossReuseAndProcessorDispose()
        {
            var processor = new ImagePreprocessor();
            var texture = new Texture2D(4, 4, TextureFormat.RGBA32, false, true);
            ImagePreprocessingResult first = null;
            ImagePreprocessingResult second = null;
            try
            {
                var colors = new Color32[16];
                Array.Fill(colors, new Color32(255, 0, 0, 0));
                texture.SetPixels32(colors); texture.Apply(false, false);
                first = processor.Process(texture);
                Assert.That(first, Is.Not.Null, "Result ownership is missing.");
                Array.Fill(colors, new Color32(0, 255, 0, 255));
                texture.SetPixels32(colors); texture.Apply(false, false);
                second = processor.Process(texture);
                processor.Dispose();
                var red = first.Pixels.DownloadToArray();
                var green = second.Pixels.DownloadToArray();
                Assert.That(red[0], Is.EqualTo(1).Within(1e-6));
                Assert.That(red[1], Is.EqualTo(0));
                Assert.That(green[0], Is.EqualTo(0));
                Assert.That(green[1], Is.EqualTo(1).Within(1e-6));
                first.Dispose(); first.Dispose();
                Assert.Throws<ObjectDisposedException>(() => { var unused = first.Pixels; });
            }
            finally
            {
                first?.Dispose(); second?.Dispose(); processor.Dispose();
                UnityEngine.Object.DestroyImmediate(texture);
            }
        }
    }
}
