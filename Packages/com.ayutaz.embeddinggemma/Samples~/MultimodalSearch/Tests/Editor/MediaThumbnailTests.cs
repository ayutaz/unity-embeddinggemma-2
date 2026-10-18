using System;
using NUnit.Framework;
using UnityEngine;

namespace EmbeddingGemma.Samples.Tests
{
    public sealed class MediaThumbnailTests
    {
        [Test] public void NonReadableSourceCanBePreviewedOnGpuWithoutChangingItsAspectOrOwnership()
        {
            Assert.That(SystemInfo.graphicsDeviceType, Is.Not.EqualTo(UnityEngine.Rendering.GraphicsDeviceType.Null), "Actual graphics required; never skip.");
            var source = new Texture2D(320, 160, TextureFormat.RGBA32, false, true); source.Apply(false, true);
            using var store = new MediaPreviewStore();
            try
            {
                var preview = MediaPreviewStore.CreateThumbnail(source); Assert.That(preview, Is.Not.Null);
                store.SetImage("preview", preview);
                Assert.That(preview.width, Is.EqualTo(160)); Assert.That(preview.height, Is.EqualTo(80)); Assert.That(preview.IsCreated(), Is.True);
                Assert.That(source.isReadable, Is.False); store.Dispose(); Assert.That(preview == null, Is.True); Assert.That(source != null, Is.True);
            }
            finally { UnityEngine.Object.DestroyImmediate(source); }
        }

        [Test] public void SmallPreviewIsNotUpscaled()
        {
            var source = new Texture2D(9, 5); using var store = new MediaPreviewStore();
            try
            {
                var preview = MediaPreviewStore.CreateThumbnail(source); Assert.That(preview, Is.Not.Null); store.SetImage("small", preview);
                Assert.That(preview.width, Is.EqualTo(9)); Assert.That(preview.height, Is.EqualTo(5));
            }
            finally { UnityEngine.Object.DestroyImmediate(source); }
        }

        [Test] public void InvalidSourceOrBoundIsRejectedWithoutAllocatingPreview()
        {
            Assert.Throws<ArgumentNullException>(() => MediaPreviewStore.CreateThumbnail(null));
            var source = new Texture2D(2, 2);
            try { Assert.Throws<ArgumentOutOfRangeException>(() => MediaPreviewStore.CreateThumbnail(source, 0)); }
            finally { UnityEngine.Object.DestroyImmediate(source); }
        }
    }
}
