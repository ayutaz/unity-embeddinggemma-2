using NUnit.Framework;
using UnityEditor;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class BoundedConsumerCacheTests
    {
        [Test]
        public void BoundedConsumerUsesProjectDisabledCacheInsteadOfGlobalServer()
        {
            Assert.That(EditorSettings.cacheServerMode,Is.EqualTo(CacheServerMode.Disabled),
                "Bounded local validation must not wait for the unavailable global Accelerator.");
        }
    }
}
