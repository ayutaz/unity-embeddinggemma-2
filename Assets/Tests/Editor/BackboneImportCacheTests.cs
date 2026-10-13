using System;
using System.Collections.Generic;
using NUnit.Framework;

namespace EmbeddingGemma.Tests
{
    internal sealed class BackboneImportCache<T>
    {
        readonly Dictionary<string, Lazy<T>> values = new Dictionary<string, Lazy<T>>();
        internal T Get(string key, Func<T> prepare)
        {
            if (!values.TryGetValue(key, out var value)) values.Add(key, value = new Lazy<T>(prepare));
            return value.Value;
        }
        internal void Clear() => values.Clear();
    }

    public sealed class BackboneImportCacheTests
    {
        [TestCase(false)]
        [TestCase(true)]
        public void OneAuditedSourceIsPreparedOnlyOnceEvenWhenImportFails(bool fail)
        {
            var cache = new BackboneImportCache<int>();
            var calls = 0;
            int Prepare() { calls++; if (fail) throw new InvalidOperationException("original import failure"); return 42; }
            for (var index = 0; index < 4; index++)
            {
                if (fail) Assert.That(() => cache.Get("audited-source", Prepare), Throws.TypeOf<InvalidOperationException>().With.Message.EqualTo("original import failure"));
                else Assert.That(cache.Get("audited-source", Prepare), Is.EqualTo(42));
            }
            Assert.That(calls, Is.EqualTo(1), "Do not re-import the same large incompatible graph for each precision/backend condition.");
        }

        [Test]
        public void NewAuditCanRetryAfterTheOriginalFailure()
        {
            var cache = new BackboneImportCache<int>();
            var calls = 0;
            int Prepare() { calls++; throw new InvalidOperationException("original import failure"); }
            Assert.That(() => cache.Get("audited-source", Prepare), Throws.TypeOf<InvalidOperationException>());
            cache.Clear();
            Assert.That(() => cache.Get("audited-source", Prepare), Throws.TypeOf<InvalidOperationException>());
            Assert.That(calls, Is.EqualTo(2));
        }

        [Test]
        public void DifferentSourceGetsItsOwnPreparation()
        {
            var cache = new BackboneImportCache<int>();
            Assert.That(() => cache.Get("old", () => throw new InvalidOperationException()), Throws.TypeOf<InvalidOperationException>());
            Assert.That(cache.Get("new", () => 43), Is.EqualTo(43));
        }
    }
}
