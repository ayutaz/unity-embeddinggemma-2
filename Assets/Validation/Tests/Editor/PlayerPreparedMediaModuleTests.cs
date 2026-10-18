using System;
using System.Linq;
using System.Reflection;
using Newtonsoft.Json.Linq;
using NUnit.Framework;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class PlayerPreparedMediaModuleTests
    {
        [TestCase(false, 13)] [TestCase(true, 11)]
        public void WorkerAuditIncludesTheTwoGraphsOwnedByNonNativeGelu(bool nativeGelu, int expected)
        {
            // SDK GpuGelu(false) composes a before/after pair around its GPU tanh kernel.
            // Those Workers are not part of the ten declared encoder segments or backbone.
            var metadata = new JObject { ["kind"] = "image", ["models"] = new JArray(Enumerable.Range(0, 10)),
                ["operations"] = new JArray(new JObject { ["kind"] = "gelu", ["sleef_tanh"] = nativeGelu },
                    new JObject { ["kind"] = "gelu", ["sleef_tanh"] = true }, new JObject { ["kind"] = "softmax" }) };
            var constructor = typeof(PlayerPreparedMediaModule).GetConstructors(BindingFlags.Instance | BindingFlags.NonPublic).Single();
            var module = constructor.Invoke(new object[] { null, metadata });
            var count = typeof(PlayerPreparedMediaModule).GetField("expectedWorkers", BindingFlags.Instance | BindingFlags.NonPublic).GetValue(module);
            Assert.That(count, Is.EqualTo(expected), "Every actual Worker must be inspected without rejecting a valid GELU recipe.");
        }
    }
}
