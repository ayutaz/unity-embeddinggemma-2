using System;
using System.Linq;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEngine;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class PlayerPreparedMediaModuleTests
    {
        sealed class Image : IAsyncImageEmbedder
        {
            public TextRole LastRole;
            public BackendType Backend => BackendType.GPUCompute;
            public float[] Embed(Texture input, TextRole role = TextRole.Raw, string title = null) { LastRole = role; return new float[768]; }
            public Task<float[]> EmbedAsync(Texture input, TextRole role = TextRole.Raw, string title = null, CancellationToken cancellationToken = default)
                => Task.FromResult(Embed(input, role, title));
            public void Dispose() { }
        }
        sealed class Audio : IAsyncAudioEmbedder
        {
            public TextRole LastRole;
            public BackendType Backend => BackendType.GPUCompute;
            public float[] Embed(AudioPcmInput input, TextRole role = TextRole.Raw, string title = null) { LastRole = role; return new float[768]; }
            public Task<float[]> EmbedAsync(AudioPcmInput input, TextRole role = TextRole.Raw, string title = null, CancellationToken cancellationToken = default)
                => Task.FromResult(Embed(input, role, title));
            public void Dispose() { }
        }
        [TestCase(false, TextRole.Query)] [TestCase(false, TextRole.Raw)]
        [TestCase(true, TextRole.Query)] [TestCase(true, TextRole.Raw)]
        public async Task PreparedAdapterForwardsTheRequestedRole(bool audio, TextRole role)
        {
            var image = new Image(); var sound = new Audio();
            var sessionConstructor = typeof(GpuMediaSession).GetConstructors(BindingFlags.Instance | BindingFlags.NonPublic).Single();
            var session = (GpuMediaSession)sessionConstructor.Invoke(new object[] { "fixture", audio ? null : image, audio ? sound : null, null, Array.Empty<IDisposable>() });
            try
            {
                var constructor = typeof(PlayerPreparedMediaModule).GetConstructors(BindingFlags.Instance | BindingFlags.NonPublic).Single();
                var metadata = new JObject { ["kind"] = audio ? "audio" : "image", ["models"] = new JArray(1), ["operations"] = new JArray() };
                var module = (IPlayerMediaModule)constructor.Invoke(new object[] { session, metadata });
                var input = new PlayerMediaCase(); var field = typeof(PlayerMediaCase).GetField("Role");
                Assert.That(field, Is.Not.Null); field.SetValue(input, role);
                await module.EmbedAsync(input, CancellationToken.None);
                Assert.That(audio ? sound.LastRole : image.LastRole, Is.EqualTo(role));
            }
            finally { await session.CloseAsync(); }
        }
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
