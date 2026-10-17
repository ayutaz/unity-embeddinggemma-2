using System;
using System.Collections;
using System.IO;
using System.Reflection;
using System.Security.Cryptography;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;

namespace EmbeddingGemma.Validation.Tests
{
    /// <summary>Opt-in bounded diagnostic prefix; never a full-model numerical acceptance.</summary>
    public sealed class PreparedImageOperationTraceTests
    {
        const BindingFlags Private = BindingFlags.Instance | BindingFlags.NonPublic;
        static object Field(object value, string name) => value.GetType().GetField(name, Private).GetValue(value);
        static string Hash(string path)
        {
            using var stream = File.OpenRead(path); using var hash = SHA256.Create();
            return BitConverter.ToString(hash.ComputeHash(stream)).Replace("-", "").ToLowerInvariant();
        }
        [UnityTearDown] public IEnumerator Exit() { if (Application.isPlaying) yield return new ExitPlayMode(); }
        [UnityTest]
        public IEnumerator PinnedImagePrefixLogsSubmissionsAndCancelsAfterGpuFence()
        {
            yield return new EnterPlayMode();
            using var cancellation = new CancellationTokenSource();
            var task = Run(cancellation); long peak = 0; var memoryLimit = false;
            while (!task.IsCompleted)
            {
                peak = Math.Max(peak, MediaExecutionMemory.Peak());
                if (peak > 12L * 1024 * 1024 * 1024) { memoryLimit = true; cancellation.Cancel(); }
                yield return null;
            }
            task.GetAwaiter().GetResult();
            Assert.That(memoryLimit, Is.False, "Memory-limit cancellation is not a completed diagnostic prefix.");
        }
        static async Task Run(CancellationTokenSource cancellation)
        {
            var directory = Environment.GetEnvironmentVariable("EG_MEDIA_BUNDLE_OUTPUT");
            var image = Environment.GetEnvironmentVariable("EG_GPU_TRACE_IMAGE");
            var output = Environment.GetEnvironmentVariable("EG_GPU_TRACE_REPORT");
            var source = Environment.GetEnvironmentVariable("EG_ACCEPTANCE_SOURCE");
            Assert.That(directory, Is.Not.Null.And.Not.Empty); Assert.That(output, Is.Not.Null.And.Not.Empty);
            Assert.That(source, Does.Match("^[0-9a-f]{40}$"));
            Assert.That(File.Exists(output), Is.False);
            var events = Path.ChangeExtension(output, ".events.jsonl"); Assert.That(File.Exists(events), Is.False);
            var descriptor = Hash(Path.Combine(directory, "bundle.json"));
            Assert.That(descriptor, Is.EqualTo("8363f9a63a95b307fbbb835f3551a41f6fa5a1d21aca3017670e6f678647e955"));
            Assert.That(Hash(image), Is.EqualTo("c2a1900a1656ce2d843e9d06801c62f927b98a51e9b95ae1159bb2c7515fa399"));
            var through = int.Parse(Environment.GetEnvironmentVariable("EG_GPU_TRACE_THROUGH") ?? "2");
            var report = new JObject { ["diagnostic_success"] = false, ["real_model_qualified"] = false,
                ["scope"] = "controlled encoder prefix; cancellation after staged GPU fence; no final vector or search acceptance",
                ["declared_runtime_source"] = source, ["bundle_descriptor_sha256"] = descriptor,
                ["runtime_assembly_sha256"] = Hash(typeof(GpuProjectionPipeline).Assembly.Location),
                ["validation_assembly_sha256"] = Hash(typeof(GpuOperationTrace).Assembly.Location),
                ["unity"] = Application.unityVersion, ["gpu"] = SystemInfo.graphicsDeviceName,
                ["graphics_api"] = SystemInfo.graphicsDeviceType.ToString(), ["through_operation"] = through,
                ["events"] = Path.GetFileName(events), ["phase"] = "load" };
            void Save() => File.WriteAllText(output, report.ToString());
            GpuMediaSession session = null; Texture2D texture = null; var targetSubmitted = false;
            Save();
            try
            {
                session = await GpuMediaBundle.Open(directory).LoadAsync("image-square", cancellation.Token);
                var encoder = Field(Field(session, "image"), "encoder");
                var pipeline = (GpuProjectionPipeline)Field(encoder, "pipeline");
                var operations = (IGpuTensorOperation[])Field(pipeline, "projections");
                Assert.That(through, Is.GreaterThanOrEqualTo(0).And.LessThan(operations.Length));
                for (var index = 0; index < operations.Length; index++)
                {
                    var original = operations[index];
                    operations[index] = GpuOperationTrace.Wrap(original, index, row =>
                    {
                        File.AppendAllText(events, row.ToString(Newtonsoft.Json.Formatting.None) + "\n");
                        report["last_event"] = row; Save();
                        if (row.Value<string>("event") == "submitted" && row.Value<int>("operation_index") == through)
                        { targetSubmitted = true; cancellation.Cancel(); }
                    });
                }
                texture = new Texture2D(2, 2, TextureFormat.RGBA32, false, true);
                Assert.That(ImageConversion.LoadImage(texture, File.ReadAllBytes(image), true), Is.True);
                report["actual_backend"] = session.Backend.ToString(); report["phase"] = "encoder_prefix"; Save();
                var cancelled = false;
                try { await session.EmbedImageAsync(texture, cancellationToken: cancellation.Token); }
                catch (OperationCanceledException) when (targetSubmitted) { cancelled = true; }
                Assert.That(cancelled, Is.True, "Expected explicit prefix cancellation after the stage fence.");
                report["prefix_fence_returned"] = true; report["phase"] = "close"; Save();
                await session.CloseAsync(); Assert.That(session.IsClosed, Is.True);
                report["session_closed"] = true; report["diagnostic_success"] = true; Save();
            }
            catch (Exception error) { report["error_type"] = error.GetType().FullName; throw; }
            finally { if (texture != null) UnityEngine.Object.DestroyImmediate(texture); session?.Dispose(); Save(); }
        }
    }
}
