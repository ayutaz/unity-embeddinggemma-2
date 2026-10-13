using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Security.Cryptography;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.Experimental.Rendering;

namespace EmbeddingGemma.Tests
{
    public sealed class ImagePreprocessorReferenceTests
    {
        static string DirectoryPath => Path.Combine(Application.dataPath, "../artifacts/image-preprocessing");
        static string OutputPath => Path.Combine(DirectoryPath, "unity-results.json");

        [TestCase("texture_linear")]
        [TestCase("texture_srgb")]
        [TestCase("render_linear")]
        [TestCase("render_srgb")]
        public void OfficialPixelsPositionsAndGPUConsumerAgree(string path)
        {
            Assert.That(SystemInfo.supportsComputeShaders, Is.True, "A real GPU is required; never skip.");
            var reference = JObject.Parse(File.ReadAllText(Path.Combine(DirectoryPath, "image-preprocessing-reference.json")));
            Assert.That(reference.Value<string>("stage"), Is.EqualTo("image_preprocessing_only"));
            Assert.That(reference["metadata"].Value<string>("model_revision"), Is.EqualTo("914f7f89142e33e77833254d9c9b90c3cef7303b"));
            foreach (var item in (JObject)reference["files"])
            {
                var bytes = File.ReadAllBytes(Path.Combine(DirectoryPath, item.Key));
                Assert.That(bytes.LongLength, Is.EqualTo(item.Value.Value<long>("bytes")));
                using var sha = SHA256.Create();
                Assert.That(BitConverter.ToString(sha.ComputeHash(bytes)).Replace("-", "").ToLowerInvariant(),
                    Is.EqualTo(item.Value.Value<string>("sha256")));
            }
            var report = File.Exists(OutputPath) ? JObject.Parse(File.ReadAllText(OutputPath)) : new JObject();
            report["stage"] = "image_preprocessing_only";
            report["encoder_validated"] = false;
            report["end_to_end_validated"] = false;
            report["gpu"] = SystemInfo.graphicsDeviceName;
            report["graphics_api"] = SystemInfo.graphicsDeviceType.ToString();
            report["runtime_assembly"] = typeof(ImagePreprocessor).Assembly.GetName().Name;
            report["reference_source"] = reference["metadata"]["source_commit"];
            var rows = new JArray(); report[path] = rows;
            void Save() => File.WriteAllText(OutputPath, report.ToString());
            Save();
            using var processor = new ImagePreprocessor();
            var graph = new FunctionalGraph();
            var pixels = graph.AddInput<float>(new TensorShape(1, 2520, 768), "pixels");
            var model = graph.Compile(pixels + 0.25f);
            using var consumer = new Worker(model, BackendType.GPUCompute);
            const BindingFlags flags = BindingFlags.Instance | BindingFlags.NonPublic;
            var cpu = (HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallback", flags).GetValue(consumer);
            var fallback = cpu.Intersect(model.layers.SelectMany(layer => layer.outputs)
                .Concat(model.inputs.Select(input => input.index))).ToArray();
            var flush = (HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallbackShouldFlushGPU", flags).GetValue(consumer);
            Assert.That(fallback, Is.Empty); Assert.That(flush, Is.Empty);
            foreach (var row in (JArray)reference["cases"])
            {
                var item = new JObject { ["id"] = row.Value<string>("id"), ["success"] = false };
                rows.Add(item); Save();
                var width = row.Value<int>("width"); var height = row.Value<int>("height");
                var raw = File.ReadAllBytes(Path.Combine(DirectoryPath, row.Value<string>("rgb_file")));
                var linear = path.EndsWith("linear", StringComparison.Ordinal);
                var texture = new Texture2D(width, height, TextureFormat.RGBA32, false, linear);
                RenderTexture rendered = null;
                try
                {
                    var colors = new Color32[width * height];
                    for (var y = 0; y < height; y++)
                        for (var x = 0; x < width; x++)
                        {
                            var offset = (y * width + x) * 3;
                            // Unity CPU pixels are bottom-up; the processor's RGB fixture is top-down.
                            colors[(height - y - 1) * width + x] = new Color32(raw[offset], raw[offset + 1], raw[offset + 2],
                                (byte)((x * 19 + y * 23) % 256)); // Alpha must not affect RGB.
                        }
                    texture.SetPixels32(colors); texture.Apply(false, true);
                    Texture input = texture;
                    if (path.StartsWith("render", StringComparison.Ordinal))
                    {
                        var format = linear ? GraphicsFormat.R8G8B8A8_UNorm : GraphicsFormat.R8G8B8A8_SRGB;
                        Assert.That(SystemInfo.IsFormatSupported(format, GraphicsFormatUsage.Render), Is.True);
                        rendered = new RenderTexture(new RenderTextureDescriptor(width, height) { graphicsFormat = format, depthBufferBits = 0 });
                        Assert.That(rendered.Create(), Is.True);
                        Graphics.CopyTexture(texture, rendered); // Exact storage transfer; no shader/gamma conversion.
                        input = rendered;
                    }
                    using var result = processor.Process(input);
                    Assert.That(result, Is.Not.Null, "Official GPU preprocessing is missing.");
                    Assert.That(result.Pixels.backendType, Is.EqualTo(BackendType.GPUCompute));
                    Assert.That(result.ResizedWidth, Is.EqualTo(row.Value<int>("resized_width")));
                    Assert.That(result.ResizedHeight, Is.EqualTo(row.Value<int>("resized_height")));
                    Assert.That(result.SoftTokenCount, Is.EqualTo(row.Value<int>("soft_tokens")));
                    var positions = Read<int>(row.Value<string>("positions_file"));
                    Assert.That(result.Positions.ToArray(), Is.EqualTo(positions));
                    consumer.SetInput("pixels", result.Pixels); consumer.Schedule();
                    var output = (Tensor<float>)consumer.PeekOutput();
                    Assert.That(output.backendType, Is.EqualTo(BackendType.GPUCompute));
                    var final = output.DownloadToArray(); // First readback, after the GPU consumer.
                    var expected = Read<float>(row.Value<string>("pixels_file"));
                    var actual = result.Pixels.DownloadToArray(); // Diagnostic readback only after chaining.
                    double maximum = 0, chainMaximum = 0; var mismatches = 0;
                    Assert.That(actual.Length, Is.EqualTo(expected.Length));
                    for (var index = 0; index < expected.Length; index++)
                    {
                        var difference = Math.Abs((double)actual[index] - expected[index]);
                        var chained = Math.Abs((double)final[index] - (expected[index] + 0.25f));
                        if (double.IsNaN(difference) || double.IsInfinity(difference) || difference > 1e-6 || chained > 1e-6) mismatches++;
                        maximum = Math.Max(maximum, difference); chainMaximum = Math.Max(chainMaximum, chained);
                    }
                    item["maximum_absolute_error"] = maximum;
                    item["downstream_maximum_absolute_error"] = chainMaximum;
                    item["mismatched_elements"] = mismatches;
                    item["actual_backend"] = result.Pixels.backendType.ToString();
                    item["download_before_chaining"] = false;
                    item["consumer_cpu_fallback"] = fallback.Length;
                    item["consumer_readback_flush"] = flush.Count;
                    Save();
                    Assert.That(mismatches, Is.EqualTo(0), row.Value<string>("id"));
                    Assert.That(input, Is.Not.Null);
                    if (rendered != null) Assert.That(rendered.IsCreated(), Is.True);
                    item["success"] = true; Save();
                }
                finally
                {
                    if (rendered != null) { rendered.Release(); UnityEngine.Object.DestroyImmediate(rendered); }
                    UnityEngine.Object.DestroyImmediate(texture);
                }
            }
        }

        static T[] Read<T>(string name) where T : struct
        {
            var bytes = File.ReadAllBytes(Path.Combine(DirectoryPath, name));
            var result = new T[bytes.Length / 4];
            Buffer.BlockCopy(bytes, 0, result, 0, bytes.Length);
            return result;
        }
    }
}
