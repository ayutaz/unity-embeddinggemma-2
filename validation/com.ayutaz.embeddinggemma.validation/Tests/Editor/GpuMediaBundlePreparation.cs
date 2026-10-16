using System;
using System.IO;
using System.Linq;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEngine;

namespace EmbeddingGemma.Validation.Tests
{
    /// <summary>Explicit preparation entry point. Reuses qualified graphs; does not run model inference or generate references.</summary>
    public static class GpuMediaBundlePreparation
    {
        public static void Run()
        {
            var output = Environment.GetEnvironmentVariable("EG_MEDIA_BUNDLE_OUTPUT");
            var module = Environment.GetEnvironmentVariable("EG_MEDIA_BUNDLE_MODULE");
            var source = Environment.GetEnvironmentVariable("EG_ACCEPTANCE_SOURCE");
            Assert.That(output, Is.Not.Null.And.Not.Empty);
            var locked = JObject.Parse(File.ReadAllText(Path.Combine(Application.dataPath, "../Packages/packages-lock.json")));
            foreach (var package in new[] { "com.ayutaz.embeddinggemma", "com.ayutaz.embeddinggemma.validation" })
                Assert.That((string)locked["dependencies"][package]["hash"], Is.EqualTo(source), "Preparation packages must match the fixed execution source.");
            if (module == "text")
            {
                var bundle = Path.GetFullPath(Path.Combine(Application.dataPath, "../../player-bundle/windows-unity21-20261012"));
                var audit = JObject.Parse(File.ReadAllText(Path.Combine(bundle, "bundle.json")));
                var writer = new GpuMediaBundleWriter(output, source);
                string Store(string name)
                {
                    var entry = audit["files"].Single(e => (string)e["name"] == name); var path = Path.Combine(bundle, name);
                    Check(path, (string)entry["sha256"]); Assert.That(new FileInfo(path).Length, Is.EqualTo((long)entry["bytes"]));
                    return writer.StoreFile(path);
                }
                writer.CommitModule("text", new JObject { ["kind"] = "text", ["model"] = Store("model-fp32.sentis"), ["tokenizer"] = Store("tokenizer.json") });
            }
            else if (module == "image-square" || module == "image-landscape" || module == "image-portrait")
                VisionNativePatchWholeGpuTests.PrepareGpuBundle(module.Substring(6) + "-0", output).GetAwaiter().GetResult();
            else if (module == "audio-1119" || module == "audio-839")
                new PartitionedAudioGpuReferenceTests().PrepareGpuBundle(module == "audio-839" ? "padded_noise" : "silence", output).GetAwaiter().GetResult();
            else throw new ArgumentException("Select text, image-square/landscape/portrait or audio-1119/839 explicitly.");
            Debug.Log("GPU media bundle prepared without model inference: " + module);
        }
        static string Cache(string path) => Path.GetFullPath(Path.Combine(Application.dataPath, "../../" + path));
        static void Check(string path, string expected) { using var stream = File.OpenRead(path); Assert.That(PlayerFileHash.Compute(stream, out _), Is.EqualTo(expected), path); }
        static string StoreChecked(GpuMediaBundleWriter writer, string path, string sha) { Check(path, sha); return writer.StoreFile(path); }

        public static JObject ImageOperation(GpuMediaBundleWriter writer, SavedProjectionPartition.Plan plan, int index, string kind,
            int blockSize, int vectorWidth, bool releaseCache, bool fuseInner, bool sleefTanh)
        {
            var op = new JObject { ["inputs"] = new JArray(plan.Inputs[index]), ["output"] = plan.Outputs[index] };
            if (kind == "native_fixed_left_pooling_block192")
            {
                var shape = plan.LeftWeightShapes[index]; op["kind"] = "fixed_left_matmul";
                op["weight_shape"] = new JArray(Enumerable.Range(0, shape.rank).Select(axis => shape[axis]));
                op["weight_file"] = writer.StoreWeights(plan.LeftWeights[index]);
            }
            else if (kind == "native_batched_matmul") op["kind"] = "batched_matmul";
            else if (kind == "native_softmax") { op["kind"] = "softmax"; op["vector_width"] = vectorWidth; }
            else if (kind == "native_precise_sleef_gelu" || kind == "native_tanh_gelu")
            {
                op["kind"] = "gelu"; op["shape"] = new JArray(1, 2520, 3072); op["release_cache"] = releaseCache;
                op["fuse_inner"] = fuseInner; op["sleef_tanh"] = sleefTanh;
            }
            else
            {
                Assert.That(kind, Does.StartWith("native_static_block")); op["kind"] = "blocked_matmul";
                op["width"] = plan.Widths[index]; op["columns"] = plan.Columns[index]; op["block_size"] = blockSize;
                op["weight_file"] = writer.StoreWeights(plan.Weights[index]);
            }
            return op;
        }
        public static void CommitImage(GpuMediaBundleWriter writer, string profile, Model[] models, JArray operations, string originalCoefficients)
        {
            var tokens = Cache("k/artifacts/multimodal-prompts");
            var backbone = profile == "square" ? Path.Combine(QualifiedImageBackboneSquare.Root, "square-fp32.sentis") : Cache("numerical-acceptance-state/il11830367896/landscape-fp32.sentis");
            var module = new JObject { ["kind"] = "image", ["resized_width"] = profile == "landscape" ? 1008 : profile == "portrait" ? 576 : 768,
                ["resized_height"] = profile == "landscape" ? 576 : profile == "portrait" ? 1008 : 768,
                ["models"] = new JArray(models.Select(writer.StoreModel)), ["operations"] = operations,
                ["backbone"] = StoreChecked(writer, backbone, profile == "square" ? "f6f4ba9d036e70c3d26b5932daca3facecbc0c45753a66b2da72e7b8f02519ce" : "178aea5983304c6f6bac3c52044c9c12dc71a459a38bd856e7fba75b70478597"),
                ["tokenizer"] = StoreChecked(writer, Path.Combine(tokens, "tokenizer.json"), "4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4"),
                ["tokenizer_config"] = StoreChecked(writer, Path.Combine(tokens, "tokenizer_config.json"), "17bd5d6e9364ca49a534e1502076593317c298d4a663623091ed45388f004874"),
                ["original_encoder_coefficients_sha256"] = originalCoefficients };
            writer.CommitModule("image-" + profile, module);
        }
        public static JObject AudioOperation(GpuMediaBundleWriter writer, string directory, JObject original)
        {
            var kind = (string)original["kind"];
            var op = new JObject { ["inputs"] = original["inputs"]?.DeepClone() ?? new JArray((string)original["input"]), ["output"] = (string)original["output"] };
            if (new[] { "silu", "tanh", "glu", "batched_matmul", "softmax", "scalar_divide" }.Contains(kind))
            {
                op["kind"] = kind;
                if (kind == "softmax") op["vector_width"] = 8;
                if (kind == "scalar_divide") op["divisor"] = original.Value<float>("divisor");
            }
            else
            {
                var path = Path.Combine(directory, original.Value<string>("file")); Check(path, original.Value<string>("weight_sha256"));
                op["weight_file"] = writer.StoreFile(path);
                if (kind == "sequential_scale") { op["kind"] = kind; op["first_scale"] = original.Value<float>("first_scale"); }
                else
                {
                    var shape = original["weight_shape"].ToObject<int[]>(); Assert.That(shape.Length, Is.EqualTo(2));
                    op["kind"] = "blocked_matmul"; op["width"] = shape[0]; op["columns"] = shape[1]; op["block_size"] = 192;
                }
            }
            return op;
        }
        public static void CommitAudio(GpuMediaBundleWriter writer, int validFrames, Model[] models, JArray operations)
        {
            var root = Cache("k/artifacts/audio-real-chain"); var tokens = Path.Combine(root, "tokenizer");
            var final = Cache("k/artifacts/audio-final-backbone"); var profile = "frames-1119-valid-" + validFrames;
            Check(Path.Combine(final, "audio-embedding-reference.json"), "281d12573b431667e8841033db1c6a10dce4382ef98c9a5345c18044a1194001");
            var metadata = JObject.Parse(File.ReadAllText(Path.Combine(final, "audio-embedding-reference.json")))["metadata"];
            var module = new JObject { ["kind"] = "audio", ["valid_audio_frames"] = validFrames, ["models"] = new JArray(models.Select(writer.StoreModel)), ["operations"] = operations,
                ["preprocessor"] = StoreChecked(writer, Path.Combine(root, "preprocessor-96d31a4831db40699e7210196e2acead-fp32.sentis"), "da7dd8f65ebb2789360235d657646f47e315f062fc8ccc3b10e3285ecaf6ed8d"),
                ["backbone"] = StoreChecked(writer, Path.Combine(final, profile + "-fp32.sentis"), validFrames == 839 ? "4f30639749fa7e54ad0067a99d1b28c1f8a85df23726c19adfe6d7baf38f63ad" : "484b88de0c8d77b9f36ceab79b2ffd5772444d97a3b2f4571380f159d7c389fa"),
                ["tokenizer"] = StoreChecked(writer, Path.Combine(tokens, "tokenizer.json"), metadata["input_sha256"].Value<string>("tokenizer.json")),
                ["tokenizer_config"] = StoreChecked(writer, Path.Combine(tokens, "tokenizer_config.json"), metadata["input_sha256"].Value<string>("tokenizer_config.json")) };
            writer.CommitModule("audio-" + validFrames, module);
        }
    }
}
