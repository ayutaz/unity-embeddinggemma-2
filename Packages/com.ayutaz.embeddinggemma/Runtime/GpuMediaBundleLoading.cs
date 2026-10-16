using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json.Linq;
using Unity.InferenceEngine;
using UnityEngine;

namespace EmbeddingGemma
{
    public sealed partial class GpuMediaBundle
    {
        /// <summary>Only the files needed by this module, including shared objects, without duplicates.</summary>
        public string[] FilesForModule(string name)
        {
            var module = Module(name); var files = new HashSet<string>(StringComparer.Ordinal) { (string)module["tokenizer"] };
            if ((string)module["kind"] == "text") files.Add((string)module["model"]);
            else
            {
                files.Add((string)module["tokenizer_config"]); files.Add((string)module["backbone"]);
                if (module["preprocessor"] != null) files.Add((string)module["preprocessor"]);
                foreach (var model in (JArray)module["models"]) files.Add((string)model);
                foreach (var operation in (JArray)module["operations"])
                    if (operation["weight_file"] != null) files.Add((string)operation["weight_file"]);
            }
            return files.OrderBy(value => value, StringComparer.Ordinal).ToArray();
        }

        /// <summary>
        /// Verifies all module objects before loading any model; yields between
        /// file/model/operation loads on Unity's main thread. A failed or cancelled
        /// load releases every created Worker and kernel before returning.
        /// </summary>
        public async Task<GpuMediaSession> LoadAsync(string name, CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested(); var module = Module(name);
            if (SynchronizationContext.Current == null || !SystemInfo.supportsComputeShaders)
                throw new NotSupportedException("Prepared modules require Unity's main thread and an actual compute GPU.");
            var paths = new Dictionary<string, string>(StringComparer.Ordinal);
            foreach (var file in FilesForModule(name))
            {
                await Task.Yield(); cancellationToken.ThrowIfCancellationRequested(); paths.Add(file, VerifyFile(file));
            }
            var resources = new List<IDisposable>(); var transferred = false;
            T Own<T>(T value) where T : IDisposable { resources.Add(value); return value; }
            try
            {
                if ((string)module["kind"] == "text")
                {
                    var model = ModelLoader.Load(paths[(string)module["model"]]);
                    cancellationToken.ThrowIfCancellationRequested();
                    var query = Own(new TextEmbedder(model, File.ReadAllText(paths[(string)module["tokenizer"]]), BackendType.GPUCompute));
                    AuditWorker(query, model);
                    var result = new GpuMediaSession(name, null, null, query, resources.ToArray()); transferred = true; return result;
                }
                var models = new List<Model>();
                foreach (var file in (JArray)module["models"])
                {
                    await Task.Yield(); cancellationToken.ThrowIfCancellationRequested(); models.Add(ModelLoader.Load(paths[(string)file]));
                }
                var operations = new List<IGpuTensorOperation>(); var inputs = new List<string[]>(); var outputs = new List<string>();
                foreach (JObject operation in (JArray)module["operations"])
                {
                    await Task.Yield(); cancellationToken.ThrowIfCancellationRequested();
                    operations.Add(Own(CreateOperation(operation, paths)));
                    inputs.Add(((JArray)operation["inputs"]).Select(value => (string)value).ToArray()); outputs.Add((string)operation["output"]);
                }
                var isImage = (string)module["kind"] == "image";
                var imageProfile = isImage ? new ImageEncoderProfile((int)module["resized_width"], (int)module["resized_height"]) : null;
                var audioProfile = isImage ? null : new AudioEncoderProfile((int)module["valid_audio_frames"]);
                var profile = new MultimodalBackboneProfile(1024, 258880, 258881, imageProfile?.FeatureCount ?? 0, audioProfile?.FeatureCount ?? 0, 262144, 0);
                var prompts = new MultimodalPromptBuilder(File.ReadAllText(paths[(string)module["tokenizer"]]),
                    File.ReadAllText(paths[(string)module["tokenizer_config"]]), profile);
                await Task.Yield(); cancellationToken.ThrowIfCancellationRequested();
                var backboneModel = ModelLoader.Load(paths[(string)module["backbone"]]);
                var backbone = Own(new MultimodalBackbone(backboneModel, profile, BackendType.GPUCompute)); AuditWorker(backbone, backboneModel);
                GpuMediaSession session;
                if (isImage)
                {
                    var preprocessing = Own(new ImagePreprocessor(BackendType.GPUCompute));
                    var encoder = Own(new PartitionedImageFeatureEncoder(models.ToArray(), inputs.ToArray(), outputs.ToArray(), operations.ToArray(), imageProfile));
                    var api = Own(new ImageEmbedder(preprocessing, encoder, backbone, prompts));
                    session = new GpuMediaSession(name, api, null, null, resources.ToArray());
                }
                else
                {
                    var preprocessingModel = ModelLoader.Load(paths[(string)module["preprocessor"]]);
                    var preprocessing = Own(new AudioPreprocessor(preprocessingModel, BackendType.GPUCompute)); AuditWorker(preprocessing, preprocessingModel);
                    var encoder = Own(new PartitionedAudioFeatureEncoder(models.ToArray(), inputs.ToArray(), outputs.ToArray(), operations.ToArray(), audioProfile));
                    var api = Own(new AudioEmbedder(preprocessing, encoder, backbone, prompts));
                    session = new GpuMediaSession(name, null, api, null, resources.ToArray());
                }
                cancellationToken.ThrowIfCancellationRequested(); transferred = true; return session;
            }
            finally
            {
                if (!transferred)
                    for (var index = resources.Count - 1; index >= 0; index--) resources[index].Dispose();
            }
        }
        static TensorShape Shape(JToken value) => new TensorShape(((JArray)value).Select(dimension => (int)dimension).ToArray());
        static float[] Weights(string path)
        {
            var bytes = File.ReadAllBytes(path);
            if (bytes.Length == 0 || bytes.Length % sizeof(float) != 0) throw new InvalidDataException("Expected little-endian FP32 coefficients.");
            if (!BitConverter.IsLittleEndian)
                for (var index = 0; index < bytes.Length; index += sizeof(float)) Array.Reverse(bytes, index, sizeof(float));
            var values = new float[bytes.Length / sizeof(float)]; Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length); return values;
        }
        static IGpuTensorOperation CreateOperation(JObject operation, Dictionary<string, string> paths)
        {
            float[] Coefficients() => Weights(paths[(string)operation["weight_file"]]);
            switch ((string)operation["kind"])
            {
                case "blocked_matmul": return new GpuBlockedMatmul((int)operation["width"], (int)operation["columns"], Coefficients(), (int)operation["block_size"]);
                case "fixed_left_matmul": return new GpuFixedLeftMatmul(Shape(operation["weight_shape"]), Coefficients());
                case "batched_matmul": return new GpuBatchedMatmul();
                case "gelu": return new GpuGelu(Shape(operation["shape"]), (bool)operation["release_cache"], (bool)operation["fuse_inner"], (bool)operation["sleef_tanh"]);
                case "softmax": return new GpuSoftmax((int)operation["vector_width"]);
                case "silu": return new GpuSilu();
                case "tanh": return new GpuTanh();
                case "scalar_divide": return new GpuScalarDivide((float)operation["divisor"]);
                case "glu": return new GpuGlu();
                case "sequential_scale": return new GpuSequentialScale((float)operation["first_scale"], Coefficients());
                default: throw new InvalidDataException("Unsupported GPU operation.");
            }
        }
        static void AuditWorker(object owner, Model model)
        {
            const BindingFlags fields = BindingFlags.Instance | BindingFlags.NonPublic;
            var worker = owner.GetType().GetField("worker", fields)?.GetValue(owner) as Worker;
            var cpu = typeof(Worker).GetField("m_LayerCPUFallback", fields)?.GetValue(worker) as HashSet<int>;
            var flush = typeof(Worker).GetField("m_LayerCPUFallbackShouldFlushGPU", fields)?.GetValue(worker) as HashSet<int>;
            if (worker == null || worker.backendType != BackendType.GPUCompute || cpu == null || flush == null || flush.Count != 0 ||
                cpu.Intersect(model.layers.SelectMany(layer => layer.outputs).Concat(model.inputs.Select(input => input.index))).Any())
                throw new NotSupportedException("Prepared modules cannot use CPU computation or intermediate GPU readback.");
        }
    }
}
