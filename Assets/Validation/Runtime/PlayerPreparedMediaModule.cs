using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json.Linq;
using Unity.InferenceEngine;
using UnityEngine;

namespace EmbeddingGemma.Validation
{
    /// <summary>Real GPU bundle adapter. Reflection is limited to the pinned SDK/Worker ownership contract.</summary>
    public sealed class PlayerPreparedMediaModule : IPlayerMediaModule
    {
        const BindingFlags Private = BindingFlags.Instance | BindingFlags.NonPublic;
        readonly GpuMediaSession session;
        readonly int expectedWorkers, expectedEncoderSegments, expectedAuxiliaryWorkers;
        readonly bool image;
        PlayerPreparedMediaModule(GpuMediaSession session, JObject metadata)
        {
            this.session = session; image = (string)metadata["kind"] == "image";
            expectedEncoderSegments = ((JArray)metadata["models"]).Count;
            expectedAuxiliaryWorkers = 2 * ((JArray)metadata["operations"]).Count(operation =>
                (string)operation["kind"] == "gelu" && (bool?)operation["sleef_tanh"] == false);
            expectedWorkers = expectedEncoderSegments + expectedAuxiliaryWorkers + (image ? 1 : 2);
        }
        public static async Task<IPlayerMediaModule> Load(GpuMediaBundle bundle, string module)
        {
            var metadata = bundle.Module(module);
            if ((string)metadata["kind"] != "image" && (string)metadata["kind"] != "audio")
                throw new ArgumentException("Use the existing text Player harness for text coverage.");
            return new PlayerPreparedMediaModule(await bundle.LoadAsync(module), metadata);
        }
        public Task<float[]> EmbedAsync(PlayerMediaCase input, CancellationToken token)
            => image ? session.EmbedImageAsync((Texture)input.Input, cancellationToken: token)
                : session.EmbedAudioAsync((AudioPcmInput)input.Input, cancellationToken: token);
        IDisposable[] Resources()
            => typeof(GpuMediaSession).GetField("resources", Private)?.GetValue(session) as IDisposable[]
                ?? throw new NotSupportedException("Live module ownership metadata is unavailable.");
        public JObject InspectGpuExecution()
        {
            if (session.IsClosed) throw new ObjectDisposedException(nameof(GpuMediaSession));
            var workers = new List<Worker>(); var segments = 0; var auxiliary = 0;
            foreach (var resource in Resources())
            {
                if (resource == null) throw new NotSupportedException("Missing live module resource.");
                var worker = resource.GetType().GetField("worker", Private)?.GetValue(resource) as Worker;
                if (worker != null) workers.Add(worker);
                var pipeline = resource.GetType().GetField("pipeline", Private)?.GetValue(resource) as GpuProjectionPipeline;
                if (pipeline == null) continue;
                var pipelineWorkers = typeof(GpuProjectionPipeline).GetField("workers", Private)?.GetValue(pipeline) as Worker[];
                if (pipelineWorkers == null || pipelineWorkers.Length == 0 || pipelineWorkers.Any(value => value == null))
                    throw new NotSupportedException("Live encoder segment Workers are unavailable.");
                if (resource is PartitionedImageFeatureEncoder || resource is PartitionedAudioFeatureEncoder) segments += pipelineWorkers.Length;
                else auxiliary += pipelineWorkers.Length;
                workers.AddRange(pipelineWorkers);
            }
            if (segments == 0 || segments != expectedEncoderSegments || auxiliary != expectedAuxiliaryWorkers ||
                workers.Count != expectedWorkers || workers.Distinct().Count() != workers.Count)
                throw new NotSupportedException("Live Worker count differs from the prepared module recipe.");
            return new JObject { ["success"] = true, ["worker_count"] = workers.Count, ["encoder_segment_count"] = segments, ["auxiliary_worker_count"] = auxiliary,
                ["workers"] = new JArray(workers.Select(PlayerGpuExecutionAudit.Inspect)), ["actual_backend"] = "GPUCompute" };
        }
        public Task CloseAsync() => session.CloseAsync();
        public bool IsReleased
        {
            get
            {
                if (!session.IsClosed) return false;
                foreach (var resource in Resources())
                    foreach (var field in resource.GetType().GetFields(Private))
                    {
                        var value = field.GetValue(resource);
                        if ((typeof(Worker).IsAssignableFrom(field.FieldType) || field.FieldType == typeof(GpuProjectionPipeline) ||
                            field.FieldType == typeof(GpuConstantWorker)) && value != null) return false;
                        if (value is ComputeBuffer buffer && buffer.IsValid()) return false;
                        if (value is ComputeShader shader && shader != null) return false;
                        if (value is Tensor tensor)
                            try { if (tensor.dataOnBackend != null) return false; } catch (ObjectDisposedException) { }
                    }
                return true;
            }
        }
    }
}
