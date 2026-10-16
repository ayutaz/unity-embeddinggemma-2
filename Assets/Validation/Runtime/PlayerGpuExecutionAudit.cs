using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Newtonsoft.Json.Linq;
using Unity.InferenceEngine;

namespace EmbeddingGemma.Validation
{
    /// <summary>Metadata-only audit of the pinned Sentis Worker's actual layer assignments.</summary>
    public static class PlayerGpuExecutionAudit
    {
        const BindingFlags Private = BindingFlags.Instance | BindingFlags.NonPublic;

        public static JObject Evaluate(string backend, int[] computedOutputs, int[] cpuFallback, int[] intermediateFlush)
        {
            if (computedOutputs == null || cpuFallback == null || intermediateFlush == null)
                throw new ArgumentNullException("Live Worker layer assignments are required.");
            var computed = new HashSet<int>(computedOutputs);
            var cpu = cpuFallback.Intersect(computed).OrderBy(i => i).ToArray();
            var flush = intermediateFlush.Distinct().OrderBy(i => i).ToArray();
            if (backend != "GPUCompute" || computed.Count == 0 || cpu.Length != 0 || flush.Length != 0)
                throw new InvalidOperationException("GPU execution audit failed: backend=" + backend + ", computed outputs=" + computed.Count
                    + ", CPU computation outputs=[" + string.Join(",", cpu) + "], intermediate flush=[" + string.Join(",", flush) + "].");
            return new JObject { ["success"] = true, ["worker_backend"] = backend,
                ["computed_layer_output_count"] = computed.Count, ["computation_cpu_fallback_indices"] = new JArray(cpu),
                ["intermediate_flush_indices"] = new JArray(flush),
                ["cpu_shape_or_input_indices"] = new JArray(cpuFallback.Except(computed).OrderBy(i => i)),
                ["tensor_values_downloaded_by_auditor"] = false };
        }

        public static JObject Inspect(Worker worker)
        {
            if (worker == null) throw new ArgumentNullException(nameof(worker));
            var model = typeof(Worker).GetField("m_Model", Private)?.GetValue(worker) as Model;
            var cpu = typeof(Worker).GetField("m_LayerCPUFallback", Private)?.GetValue(worker) as HashSet<int>;
            var flush = typeof(Worker).GetField("m_LayerCPUFallbackShouldFlushGPU", Private)?.GetValue(worker) as HashSet<int>;
            if (model == null || cpu == null || flush == null)
                throw new NotSupportedException("Pinned Sentis Worker metadata is unavailable; cannot certify GPU execution.");
            return Evaluate(worker.backendType.ToString(), model.layers.SelectMany(layer => layer.outputs).ToArray(), cpu.ToArray(), flush.ToArray());
        }
    }
}
