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

        public static JObject EvaluateModel(Model model, string backend, int[] cpuFallback, int[] intermediateFlush)
        {
            if (model == null || cpuFallback == null || intermediateFlush == null)
                throw new ArgumentNullException("Live Worker graph and assignments are required.");
            if (model.layers.Count != 0)
                return Evaluate(backend, model.layers.SelectMany(layer => layer.outputs).ToArray(), cpuFallback, intermediateFlush);
            var inputs = model.inputs.Select(input => input.index).ToArray();
            var outputs = model.outputs.Select(output => output.index).ToArray();
            // Partition boundaries can carry existing GPU tensors without executing a layer.
            // An empty graph alone is insufficient: prove every output is an existing input.
            if (backend != "GPUCompute" || model.constants.Count != 0 || inputs.Length == 0 || outputs.Length == 0 ||
                inputs.Any(index => index < 0) || inputs.Distinct().Count() != inputs.Length ||
                outputs.Distinct().Count() != outputs.Length || outputs.Except(inputs).Any() ||
                cpuFallback.Length != 0 || intermediateFlush.Length != 0)
                throw new InvalidOperationException("Empty Worker graph is not a GPU-owned input alias.");
            return new JObject { ["success"] = true, ["worker_backend"] = backend,
                ["execution_kind"] = "input_alias", ["computed_layer_output_count"] = 0,
                ["input_alias_output_indices"] = new JArray(outputs),
                ["computation_cpu_fallback_indices"] = new JArray(), ["intermediate_flush_indices"] = new JArray(),
                ["cpu_shape_or_input_indices"] = new JArray(), ["tensor_values_downloaded_by_auditor"] = false };
        }

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
                ["execution_kind"] = "computation",
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
            return EvaluateModel(model, worker.backendType.ToString(), cpu.ToArray(), flush.ToArray());
        }
    }
}
