using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json.Linq;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.Rendering;

namespace EmbeddingGemma.Validation.Tests
{
    /// <summary>Diagnostic only: execute at most eight original segment/operation pairs, without a backbone.</summary>
    public static class IsolatedGpuPrefix
    {
        public static async Task<Tensor<float>> ExecuteAsync(Model[] models, string[][] inputs, string[] outputs,
            IGpuTensorOperation[] operations, Tensor<float> input, Action<JObject> observe, CancellationToken token = default, bool retireCompletedCache = false)
        {
            if (models == null || models.Length < 1 || models.Length > 8 || operations == null || inputs == null || outputs == null ||
                operations.Length != models.Length || inputs.Length != models.Length || outputs.Length != models.Length ||
                models.Any(m => m == null) || operations.Any(o => o == null || o.Backend != BackendType.GPUCompute))
                throw new ArgumentException("Expected one to eight original segment/operation pairs.");
            if (input == null || input.backendType != BackendType.GPUCompute || input.dataOnBackend == null || observe == null)
                throw new ArgumentException("A borrowed GPU input and diagnostic observer are required.");
            if (SynchronizationContext.Current == null || !SystemInfo.supportsGraphicsFence)
                throw new NotSupportedException("Unity main-thread GPU fences are required.");
            if (models[0].inputs.Count != 1) throw new ArgumentException("Expected the original single-input first segment.");
            var owners = new List<GpuConstantWorker>(); var owned = new HashSet<Tensor>();
            var values = new Dictionary<string, Tensor<float>> { [models[0].inputs[0].name] = input };
            var releaseWorkers = new List<Action>(); var releaseBackends = new List<Action>();
            var liveAfter = new HashSet<string>[models.Length]; var live = new HashSet<string> { outputs[outputs.Length - 1] };
            for (var stage = models.Length - 1; stage >= 0; stage--)
            {
                liveAfter[stage] = new HashSet<string>(live);
                live.UnionWith(models[stage].inputs.Select(value => value.name)); live.UnionWith(inputs[stage]);
            }
            bool pending = false, hasFence = false; GraphicsFence fence = default;
            void Emit(string name, int stage, Tensor<float> tensor = null, int? retired = null) => observe(new JObject {
                ["event"] = name, ["stage"] = stage, ["utc"] = DateTime.UtcNow.ToString("O"),
                ["backend"] = tensor?.backendType.ToString(), ["shape"] = tensor?.shape.ToString(),
                ["full_model_qualified"] = false, ["intermediate_readback"] = false, ["retired_count"] = retired });
            try
            {
                foreach (var model in models)
                {
                    token.ThrowIfCancellationRequested();
                    var owner = new GpuConstantWorker(model, BackendType.GPUCompute); owners.Add(owner);
                    const BindingFlags fields = BindingFlags.Instance | BindingFlags.NonPublic;
                    var cpu = (HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallback", fields)?.GetValue(owner.Worker);
                    var flush = (HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallbackShouldFlushGPU", fields)?.GetValue(owner.Worker);
                    var data = model.layers.SelectMany(layer => layer.outputs).Concat(model.inputs.Select(value => value.index));
                    if (cpu == null || flush == null || cpu.Intersect(data).Any() || flush.Count != 0)
                        throw new NotSupportedException("Original prefix cannot use CPU computation or intermediate readback.");
                    if (retireCompletedCache)
                    {
                        var release = typeof(GpuProjectionPipeline).GetMethod("CacheRelease", BindingFlags.Static | BindingFlags.NonPublic);
                        var backendRelease = typeof(GpuProjectionPipeline).GetMethod("CompletedBackendCacheRelease", BindingFlags.Static | BindingFlags.NonPublic);
                        if (release == null || backendRelease == null) throw new NotSupportedException("Pinned original cache retirement unavailable.");
                        releaseWorkers.Add((Action)release.Invoke(null, new object[] { owner.Worker, model }));
                        releaseBackends.Add((Action)backendRelease.Invoke(null, new object[] { owner.Worker }));
                    }
                }
                for (var stage = 0; stage < models.Length; stage++)
                {
                    token.ThrowIfCancellationRequested(); Emit("stage_begin", stage);
                    var worker = owners[stage].Worker; var model = models[stage];
                    foreach (var slot in model.inputs)
                    {
                        if (!values.TryGetValue(slot.name, out var value)) throw new InvalidOperationException("Missing original segment input: " + slot.name);
                        worker.SetInput(slot.name, value);
                    }
                    pending = true; hasFence = false; worker.Schedule(); Emit("model_submitted", stage);
                    foreach (var slot in model.outputs)
                    {
                        Tensor copy = null; worker.CopyOutput(slot.name, ref copy);
                        if (copy != null) owned.Add(copy);
                        if (!(copy is Tensor<float> value) || value.backendType != BackendType.GPUCompute)
                            throw new InvalidOperationException("Original boundary must remain GPUCompute.");
                        values[slot.name] = value;
                    }
                    if (retireCompletedCache) { releaseWorkers[stage]?.Invoke(); Emit("original_worker_cache_retired", stage); }
                    var operands = inputs[stage].Select(name => values[name]).ToArray();
                    if (operands.Length != (operations[stage] is IGpuBinaryTensorOperation ? 2 : 1))
                        throw new InvalidOperationException("Original operation arity differs.");
                    Emit("operation_begin", stage, operands[0]);
                    var result = operations[stage] is IGpuBinaryTensorOperation binary
                        ? binary.Execute(operands[0], operands[1]) : operations[stage].Execute(operands[0]);
                    if (result == null || operands.Any(value => ReferenceEquals(result, value)))
                        throw new InvalidOperationException("Original operation must return an independent owned tensor.");
                    owned.Add(result);
                    if (result.backendType != BackendType.GPUCompute) throw new InvalidOperationException("Original operation left GPUCompute.");
                    values[outputs[stage]] = result; Emit("operation_submitted", stage, result);
                    if (retireCompletedCache)
                    {
                        foreach (var name in values.Keys.Where(name => !liveAfter[stage].Contains(name)).ToArray()) values.Remove(name);
                        var retained = new HashSet<Tensor>(values.Values); var retired = 0;
                        foreach (var tensor in owned.Where(value => !retained.Contains(value)).ToArray())
                        { tensor.Dispose(); owned.Remove(tensor); retired++; }
                        Emit("dead_boundaries_retired", stage, retired: retired);
                    }
                    fence = Graphics.CreateGraphicsFence(GraphicsFenceType.AsyncQueueSynchronisation, SynchronisationStageFlags.AllGPUOperations);
                    hasFence = true;
                    do { await Task.Yield(); } while (!fence.passed);
                    pending = false; Emit("fence_passed", stage, result);
                    if (retireCompletedCache) { releaseBackends[stage]?.Invoke(); Emit("original_backend_cache_retired", stage); }
                    token.ThrowIfCancellationRequested();
                }
                var final = values[outputs[outputs.Length - 1]]; owned.Remove(final); return final;
            }
            finally
            {
                try
                {
                    if (pending)
                    {
                        if (!hasFence) fence = Graphics.CreateGraphicsFence(GraphicsFenceType.AsyncQueueSynchronisation, SynchronisationStageFlags.AllGPUOperations);
                        while (!fence.passed) await Task.Yield();
                    }
                }
                finally
                {
                    foreach (var tensor in owned) tensor.Dispose();
                    foreach (var owner in owners) owner.Dispose();
                }
            }
        }
    }
}
