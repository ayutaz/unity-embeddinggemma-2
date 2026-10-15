using System;
using System.Collections.Generic;
using System.Reflection;
using Unity.InferenceEngine;

namespace EmbeddingGemma.Tests
{
    /// <summary>Editor diagnosis only. Downloads live tensors without altering model outputs or layers.</summary>
    public static class WorkerObservation
    {
        public sealed class Snapshot
        {
            public int LayerIndex, TensorIndex;
            public string LayerType;
            public BackendType Backend;
            public TensorShape Shape;
            public float[] Values;
        }
        public sealed class Result
        {
            public float[] Baseline, Final;
            public bool FinalMatchesBaselineBits;
            public readonly List<Snapshot> Snapshots = new List<Snapshot>();
            public readonly List<int> UnavailableTensorIndexes = new List<int>();
        }

        public static Result Observe(Model model, BackendType backend, Tensor<float> input, TensorShape captureShape, int limit)
            => ObserveCore(model, backend, input, captureShape, limit, null);

        public static Result ObserveIndexes(Model model, BackendType backend, Tensor<float> input, HashSet<int> indexes)
        {
            if (indexes == null || indexes.Count == 0) throw new ArgumentException("Request at least one tensor index.", nameof(indexes));
            return ObserveCore(model, backend, input, default, indexes.Count, indexes);
        }

        static Result ObserveCore(Model model, BackendType backend, Tensor<float> input, TensorShape captureShape, int limit, HashSet<int> indexes)
        {
            if (model == null || input == null) throw new ArgumentNullException();
            if (limit < 1) throw new ArgumentOutOfRangeException(nameof(limit));
            var result = new Result();
            using var worker = new Worker(model, backend);
            worker.Schedule(input);
            result.Baseline = ((Tensor<float>)worker.PeekOutput()).DownloadToArray();
            var field = typeof(Worker).GetField("m_Storage", BindingFlags.Instance | BindingFlags.NonPublic);
            if (field == null) throw new NotSupportedException("Pinned Sentis storage inspection is unavailable.");
            var storage = field.GetValue(worker);
            var getTensor = storage.GetType().GetMethod("GetTensor", BindingFlags.Public | BindingFlags.Instance, null, new[] { typeof(int) }, null);
            if (getTensor == null) throw new NotSupportedException("Pinned Sentis tensor lookup is unavailable.");
            var fallbackField = typeof(Worker).GetField("m_LayerCPUFallback", BindingFlags.Instance | BindingFlags.NonPublic);
            var cpuLayers = fallbackField?.GetValue(worker) as HashSet<int>;
            if (cpuLayers == null) throw new NotSupportedException("Pinned Sentis iterator fallback classification is unavailable.");
            var yieldedLayers = new List<int>();
            for (var index = 0; index < model.layers.Count; index++)
                if (!cpuLayers.Contains(model.layers[index].outputs[0])) yieldedLayers.Add(index);
            var iterator = worker.ScheduleIterable(input);
            var yieldedIndex = 0;
            try
            {
                while (iterator.MoveNext())
                {
                    var layerIndex = yieldedLayers[yieldedIndex];
                    var layer = model.layers[layerIndex];
                    if (result.Snapshots.Count < limit)
                    {
                        foreach (var index in layer.outputs)
                        {
                            Tensor tensor;
                            try { tensor = getTensor.Invoke(storage, new object[] { index }) as Tensor; }
                            catch (TargetInvocationException error) when (error.InnerException is KeyNotFoundException)
                            { result.UnavailableTensorIndexes.Add(index); continue; }
                            if (tensor is Tensor<float> value && (indexes != null ? indexes.Contains(index) : value.shape == captureShape) && result.Snapshots.Count < limit)
                                result.Snapshots.Add(new Snapshot { LayerIndex = layerIndex, TensorIndex = index,
                                    LayerType = layer.GetType().Name, Backend = value.backendType, Shape = value.shape, Values = value.DownloadToArray() });
                        }
                    }
                    yieldedIndex++;
                }
            }
            finally { (iterator as IDisposable)?.Dispose(); }
            result.Final = ((Tensor<float>)worker.PeekOutput()).DownloadToArray();
            result.FinalMatchesBaselineBits = result.Baseline.Length == result.Final.Length;
            for (var index = 0; index < result.Final.Length && result.FinalMatchesBaselineBits; index++)
                result.FinalMatchesBaselineBits &= BitConverter.SingleToInt32Bits(result.Final[index]) == BitConverter.SingleToInt32Bits(result.Baseline[index]);
            return result;
        }
    }
}
