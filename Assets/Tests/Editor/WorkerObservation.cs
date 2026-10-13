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
            var iterator = worker.ScheduleIterable(input);
            var layerIndex = 0;
            try
            {
                while (iterator.MoveNext())
                {
                    var layer = model.layers[layerIndex];
                    if (result.Snapshots.Count < limit)
                    {
                        foreach (var index in layer.outputs)
                        {
                            Tensor tensor;
                            try { tensor = getTensor.Invoke(storage, new object[] { index }) as Tensor; }
                            catch (TargetInvocationException error) when (error.InnerException is KeyNotFoundException)
                            { result.UnavailableTensorIndexes.Add(index); continue; }
                            if (tensor is Tensor<float> value && value.shape == captureShape && result.Snapshots.Count < limit)
                                result.Snapshots.Add(new Snapshot { LayerIndex = layerIndex, TensorIndex = index,
                                    LayerType = layer.GetType().Name, Backend = value.backendType, Values = value.DownloadToArray() });
                        }
                    }
                    layerIndex++;
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
