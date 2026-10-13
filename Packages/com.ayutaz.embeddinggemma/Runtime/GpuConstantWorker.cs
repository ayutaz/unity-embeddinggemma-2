using System;
using System.Collections.Generic;
using System.Reflection;
using Unity.InferenceEngine;
using UnityEngine;

namespace EmbeddingGemma
{
    /// <summary>
    /// Owns a Sentis Worker; borrows its Model without changing layers, inputs or weights.
    /// On GPUCompute, uploads large GPU-used constants in bounded 32-bit-word copies before
    /// scheduling. CPU-needed shape constants remain on CPU. Call on the main thread.
    /// Storage inspection is pinned to Sentis 2.6.1 and preserved by the package link.xml.
    /// </summary>
    public sealed class GpuConstantWorker : IDisposable
    {
        public const int DefaultChunkBytes = 8 * 1024 * 1024;
        public const int MaximumChunkBytes = 16 * 1024 * 1024;
        Worker worker;
        public Worker Worker => worker ?? throw new ObjectDisposedException(nameof(GpuConstantWorker));
        public int UploadedConstantCount { get; private set; }
        public int UploadCallCount { get; private set; }
        public int MaximumUploadBytes { get; private set; }
        public int SkippedCpuConstantCount { get; private set; }

        public GpuConstantWorker(Model model, BackendType backend, int chunkBytes = DefaultChunkBytes)
        {
            if (model == null) throw new ArgumentNullException(nameof(model));
            if (backend != BackendType.CPU && backend != BackendType.GPUCompute)
                throw new ArgumentOutOfRangeException(nameof(backend), "Use CPU or GPUCompute explicitly.");
            if (chunkBytes < 4 || chunkBytes > MaximumChunkBytes || chunkBytes % 4 != 0)
                throw new ArgumentOutOfRangeException(nameof(chunkBytes), "Use a word-aligned budget from 4 bytes to 16 MiB.");
            if (backend == BackendType.GPUCompute && !SystemInfo.supportsComputeShaders)
                throw new NotSupportedException("This device cannot run GPUCompute.");
            try
            {
                worker = new Worker(model, backend);
                if (worker.backendType != backend) throw new NotSupportedException("The requested backend was not created.");
                if (backend == BackendType.CPU) return;
                const BindingFlags fields = BindingFlags.Instance | BindingFlags.NonPublic;
                var storageField = typeof(Worker).GetField("m_Storage", fields);
                var fallbackField = typeof(Worker).GetField("m_LayerCPUFallback", fields);
                if (storageField == null || fallbackField == null)
                    throw new NotSupportedException("Pinned Sentis Worker storage inspection is unavailable.");
                var storage = storageField.GetValue(worker);
                if (storage == null || storage.GetType().FullName != "Unity.InferenceEngine.ModelStorage")
                    throw new NotSupportedException("Expected pinned Sentis ModelStorage.");
                var getTensor = storage.GetType().GetMethod("GetTensor", BindingFlags.Public | BindingFlags.Instance, null, new[] { typeof(int) }, null);
                var cpuRequired = fallbackField.GetValue(worker) as HashSet<int>;
                if (getTensor == null || cpuRequired == null)
                    throw new NotSupportedException("Pinned Sentis constant lookup is unavailable.");
                foreach (var constant in model.constants)
                {
                    var tensor = getTensor.Invoke(storage, new object[] { constant.index }) as Tensor;
                    if (tensor == null) throw new InvalidOperationException("Prepared constant tensor is missing.");
                    if ((long)tensor.count * 4 <= chunkBytes) continue;
                    if (cpuRequired.Contains(constant.index)) { SkippedCpuConstantCount++; continue; }
                    Upload(tensor, chunkBytes);
                }
            }
            catch { Dispose(); throw; }
        }

        void Upload(Tensor tensor, int chunkBytes)
        {
            if (!(tensor.dataOnBackend is CPUTensorData cpu) || cpu.array == null)
                throw new NotSupportedException("Expected CPU storage for a newly prepared constant.");
            if (tensor.dataType != DataType.Float && tensor.dataType != DataType.Int &&
                tensor.dataType != DataType.Short && tensor.dataType != DataType.Byte)
                throw new NotSupportedException("Unsupported constant storage type.");
            // Sentis packs Short/Byte into int32 words. Copy raw words, never numeric casts.
            // The native view borrows CPU storage and must not be disposed separately.
            var words = cpu.array.GetNativeArrayHandle<int>();
            if (words.Length < tensor.count) throw new InvalidOperationException("Constant storage is shorter than its packed word count.");
            ComputeTensorData destination = null;
            try
            {
                destination = new ComputeTensorData(tensor.count, false);
                var limit = chunkBytes / 4;
                for (var offset = 0; offset < tensor.count;)
                {
                    var count = Math.Min(limit, tensor.count - offset);
                    destination.buffer.SetData(words, offset, offset, count);
                    UploadCallCount++;
                    MaximumUploadBytes = Math.Max(MaximumUploadBytes, count * 4);
                    offset += count;
                }
                tensor.AdoptTensorData(destination);
                destination = null; // Ownership transferred to the Worker-owned constant Tensor.
                UploadedConstantCount++;
            }
            finally { destination?.Dispose(); }
        }

        public void Dispose() { worker?.Dispose(); worker = null; }
    }
}
