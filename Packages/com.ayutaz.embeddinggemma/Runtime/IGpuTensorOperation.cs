using System;
using Unity.InferenceEngine;

namespace EmbeddingGemma
{
    /// <summary>Borrow a GPU tensor and return an independent owned GPU tensor.</summary>
    public interface IGpuTensorOperation : IDisposable
    {
        BackendType Backend { get; }
        Tensor<float> Execute(Tensor<float> input);
    }
}
