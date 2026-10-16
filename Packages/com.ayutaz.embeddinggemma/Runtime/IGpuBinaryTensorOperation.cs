using Unity.InferenceEngine;

namespace EmbeddingGemma
{
    /// <summary>Borrow two GPU tensors and return an independent owned GPU tensor.</summary>
    public interface IGpuBinaryTensorOperation : IGpuTensorOperation
    {
        Tensor<float> Execute(Tensor<float> left,Tensor<float> right);
    }
}
