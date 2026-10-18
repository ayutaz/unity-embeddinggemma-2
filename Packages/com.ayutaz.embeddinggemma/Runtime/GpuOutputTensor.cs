using System;
using Unity.InferenceEngine;
namespace EmbeddingGemma
{
    // Write-only operation outputs must not allocate/initialize/upload CPU storage.
    internal static class GpuOutputTensor
    {
        public static Tensor<float> Allocate(TensorShape shape)
        {
            if (shape.length <= 0) throw new ArgumentException("A nonempty write-only output shape is required.", nameof(shape));
            // The caller pins fresh GPU storage and its shader writes every element.
            // A shape-only Tensor constructor would allocate/zero CPU memory and
            // Pin would upload that unused data before the shader overwrites it.
            return new Tensor<float>(shape, (ITensorData)null);
        }
    }
}
