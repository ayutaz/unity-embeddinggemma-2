using System;
using System.Linq;
using Unity.InferenceEngine;

namespace EmbeddingGemma.Tests
{
    /// <summary>Candidate graph only. FP32 products and compensation remain on the requested device.</summary>
    public static class CompensatedProjection
    {
        public static Model Build(float[] weights, int width, int columns, int chunk)
        {
            if (width <= 0 || columns <= 0 || chunk <= 0)
                throw new ArgumentOutOfRangeException(nameof(chunk), "Projection sizes and chunk must be positive.");
            if (weights == null || weights.LongLength != (long)width * columns ||
                weights.Any(value => float.IsNaN(value) || float.IsInfinity(value)))
                throw new ArgumentException("Expected finite row-major [width,columns] FP32 weights.", nameof(weights));
            var graph = new FunctionalGraph();
            var input = graph.AddInput<float>(new TensorShape(1, width), "input");
            var total = Functional.Constant(new TensorShape(1, columns), new float[columns]);
            var correction = Functional.Constant(new TensorShape(1, columns), new float[columns]);
            for (var start = 0; start < width; start += chunk)
            {
                var count = Math.Min(chunk, width - start);
                var block = new float[count * columns];
                Array.Copy(weights, start * columns, block, 0, block.Length);
                var part = Functional.MatMul(input[.., start..(start + count)],
                    Functional.Constant(new TensorShape(count, columns), block));
                var next = total + part;
                correction = correction + Functional.Where(Functional.Abs(total) >= Functional.Abs(part),
                    (total - next) + part, (part - next) + total);
                total = next;
            }
            return graph.Compile(total + correction);
        }
    }
}
