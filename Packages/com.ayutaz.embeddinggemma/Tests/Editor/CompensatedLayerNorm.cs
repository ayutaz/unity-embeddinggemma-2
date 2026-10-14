using System;
using System.Linq;
using Unity.InferenceEngine;

namespace EmbeddingGemma.Tests
{
    /// <summary>Editor candidate: centered FP32 variance and compensated block reductions.</summary>
    public static class CompensatedLayerNorm
    {
        public static Model Build(float[] scale, float[] bias, int chunk, int[] shape, float epsilon)
        {
            if (chunk <= 0 || shape == null || shape.Length < 1 || shape.Length > 4 || shape.Any(value => value <= 0)
                || epsilon <= 0 || float.IsNaN(epsilon) || float.IsInfinity(epsilon))
                throw new ArgumentOutOfRangeException(nameof(chunk), "Norm shape, block size and epsilon must be positive.");
            var width = shape[shape.Length - 1];
            if (scale == null || bias == null || scale.Length != width || bias.Length != width
                || scale.Concat(bias).Any(value => float.IsNaN(value) || float.IsInfinity(value)))
                throw new ArgumentException("Expected finite last-axis scale and bias vectors.");
            var graph = new FunctionalGraph(); var input = graph.AddInput<float>(new TensorShape(shape), "input");
            var mean = Sum(input, width, chunk) / width;
            var centered = input - mean;
            var variance = Sum(centered * centered, width, chunk) / width;
            var normalized = centered / Functional.Sqrt(variance + epsilon);
            return graph.Compile(normalized * Functional.Constant((float[])scale.Clone()) + Functional.Constant((float[])bias.Clone()));
        }
        static FunctionalTensor Sum(FunctionalTensor input, int width, int chunk)
        {
            var total = Functional.Constant(0f); var correction = Functional.Constant(0f);
            for (var start = 0; start < width;)
            {
                var count = Math.Min(chunk, width - start);
                var part = Functional.ReduceSum(Functional.Narrow(input, -1, start, count), -1, true);
                var next = total + part;
                correction = correction + Functional.Where(Functional.Abs(total) >= Functional.Abs(part),
                    (total - next) + part, (part - next) + total);
                total = next; start += count;
            }
            return total + correction;
        }
    }
}
