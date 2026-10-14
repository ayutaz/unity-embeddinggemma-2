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
            return BuildForShape(weights, columns, chunk, new[] { 1, width });
        }

        public static Model BuildForShape(float[] weights, int columns, int chunk, int[] shape)
        {
            return BuildClipped(weights, columns, chunk, shape,
                new[] { float.NegativeInfinity, float.PositiveInfinity, float.NegativeInfinity, float.PositiveInfinity });
        }

        public static Model BuildClipped(float[] weights, int columns, int chunk, int[] shape, float[] bounds)
        {
            if (bounds == null || bounds.Length != 4 || bounds.Any(float.IsNaN) || bounds[0] > bounds[1] || bounds[2] > bounds[3])
                throw new ArgumentException("Expected ordered input and output clip bounds.", nameof(bounds));
            if (shape == null || (shape.Length != 2 && shape.Length != 3) || shape.Any(value => value <= 0)
                || columns <= 0 || chunk <= 0)
                throw new ArgumentOutOfRangeException(nameof(chunk), "Projection sizes and chunk must be positive.");
            var width = shape[shape.Length - 1];
            if (weights == null || weights.LongLength != (long)width * columns ||
                weights.Any(value => float.IsNaN(value) || float.IsInfinity(value)))
                throw new ArgumentException("Expected finite row-major [width,columns] FP32 weights.", nameof(weights));
            var graph = new FunctionalGraph();
            var input = graph.AddInput<float>(new TensorShape(shape), "input");
            var working = Functional.Clamp(input, bounds[0], bounds[1]);
            var outputShape = (int[])shape.Clone(); outputShape[outputShape.Length - 1] = columns;
            var outputCount = outputShape.Aggregate(1L, (count, size) => count * size);
            if (outputCount > int.MaxValue) throw new ArgumentOutOfRangeException(nameof(shape), "Output size exceeds the tensor budget.");
            var total = Functional.Constant(new TensorShape(outputShape), new float[(int)outputCount]);
            var correction = Functional.Constant(new TensorShape(outputShape), new float[(int)outputCount]);
            for (var start = 0; start < width;)
            {
                var count = Math.Min(chunk, width - start);
                var block = new float[count * columns];
                Array.Copy(weights, start * columns, block, 0, block.Length);
                var data = shape.Length == 2 ? working[.., start..(start + count)] : working[.., .., start..(start + count)];
                var part = Functional.MatMul(data,
                    Functional.Constant(new TensorShape(count, columns), block));
                var next = total + part;
                correction = correction + Functional.Where(Functional.Abs(total) >= Functional.Abs(part),
                    (total - next) + part, (part - next) + total);
                total = next;
                start += count;
            }
            return graph.Compile(Functional.Clamp(total + correction, bounds[2], bounds[3]));
        }
    }
}
