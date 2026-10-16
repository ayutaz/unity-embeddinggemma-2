using System;
using System.Linq;
using Unity.InferenceEngine;

namespace EmbeddingGemma.Tests
{
    /// <summary>Editor candidate for FP32 dynamic/broadcast attention products.</summary>
    public static class CompensatedMatMul
    {
        public static Model Build(int[] leftShape, int[] rightShape, int chunk)
        {
            if (chunk <= 0 || leftShape == null || rightShape == null ||
                leftShape.Length < 2 || rightShape.Length < 2 || leftShape.Length > 8 || rightShape.Length > 8 ||
                leftShape.Concat(rightShape).Any(value => value <= 0))
                throw new ArgumentOutOfRangeException(nameof(chunk), "Expected positive matrix dimensions and chunk size.");
            var width = leftShape[leftShape.Length - 1];
            if (width != rightShape[rightShape.Length - 2])
                throw new ArgumentException("Contraction dimensions must agree.");
            for (var i = 0; i < Math.Max(leftShape.Length, rightShape.Length) - 2; i++)
            {
                var a = leftShape.Length - 3 - i; var b = rightShape.Length - 3 - i;
                var x = a < 0 ? 1 : leftShape[a]; var y = b < 0 ? 1 : rightShape[b];
                if (x != y && x != 1 && y != 1) throw new ArgumentException("Batch dimensions must broadcast.");
            }
            var graph = new FunctionalGraph();
            var left = graph.AddInput<float>(new TensorShape(leftShape), "left");
            var right = graph.AddInput<float>(new TensorShape(rightShape), "right");
            var total = Functional.Constant(0f); var correction = Functional.Constant(0f);
            for (var start = 0; start < width;)
            {
                var count = Math.Min(chunk, width - start);
                var part = Functional.MatMul(Functional.Narrow(left, -1, start, count), Functional.Narrow(right, -2, start, count));
                var next = total + part;
                correction = correction + Functional.Where(Functional.Abs(total) >= Functional.Abs(part),
                    (total - next) + part, (part - next) + total);
                total = next; start += count;
            }
            return graph.Compile(total + correction);
        }
    }
}
