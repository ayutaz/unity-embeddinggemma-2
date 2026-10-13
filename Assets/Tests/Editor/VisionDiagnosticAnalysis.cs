using System;

namespace EmbeddingGemma.Tests
{
    internal struct VisionDiagnosticMetrics
    {
        internal double MaximumAbsoluteError;
        internal double MaximumToleranceRatio;
        internal int Mismatches;
        internal bool Passed => Mismatches == 0;
    }

    internal struct VisionDiagnosticContext
    {
        internal VisionDiagnosticMetrics Original;
        internal VisionDiagnosticMetrics NativeReference;
        internal bool TraceMatchesOriginal;
        internal bool CanLocalize;
    }

    internal static class VisionDiagnosticAnalysis
    {
        internal static VisionDiagnosticContext AnalyzePinnedReference(float[] expected, float[] original,
            float[] traced, float[] native, bool pythonContextMatches)
        {
            var nativeReference = Compare(expected, native);
            var context = Analyze(expected, original, traced, pythonContextMatches && nativeReference.Passed);
            context.NativeReference = nativeReference;
            return context;
        }

        internal static VisionDiagnosticMetrics Compare(float[] expected, float[] actual)
        {
            if (expected == null || actual == null) throw new ArgumentNullException();
            if (expected.Length != actual.Length || expected.Length == 0) throw new ArgumentException("Expected equal nonempty stage vectors.");
            var result = new VisionDiagnosticMetrics();
            for (var index = 0; index < expected.Length; index++)
            {
                if (!Finite(expected[index]) || !Finite(actual[index])) throw new ArgumentException("Stage vectors must be finite.");
                var difference = Math.Abs((double)actual[index] - expected[index]);
                var bound = 1e-5 + 1e-4 * Math.Abs((double)expected[index]);
                result.MaximumAbsoluteError = Math.Max(result.MaximumAbsoluteError, difference);
                result.MaximumToleranceRatio = Math.Max(result.MaximumToleranceRatio, difference / bound);
                if (difference > bound) result.Mismatches++;
            }
            return result;
        }
        internal static VisionDiagnosticContext Analyze(float[] expected, float[] original, float[] traced,
            bool pythonContextMatches)
        {
            var comparison = Compare(expected, original);
            var same = traced != null && traced.Length == original.Length;
            if (same)
                for (var index = 0; index < original.Length; index++)
                    if (!Finite(traced[index]) || traced[index] != original[index]) { same = false; break; }
            return new VisionDiagnosticContext { Original = comparison, TraceMatchesOriginal = same,
                CanLocalize = pythonContextMatches && same && !comparison.Passed };
        }

        static bool Finite(float value) => !float.IsNaN(value) && !float.IsInfinity(value);
    }
}
