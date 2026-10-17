using System;
using System.Globalization;
using System.IO;
using System.Security.Cryptography;

namespace EmbeddingGemma.Validation.Tests
{
    /// <summary>Explicit, pinned precision gates and bounded measurement allowances; no inference fallback.</summary>
    public static class GpuMediaExecutionPolicy
    {
        const string Fp32Descriptor = "8363f9a63a95b307fbbb835f3551a41f6fa5a1d21aca3017670e6f678647e955";
        const string Float16Descriptor = "1000f91d47c7def916ace1c7affb5fd01884aab9f3cf064ea62cc48e4d057292";
        const string Float16Preparation = "68d631e308e0faa8aec0a74e5bc8a38353a0a2c225de04add1123dcc9dea4828";
        static string Precision(string value)
        {
            if (value == null || value == "fp32") return "fp32";
            if (value == "float16-weights") return value;
            throw new ArgumentOutOfRangeException(nameof(value), "Use fp32 or the explicitly pinned float16-weights bundle.");
        }
        public static string DescriptorSha256(string precision) => Precision(precision) == "fp32" ? Fp32Descriptor : Float16Descriptor;
        public static double MinimumCosine(string precision) => Precision(precision) == "fp32" ? .999 : .99;
        public static long MemoryLimitBytes(string gib)
        {
            if (gib == null) return 12L * 1024 * 1024 * 1024;
            if (!int.TryParse(gib, NumberStyles.None, CultureInfo.InvariantCulture, out var value) || value < 12 || value > 16)
                throw new ArgumentOutOfRangeException(nameof(gib), "Use an explicit host allowance between 12 and 16 GiB, after checking available physical memory.");
            return value * 1024L * 1024 * 1024;
        }
        static string Hash(string path)
        { using var stream = File.OpenRead(path); using var hash = SHA256.Create(); return BitConverter.ToString(hash.ComputeHash(stream)).Replace("-", "").ToLowerInvariant(); }
        public static void Verify(string directory, string precision)
        {
            if (Hash(Path.Combine(directory, "bundle.json")) != DescriptorSha256(precision))
                throw new InvalidDataException("Bundle descriptor differs from the pinned precision candidate.");
            if (Precision(precision) == "float16-weights" && Hash(Path.Combine(directory, "float16-preparation.json")) != Float16Preparation)
                throw new InvalidDataException("Float16 source/derivation receipt differs from the independently audited preparation.");
        }
        internal static string CurrentPrecision => Precision(Environment.GetEnvironmentVariable("EG_MEDIA_PRECISION"));
        internal static double CurrentMinimumCosine => MinimumCosine(CurrentPrecision);
        internal static long CurrentMemoryLimitBytes => MemoryLimitBytes(Environment.GetEnvironmentVariable("EG_MEDIA_HOST_LIMIT_GIB"));
        internal static void VerifyCurrent(string directory) => Verify(directory, CurrentPrecision);
    }
}
