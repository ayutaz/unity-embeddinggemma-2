using System;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.ExceptionServices;
using NUnit.Framework;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class GpuMediaExecutionPolicyTests
    {
        static object Call(string name, params object[] args)
        {
            var type = AppDomain.CurrentDomain.GetAssemblies().Select(a => a.GetType("EmbeddingGemma.Validation.Tests.GpuMediaExecutionPolicy")).FirstOrDefault(t => t != null);
            Assert.That(type, Is.Not.Null, "Pinned FP32/Float16 execution policy is missing.");
            try { return type.GetMethod(name).Invoke(null, args); }
            catch (TargetInvocationException error) { ExceptionDispatchInfo.Capture(error.InnerException).Throw(); throw; }
        }
        [TestCase(null)] [TestCase("fp32")]
        public void OriginalFp32KeepsItsExactDescriptorAndNumericGate(string precision)
        {
            Assert.That(Call("DescriptorSha256", new object[] {precision}), Is.EqualTo("8363f9a63a95b307fbbb835f3551a41f6fa5a1d21aca3017670e6f678647e955"));
            Assert.That(Call("MinimumCosine", new object[] {precision}), Is.EqualTo(.999));
        }
        [Test] public void Float16UsesAnExplicitFixedDescriptorAndItsEstablishedNumericGate()
        {
            var hash = (string)Call("DescriptorSha256", "float16-weights");
            Assert.That(hash, Does.Match("\\A[a-f0-9]{64}\\z"));
            Assert.That(hash, Is.Not.EqualTo(new string('0', 64)).And.Not.EqualTo(Call("DescriptorSha256", "fp32")));
            Assert.That(Call("MinimumCosine", "float16-weights"), Is.EqualTo(.99));
        }
        [TestCase("auto")] [TestCase("half")] [TestCase("fp16")] [TestCase("")]
        public void UnknownPrecisionCannotSilentlyChooseAWeakerGate(string precision)
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => Call("MinimumCosine", precision));
            Assert.Throws<ArgumentOutOfRangeException>(() => Call("DescriptorSha256", precision));
        }
        [TestCase(null, 12)] [TestCase("12", 12)] [TestCase("16", 16)]
        public void HostBudgetDefaultsToTwelveAndAllowsOnlyExplicitBoundedValues(string value, int expected)
        { Assert.That(Call("MemoryLimitBytes", new object[] {value}), Is.EqualTo(expected * 1024L * 1024 * 1024)); }
        [TestCase("0")] [TestCase("11")] [TestCase("17")] [TestCase("unlimited")] [TestCase("12.5")]
        public void InvalidHostBudgetCannotDisableTheGuard(string value)
        { Assert.Throws<ArgumentOutOfRangeException>(() => Call("MemoryLimitBytes", value)); }
        [TestCase("fp32")] [TestCase("float16-weights")]
        public void ChangedDescriptorIsRejectedBeforeAnyModelCanBeLoaded(string precision)
        {
            var directory = Path.Combine(Path.GetTempPath(), "embeddinggemma-precision-policy-" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(directory);
            try
            {
                File.WriteAllText(Path.Combine(directory, "bundle.json"), "{}");
                Assert.Throws<InvalidDataException>(() => Call("Verify", directory, precision));
            }
            finally { Directory.Delete(directory, true); }
        }
    }
}
