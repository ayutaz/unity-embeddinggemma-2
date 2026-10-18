using System;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.ExceptionServices;
using NUnit.Framework;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class CandidateMediaBundlePolicyTests
    {
        static object Call(string name, params object[] args)
        {
            var type = AppDomain.CurrentDomain.GetAssemblies().Select(a => a.GetType("EmbeddingGemma.Validation.Tests.CandidateMediaBundlePolicy")).FirstOrDefault(t => t != null);
            Assert.That(type, Is.Not.Null, "A separate pinned composed-candidate gate is required.");
            try { return type.GetMethod(name).Invoke(null, args); }
            catch (TargetInvocationException error) { ExceptionDispatchInfo.Capture(error.InnerException).Throw(); throw; }
        }
        [TestCase("fp32", "40aefa5de06f575d31b7582c24314c1ab035da374116b3ce3036de0e919cd647", .999)]
        [TestCase("float16-weights", "8a6a90c5f656a00b024553a0412e5c4d32b4d2c9c7c5b3a0c2c146f557b4e27e", .99)]
        public void EachPrecisionUsesItsExactReviewedDistributionAndOriginalNumericGate(string precision, string sha, double cosine)
        {
            Assert.That(Call("DescriptorSha256", precision), Is.EqualTo(sha));
            Assert.That(Call("MinimumCosine", precision), Is.EqualTo(cosine));
        }
        [TestCase(null)] [TestCase("")] [TestCase("auto")] [TestCase("half")] [TestCase("fp16")]
        public void MissingOrUnknownPrecisionCannotSelectAGenerationOrWeakerGate(string precision)
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => Call("DescriptorSha256", new object[] {precision}));
            Assert.Throws<ArgumentOutOfRangeException>(() => Call("MinimumCosine", new object[] {precision}));
        }
        [TestCase("fp32")] [TestCase("float16-weights")]
        public void AChangedDescriptorIsRejectedBeforeAnyModelLoad(string precision)
        {
            var directory = Path.Combine(Path.GetTempPath(), "embeddinggemma-composed-policy-"+Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(directory);
            try
            {
                File.WriteAllText(Path.Combine(directory,"bundle.json"),"{}");
                Assert.Throws<InvalidDataException>(() => Call("Verify", directory, precision));
                Assert.That(Directory.GetFiles(directory).Length, Is.EqualTo(1));
            }
            finally { Directory.Delete(directory,true); }
        }
        [Test] public void LegacyGenerationAndMemoryGuardsRemainIndependent()
        {
            Assert.That(Call("DescriptorSha256", "fp32"), Is.Not.EqualTo(GpuMediaExecutionPolicy.DescriptorSha256("fp32")));
            Assert.That(GpuMediaExecutionPolicy.DescriptorSha256("fp32"), Is.EqualTo("8363f9a63a95b307fbbb835f3551a41f6fa5a1d21aca3017670e6f678647e955"));
            Assert.That(GpuMediaExecutionPolicy.DescriptorSha256("float16-weights"), Is.EqualTo("1000f91d47c7def916ace1c7affb5fd01884aab9f3cf064ea62cc48e4d057292"));
            Assert.That(GpuMediaExecutionPolicy.MemoryLimitBytes(null), Is.EqualTo(12L*1024*1024*1024));
        }
        [Test] public void CompleteSampleCohortIncludesAllMediaProfilesBeforeTextQueries()
        {
            Call("VerifyCases", "image-square:square-0,image-landscape:landscape-0,image-portrait:portrait-0,audio-1119:silence,audio-839:padded_noise");
        }
        [TestCase(null)] [TestCase("")] [TestCase("image-square:square-0")]
        [TestCase("image-square:square-0,image-landscape:landscape-0,image-portrait:portrait-0,audio-1119:silence")]
        [TestCase("image-square:square-0,image-landscape:landscape-0,image-portrait:portrait-0,audio-1119:silence,audio-839:noise")]
        [TestCase("image-square:square-0,image-landscape:landscape-0,image-portrait:portrait-0,audio-1119:silence,image-square:square-0")]
        public void MissingPartialWrongOrDuplicatedCohortsCannotPassTheCrossModalGate(string cases)
        { Assert.Throws<ArgumentException>(() => Call("VerifyCases", new object[] {cases})); }
    }
}
