using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.ExceptionServices;
using NUnit.Framework;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class MediaRepeatedInputPolicyTests
    {
        static Type Find(string name)
        {
            var type=AppDomain.CurrentDomain.GetAssemblies().Select(a=>a.GetType("EmbeddingGemma.Validation.Tests."+name)).FirstOrDefault(t=>t!=null);
            Assert.That(type,Is.Not.Null,"The repeated-input measurement needs an explicit source policy.");return type;
        }
        static object Invoke(Type type,object target,string name,params object[] args)
        {
            var method=type.GetMethod(name);Assert.That(method,Is.Not.Null,"The measurement entry must expose its selected policy.");
            try{return method.Invoke(target,args);}
            catch(TargetInvocationException error){ExceptionDispatchInfo.Capture(error.InnerException).Throw();throw;}
        }
        static object Create(bool candidate,string precision,string module)=>Invoke(Find("MediaRepeatedInputPolicy"),null,candidate?"CreateCandidate":"CreateRetained",precision,module);
        static T Value<T>(object policy,string name)=>(T)policy.GetType().GetProperty(name).GetValue(policy);
        public static IEnumerable<TestCaseData> CandidateCases()
        {
            foreach(var precision in new[]{"fp32","float16-weights"})
                foreach(var module in new[]{"image-square","image-landscape","image-portrait","audio-1119","audio-839"})
                    yield return new TestCaseData(precision,module);
        }
        [TestCaseSource(nameof(CandidateCases))]
        public void CandidateCoversEveryMediaProfileWithExactDescriptorAndOriginalThreshold(string precision,string module)
        {
            var policy=Create(true,precision,module);
            Assert.That(Value<string>(policy,"Precision"),Is.EqualTo(precision));Assert.That(Value<string>(policy,"Module"),Is.EqualTo(module));
            Assert.That(Value<bool>(policy,"IsCandidate"),Is.True);
            Assert.That(Value<string>(policy,"ExpectedDescriptorSha256"),Is.EqualTo(CandidateMediaBundlePolicy.DescriptorSha256(precision)));
            Assert.That(Value<double>(policy,"MinimumCosine"),Is.EqualTo(precision=="fp32"?.999:.99));
            Assert.That(policy.GetType().GetProperty("Backend").GetValue(policy).ToString(),Is.EqualTo("GPUCompute"));
        }
        [TestCase(null)] [TestCase("")] [TestCase("auto")] [TestCase("half")] [TestCase("fp16")]
        public void CandidateNeverDefaultsAMissingOrUnknownPrecision(string precision)
        {Assert.Throws<ArgumentOutOfRangeException>(()=>Create(true,precision,"audio-1119"));}
        [TestCase(null)] [TestCase("")] [TestCase("image-wide")] [TestCase("audio-0840")] [TestCase("text")]
        public void UnknownOrNonMediaModulesCannotEnterTheFixedInputMeasurement(string module)
        {Assert.Throws<ArgumentException>(()=>Create(true,"fp32",module));}
        [TestCase("fp32")] [TestCase("float16-weights")]
        public void RetainedPolicyKeepsItsOwnDescriptorAndThreshold(string precision)
        {
            var policy=Create(false,precision,"image-square");Assert.That(Value<bool>(policy,"IsCandidate"),Is.False);
            Assert.That(Value<string>(policy,"ExpectedDescriptorSha256"),Is.EqualTo(GpuMediaExecutionPolicy.DescriptorSha256(precision)));
            Assert.That(Value<double>(policy,"MinimumCosine"),Is.EqualTo(GpuMediaExecutionPolicy.MinimumCosine(precision)));
        }
        [TestCase(true,"fp32")] [TestCase(true,"float16-weights")]
        [TestCase(false,"fp32")] [TestCase(false,"float16-weights")]
        public void ChangedDescriptorFailsBeforeModelLoading(bool candidate,string precision)
        {
            var policy=Create(candidate,precision,"audio-1119");var directory=Path.Combine(Path.GetTempPath(),"media-repeat-policy-"+Guid.NewGuid().ToString("N"));Directory.CreateDirectory(directory);
            try{File.WriteAllText(Path.Combine(directory,"bundle.json"),"{}");Assert.Throws<InvalidDataException>(()=>Invoke(policy.GetType(),policy,"Verify",directory));Assert.That(Directory.GetFiles(directory).Length,Is.EqualTo(1));}
            finally{Directory.Delete(directory,true);}
        }
        [Test] public void CallerCannotReplaceTheFourIdenticalInputRounds()
        {
            var policy=Create(true,"fp32","audio-1119");var rounds=Value<string[]>(policy,"Rounds");
            Assert.That(rounds,Is.EqualTo(new[]{"first","warm","repeat-1","repeat-2"}));rounds[0]="changed";
            Assert.That(Value<string[]>(policy,"Rounds")[0],Is.EqualTo("first"));
        }
        [Test] public void CapturedPrecisionDoesNotChangeWithLaterEnvironmentEdits()
        {
            var policy=Create(true,"fp32","image-square");var old=Environment.GetEnvironmentVariable("EG_MEDIA_PRECISION");
            try{Environment.SetEnvironmentVariable("EG_MEDIA_PRECISION","float16-weights");Assert.That(Value<string>(policy,"Precision"),Is.EqualTo("fp32"));Assert.That(Value<double>(policy,"MinimumCosine"),Is.EqualTo(.999));}
            finally{Environment.SetEnvironmentVariable("EG_MEDIA_PRECISION",old);}
        }
        [Test] public void RetainedDefaultRemainsExplicitFp32()
        {Assert.That(Value<string>(Create(false,null,"audio-839"),"Precision"),Is.EqualTo("fp32"));}
        [TestCase("")] [TestCase("float16")] [TestCase("auto")]
        public void RetainedPolicyStillRejectsUnknownPrecision(string precision)
        {Assert.Throws<ArgumentOutOfRangeException>(()=>Create(false,precision,"audio-839"));}
        [TestCase(false,"MediaRepeatedInputPerformanceTests")]
        [TestCase(true,"CandidateMediaRepeatedInputPerformanceTests")]
        public void EachRealMeasurementEntrySelectsItsOwnPolicy(bool candidate,string name)
        {
            var precision=Environment.GetEnvironmentVariable("EG_MEDIA_PRECISION");var module=Environment.GetEnvironmentVariable("EG_MEDIA_BUNDLE_MODULE");
            try
            {
                Environment.SetEnvironmentVariable("EG_MEDIA_PRECISION","fp32");Environment.SetEnvironmentVariable("EG_MEDIA_BUNDLE_MODULE","audio-839");
                var policy=Invoke(Find(name),null,"CreatePolicy");Assert.That(Value<bool>(policy,"IsCandidate"),Is.EqualTo(candidate));
                Assert.That(Value<string>(policy,"Module"),Is.EqualTo("audio-839"));
            }
            finally{Environment.SetEnvironmentVariable("EG_MEDIA_PRECISION",precision);Environment.SetEnvironmentVariable("EG_MEDIA_BUNDLE_MODULE",module);}
        }
    }
}
