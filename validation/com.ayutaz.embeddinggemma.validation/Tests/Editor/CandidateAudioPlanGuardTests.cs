using System;
using System.IO;
using System.Linq;
using System.Reflection;
using Newtonsoft.Json.Linq;
using NUnit.Framework;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class CandidateAudioPlanGuardTests
    {
        static (JObject plan,JObject receipt) Fixture()
        {
            const string source="b6ede71cbf4170ae10a61344455dbab56a8231cf";
            var plan=new JObject
            {
                ["producer"]=source,["profile"]="frames-1119-valid-1119",
                ["source_graph_sha256"]="525871a8dbccbbd75b5c521ab959588323938e58fac361d69035eb80a61d2095",
                ["seed_source"]="0cee4c7b77c89410895d52e4420816a5e66a8845",["seed_run"]=39117103059L,
                ["original_initializer_bytes_preserved"]=true,["source_initializer_bytes_preserved"]=true,
                ["original_initializer_count"]=587,["original_references_changed"]=false,
                ["numeric_acceptance"]=false,["gpu_validated"]=false,["generation_qualified"]=false,["original_golden_changed"]=false,
                ["operations"]=new JArray(Enumerable.Range(0,267).Select(_=>new JObject())),
                ["segments"]=new JArray(Enumerable.Range(0,268).Select(_=>new JObject()))
            };
            var receipt=new JObject
            {
                ["success"]=true,["source"]=source,["case_id"]="silence",
                ["operation_count"]=267,["segment_count"]=268,
                ["plan_sha256"]="d8440b4ffc7f8b37e11bee1da1da0b7cadb44e708a1790aab247e1fca40577f9",
                ["seed_source"]=plan["seed_source"].DeepClone(),["seed_run"]=39117103059L,
                ["numeric_acceptance"]=false,["gpu_validated"]=false,["generation_qualified"]=false,["original_golden_changed"]=false
            };
            return (plan,receipt);
        }
        static MethodInfo Required()
        {
            var type=typeof(CandidateAudioPlanGuardTests).Assembly.GetType("EmbeddingGemma.Validation.Tests.CandidateAudioPlanGuard");
            Assert.That(type,Is.Not.Null,"Candidate provenance guard is missing.");
            var method=type.GetMethod("Check",BindingFlags.Public|BindingFlags.Static);
            Assert.That(method,Is.Not.Null);return method;
        }
        [Test]
        public void MatchingMetadataCanProceedToSeparatePinnedByteAudit()
        {
            var (plan,receipt)=Fixture();
            Required().Invoke(null,new object[]{plan,receipt,"frames-1119-valid-1119"});
        }
        [TestCase("producer")]
        [TestCase("seed")]
        [TestCase("graph")]
        [TestCase("receipt-plan-pin")]
        [TestCase("profile")]
        [TestCase("coefficients")]
        [TestCase("segments")]
        [TestCase("receipt")]
        [TestCase("qualification")]
        [TestCase("missing-qualification")]
        [TestCase("golden")]
        [TestCase("unknown-profile")]
        public void InconsistentCandidateMetadataFailsBeforeAnyModelLoad(string problem)
        {
            var (plan,receipt)=Fixture();var profile="frames-1119-valid-1119";
            if(problem=="producer")plan["producer"]=new string('a',40);
            if(problem=="seed")plan["seed_run"]=1;
            if(problem=="graph")plan["source_graph_sha256"]=new string('0',64);
            if(problem=="receipt-plan-pin")receipt["plan_sha256"]=new string('0',64);
            if(problem=="profile")receipt["case_id"]="padded_noise";
            if(problem=="coefficients")plan["original_initializer_bytes_preserved"]=false;
            if(problem=="segments")((JArray)plan["segments"]).RemoveAt(0);
            if(problem=="receipt")receipt["success"]=false;
            if(problem=="qualification")receipt["gpu_validated"]=true;
            if(problem=="missing-qualification")receipt.Remove("gpu_validated");
            if(problem=="golden")plan["original_references_changed"]=true;
            if(problem=="unknown-profile")profile="unknown";
            var method=Required();var error=Assert.Throws<TargetInvocationException>(()=>method.Invoke(null,new object[]{plan,receipt,profile}));
            Assert.That(error.InnerException,Is.InstanceOf<InvalidDataException>());
        }
    }
}
