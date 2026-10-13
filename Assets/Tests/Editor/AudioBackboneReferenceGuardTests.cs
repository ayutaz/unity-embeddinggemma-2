using System;
using System.Linq;
using System.Reflection;
using System.Runtime.ExceptionServices;
using Newtonsoft.Json.Linq;
using NUnit.Framework;

namespace EmbeddingGemma.Tests
{
    public sealed class AudioBackboneReferenceGuardTests
    {
        static object Invoke(string name, params object[] arguments)
        {
            var type=typeof(AudioBackboneReferenceGuardTests).Assembly.GetType("EmbeddingGemma.Tests.AudioBackboneReferenceGuard");
            Assert.That(type,Is.Not.Null,"Audio final backbone acceptance guard is missing");
            try{return type.GetMethod(name).Invoke(null,arguments);}
            catch(TargetInvocationException error){ExceptionDispatchInfo.Capture(error.InnerException).Throw();throw;}
        }
        static JObject Reference()
        {
            var rows=new JArray();
            foreach(var id in new[] {"silence","impulse","sine","nyquist","noise","padded_noise"})
                rows.Add(new JObject{["id"]=id,["profile"]=id=="padded_noise"?"partial":"full"});
            return new JObject{["metadata"]=new JObject{["source_commit"]=new string('b',40),
                ["model_revision"]="914f7f89142e33e77833254d9c9b90c3cef7303b",["embedding_dimension"]=768},
                ["profiles"]=new JObject{["full"]=new JObject{["sequence_length"]=1024,["image_feature_count"]=0,["audio_feature_count"]=280,["feature_width"]=512},
                    ["partial"]=new JObject{["sequence_length"]=1024,["image_feature_count"]=0,["audio_feature_count"]=210,["feature_width"]=512}},["cases"]=rows};
        }
        static JObject Validation(JObject reference) => new JObject{["metadata"]=reference["metadata"].DeepClone(),
            ["success"]=true,["stage"]="python_audio_embeddings_only",["python_native_saved_chain_validated"]=true,["case_count"]=6,["profile_count"]=2};
        [Test]
        public void CompleteNativeReferenceIsAcceptedWithoutAcceptingTheEncoder()
        {var reference=Reference();Assert.DoesNotThrow(()=>Invoke("ValidateContext",reference,Validation(reference),new string('b',40)));}
        [TestCase("source")] [TestCase("case")] [TestCase("profile")]
        public void ChangedOrIncompleteContextIsRejected(string problem)
        {
            var reference=Reference();var validation=Validation(reference);
            if(problem=="source") reference["metadata"]["source_commit"]=new string('c',40);
            if(problem=="case") ((JArray)reference["cases"]).RemoveAt(0);
            if(problem=="profile") reference["profiles"]["partial"]["image_feature_count"]=1;
            Assert.Throws<ArgumentException>(()=>Invoke("ValidateContext",reference,validation,new string('b',40)));
        }
        static float[] Vector(double cosine)
        {var result=new float[768];result[0]=(float)cosine;result[1]=(float)Math.Sqrt(1-cosine*cosine);return result;}
        [TestCase(0.9995,false,true)] [TestCase(0.998,false,false)] [TestCase(0.995,true,true)] [TestCase(0.98,true,false)]
        public void FinalVectorUsesTheExistingFp32AndFloat16CosineContract(double cosine,bool half,bool success)
        {
            var result=(JObject)Invoke("CompareVectors",Vector(cosine),Vector(1),half);
            Assert.That(result.Value<bool>("success"),Is.EqualTo(success));
            Assert.That(result.Value<double>("cosine"),Is.EqualTo(cosine).Within(1e-6));
            Assert.That(result.Value<int>("elementwise_mismatches"),Is.GreaterThan(0));
        }
        [Test]
        public void NonFiniteOutputCannotPassCosineComparison()
        {var actual=Vector(1);actual[3]=float.NaN;Assert.Throws<ArgumentException>(()=>Invoke("CompareVectors",actual,Vector(1),false));}
    }
}
