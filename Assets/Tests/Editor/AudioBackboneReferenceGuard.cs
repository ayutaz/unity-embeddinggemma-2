using System;
using System.Linq;
using Newtonsoft.Json.Linq;

namespace EmbeddingGemma.Tests
{
    public static class AudioBackboneReferenceGuard
    {
        public static void ValidateContext(JObject reference, JObject validation, string source)
        {
            var metadata=reference["metadata"];
            if(metadata==null || !JToken.DeepEquals(metadata,validation["metadata"]) ||
                metadata.Value<string>("source_commit")!=source ||
                metadata.Value<string>("model_revision")!="914f7f89142e33e77833254d9c9b90c3cef7303b" ||
                metadata.Value<int>("embedding_dimension")!=768 || validation.Value<bool>("success")!=true ||
                validation.Value<string>("stage")!="python_audio_embeddings_only" ||
                validation.Value<bool>("python_native_saved_chain_validated")!=true ||
                validation.Value<int>("case_count")!=6 || validation.Value<int>("profile_count")!=2)
                throw new ArgumentException("Native final audio source/completion metadata mismatch.");
            var cases=reference["cases"] as JArray; var profiles=reference["profiles"] as JObject;
            var signals=new[] {"silence","impulse","sine","nyquist","noise","padded_noise"};
            if(cases==null || cases.Count!=6 || !cases.Select(row=>row.Value<string>("id")).OrderBy(x=>x).SequenceEqual(signals.OrderBy(x=>x)) ||
                profiles==null || profiles.Count!=2 || cases.Any(row=>profiles[row.Value<string>("profile")]==null))
                throw new ArgumentException("Native final audio requires six distinct signals and two profiles.");
            foreach(var profile in profiles)
                if(profile.Value.Value<int>("sequence_length")!=1024 || profile.Value.Value<int>("image_feature_count")!=0 ||
                    profile.Value.Value<int>("feature_width")!=512 ||
                    (profile.Value.Value<int>("audio_feature_count")!=280 && profile.Value.Value<int>("audio_feature_count")!=210))
                    throw new ArgumentException("Native final audio profile mismatch.");
        }
        public static JObject CompareVectors(float[] actual,float[] expected,bool half)
        {
            if(actual==null || expected==null || actual.Length!=768 || expected.Length!=768 ||
                actual.Concat(expected).Any(x=>float.IsNaN(x)||float.IsInfinity(x)))
                throw new ArgumentException("Final audio vector shape/nonfinite mismatch.");
            double dot=0,left=0,right=0,maximum=0,ratio=0;int mismatches=0;
            for(int index=0;index<768;index++)
            {
                var diff=Math.Abs((double)actual[index]-expected[index]);var bound=1e-5+1e-4*Math.Abs((double)expected[index]);
                maximum=Math.Max(maximum,diff);ratio=Math.Max(ratio,diff/bound);if(diff>bound)mismatches++;
                dot+=(double)actual[index]*expected[index];left+=(double)actual[index]*actual[index];right+=(double)expected[index]*expected[index];
            }
            if(left*right<=0) throw new ArgumentException("Final audio vector norm is zero.");
            var cosine=dot/Math.Sqrt(left*right);
            return new JObject{["cosine"]=cosine,["success"]=cosine>=(half?0.99:0.999),
                ["maximum_absolute_error"]=maximum,["maximum_tolerance_ratio"]=ratio,["elementwise_mismatches"]=mismatches};
        }
    }
}
