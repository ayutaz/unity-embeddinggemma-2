using System;
using System.Linq;
using System.Reflection;
using System.Runtime.ExceptionServices;
using Newtonsoft.Json.Linq;
using NUnit.Framework;

namespace EmbeddingGemma.Tests
{
    public sealed class AudioChainReferenceGuardTests
    {
        static object Invoke(string method, params object[] args)
        {
            var type=typeof(AudioChainReferenceGuardTests).Assembly.GetType("EmbeddingGemma.Tests.AudioChainReferenceGuard");
            Assert.That(type,Is.Not.Null,"Real audio chain acceptance guard is missing");
            try{return type.GetMethod(method).Invoke(null,args);}
            catch(TargetInvocationException error){ExceptionDispatchInfo.Capture(error.InnerException).Throw();throw;}
        }
        static readonly string Source=new string('b',40);
        static JObject Metadata() => new JObject{["source_commit"]=Source,
            ["model_revision"]="914f7f89142e33e77833254d9c9b90c3cef7303b",
            ["input_sha256"]=new JObject{["config.json"]=new string('a',64),["processor_config.json"]=new string('c',64),["model.safetensors"]=new string('d',64)}};
        static JObject Preprocessing() => new JObject{["source_commit"]=Source,
            ["model_revision"]="914f7f89142e33e77833254d9c9b90c3cef7303b",["processor_sha256"]=new string('c',64)};
        [Test] public void MatchingProducerAndSnapshotAreAccepted()
        {Assert.DoesNotThrow(()=>Invoke("ValidateProducers",Metadata(),Metadata(),Preprocessing(),Source));}
        [TestCase("encoder_source")] [TestCase("preprocessor_source")] [TestCase("revision")] [TestCase("weights")]
        public void ChangedProducerOrSnapshotIsRejected(string problem)
        {
            var final=Metadata();var encoder=Metadata();var preprocessing=Preprocessing();
            if(problem=="encoder_source")encoder["source_commit"]=new string('e',40);
            if(problem=="preprocessor_source")preprocessing["source_commit"]=new string('e',40);
            if(problem=="revision")encoder["model_revision"]=new string('e',40);
            if(problem=="weights")encoder["input_sha256"]["model.safetensors"]=new string('e',64);
            Assert.Throws<ArgumentException>(()=>Invoke("ValidateProducers",final,encoder,preprocessing,Source));
        }
        static int[] Mask(){var mask=new int[179200];Array.Fill(mask,1,0,134400);return mask;}
        [Test] public void PaddedFixtureBecomesExactOwnedPcmPrefix()
        {
            var waveform=new float[179200];waveform[0]=0.5f;
            var input=(AudioPcmInput)Invoke("FromPaddedPcm",waveform,Mask(),839);
            waveform[0]=0;Assert.That(input.SourceSampleCount,Is.EqualTo(134400));
            Assert.That(input.ValidFrameCount,Is.EqualTo(839));Assert.That(input.Waveform[0],Is.EqualTo(0.5f));
        }
        [TestCase("mask_hole")] [TestCase("padding")] [TestCase("frame_count")] [TestCase("nonfinite")]
        public void InvalidPcmFixtureIsRejected(string problem)
        {
            var waveform=new float[179200];var mask=Mask();var frames=839;
            if(problem=="mask_hole")mask[17]=0;
            if(problem=="padding")waveform[134400]=0.1f;
            if(problem=="frame_count")frames=838;
            if(problem=="nonfinite")waveform[0]=float.NaN;
            Assert.Throws<ArgumentException>(()=>Invoke("FromPaddedPcm",waveform,mask,frames));
        }
        static JArray Conditions()
        {
            var conditions=new JArray();
            foreach(var precision in new[]{"fp32","float16"})foreach(var backend in new[]{"CPU","GPUCompute"})
            {
                var rows=new JArray();foreach(var id in new[]{"silence","impulse","sine","nyquist","noise","padded_noise"})
                    rows.Add(new JObject{["id"]=id,["success"]=true,["actual_backend"]=backend,["embedding_dimension"]=768,
                        ["norm_squared"]=1.0,["intermediate_devices_matched"]=true});
                conditions.Add(new JObject{["precision"]=precision,["requested_backend"]=backend,["success"]=true,["phase"]="completed",
                    ["cpu_fallback_count"]=0,["readback_flush_count"]=0,["intermediate_readback_count"]=0,["cases"]=rows});
            }
            return conditions;
        }
        [TestCase(false,false,false)] [TestCase(true,false,false)] [TestCase(true,true,true)]
        public void PassingVectorsDoNotEraseUnpassedEncoderGates(bool python,bool sentis,bool expected)
        {
            var result=(JObject)Invoke("Summarize",Conditions(),python,sentis);
            Assert.That(result.Value<bool>("final_vectors_validated"),Is.True);
            Assert.That(result.Value<bool>("end_to_end_validated"),Is.EqualTo(expected));
        }
        [TestCase("missing_case")] [TestCase("gpu_fallback")] [TestCase("actual_cpu")] [TestCase("unnormalized")]
        public void IncompleteOrInvalidExecutionCannotPass(string problem)
        {
            var conditions=Conditions();
            if(problem=="missing_case")((JArray)conditions[0]["cases"]).RemoveAt(0);
            if(problem=="gpu_fallback")conditions[1]["cpu_fallback_count"]=1;
            if(problem=="actual_cpu")conditions[1]["cases"][0]["actual_backend"]="CPU";
            if(problem=="unnormalized")conditions[0]["cases"][0]["norm_squared"]=4;
            var result=(JObject)Invoke("Summarize",conditions,true,true);
            Assert.That(result.Value<bool>("final_vectors_validated"),Is.False);
            Assert.That(result.Value<bool>("end_to_end_validated"),Is.False);
        }
    }
}
