using System;
using System.Linq;
using System.Text.RegularExpressions;
using Newtonsoft.Json.Linq;

namespace EmbeddingGemma.Tests
{
    public static class AudioChainReferenceGuard
    {
        static readonly string[] Signals={"silence","impulse","sine","nyquist","noise","padded_noise"};
        public static JObject DescribeValues(float[] values)
        {
            if(values==null)throw new ArgumentNullException(nameof(values));
            var finite=values.Where(value=>!float.IsNaN(value)&&!float.IsInfinity(value)).ToArray();
            return new JObject{["length"]=values.Length,["finite_count"]=finite.Length,
                ["nan_count"]=values.Count(float.IsNaN),["infinity_count"]=values.Count(float.IsInfinity),
                ["finite_sum_squares"]=finite.Sum(value=>(double)value*value),
                ["maximum_absolute_finite_value"]=finite.Length==0?0:finite.Max(value=>Math.Abs((double)value))};
        }
        public static void ValidateProducers(JObject final,JObject encoder,JObject preprocessing,string source)
        {
            const string revision="914f7f89142e33e77833254d9c9b90c3cef7303b";
            if(source==null || !Regex.IsMatch(source,"^[0-9a-f]{40}$") ||
                new[]{final,encoder,preprocessing}.Any(meta=>meta==null || meta.Value<string>("source_commit")!=source || meta.Value<string>("model_revision")!=revision))
                throw new ArgumentException("Audio chain producer/revision mismatch.");
            foreach(var key in new[]{"config.json","processor_config.json","model.safetensors"})
            {
                var left=final["input_sha256"]?.Value<string>(key);var right=encoder["input_sha256"]?.Value<string>(key);
                if(left==null || !Regex.IsMatch(left,"^[0-9a-f]{64}$") || left!=right)
                    throw new ArgumentException("Audio chain snapshot mismatch: "+key);
            }
            if(preprocessing.Value<string>("processor_sha256")!=final["input_sha256"].Value<string>("processor_config.json"))
                throw new ArgumentException("Audio chain processor hash mismatch.");
        }
        public static AudioPcmInput FromPaddedPcm(float[] waveform,int[] mask,int validFrames)
        {
            if(waveform==null || mask==null || waveform.Length!=179200 || mask.Length!=179200)
                throw new ArgumentException("Expected fixed PCM and sample-mask fixtures.");
            var count=Array.FindIndex(mask,x=>x!=1);if(count<0)count=mask.Length;
            for(var index=count;index<mask.Length;index++)
                if(mask[index]!=0 || waveform[index]!=0)throw new ArgumentException("PCM mask must be a contiguous prefix with zero sample padding.");
            var input=new AudioPcmInput(waveform.Take(count).ToArray());
            if(input.ValidFrameCount!=validFrames)throw new ArgumentException("PCM and encoder frame prefixes differ.");
            return input;
        }
        public static JObject Summarize(JArray conditions,bool pythonOnnxEncoderPassed,bool sentisEncoderPassed)
        {
            var combinations=new[]{"fp32/CPU","fp32/GPUCompute","float16/CPU","float16/GPUCompute"};
            var passed=conditions!=null && conditions.Count==4 &&
                conditions.Select(row=>row.Value<string>("precision")+"/"+row.Value<string>("requested_backend")).OrderBy(x=>x).SequenceEqual(combinations.OrderBy(x=>x));
            if(passed)foreach(var condition in conditions)
            {
                var rows=condition["cases"] as JArray;var backend=condition.Value<string>("requested_backend");
                passed &= condition.Value<bool?>("success")==true && condition.Value<string>("phase")=="completed" &&
                    condition.Value<int?>("cpu_fallback_count")==0 && condition.Value<int?>("readback_flush_count")==0 &&
                    condition.Value<int?>("intermediate_readback_count")==0 && rows!=null && rows.Count==6 &&
                    rows.Select(row=>row.Value<string>("id")).OrderBy(x=>x).SequenceEqual(Signals.OrderBy(x=>x));
                if(rows==null)continue;
                foreach(var row in rows)
                {
                    var norm=row.Value<double?>("norm_squared");
                    passed &= row.Value<bool?>("success")==true && row.Value<string>("actual_backend")==backend &&
                        row.Value<int?>("embedding_dimension")==768 && row.Value<bool?>("intermediate_devices_matched")==true &&
                        norm.HasValue && !double.IsNaN(norm.Value) && !double.IsInfinity(norm.Value) && Math.Abs(norm.Value-1)<=0.002;
                }
            }
            var complete=passed && pythonOnnxEncoderPassed && sentisEncoderPassed;
            return new JObject{["success"]=complete,["final_vectors_validated"]=passed,
                ["encoder_python_gate_passed"]=pythonOnnxEncoderPassed,["encoder_sentis_gate_passed"]=sentisEncoderPassed,["end_to_end_validated"]=complete};
        }
    }
}
