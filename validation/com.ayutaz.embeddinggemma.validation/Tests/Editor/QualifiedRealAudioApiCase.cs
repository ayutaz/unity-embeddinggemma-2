using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEngine;

namespace EmbeddingGemma.Validation.Tests
{
    // The original PCM, preprocessing graph, accepted GPU encoder and original
    // Document backbone are exercised together. Only the final vector is read back.
    public static class QualifiedRealAudioApiCase
    {
        const BindingFlags Fields=BindingFlags.Instance|BindingFlags.NonPublic;
        sealed class ObservedPreprocessor : IAudioPreprocessor
        {
            readonly IAudioPreprocessor inner;
            public int Calls;
            public AudioPreprocessingResult Last;
            public BackendType Backend=>inner.Backend;
            public ObservedPreprocessor(IAudioPreprocessor inner){this.inner=inner;}
            public AudioPreprocessingResult Process(AudioPcmInput input)
            {
                Calls++;Last=inner.Process(input);
                Assert.That(Last.Features.backendType,Is.EqualTo(BackendType.GPUCompute));
                Assert.That(Last.FrameMask.backendType,Is.EqualTo(BackendType.GPUCompute));return Last;
            }
        }
        sealed class ObservedEncoder : IAudioFeatureEncoder
        {
            readonly IAudioFeatureEncoder inner;
            public int Calls;
            public AudioEncodingResult Last;
            public BackendType Backend=>inner.Backend;
            public AudioEncoderProfile Profile=>inner.Profile;
            public ObservedEncoder(IAudioFeatureEncoder inner){this.inner=inner;}
            public AudioEncodingResult Encode(AudioPreprocessingResult input)
            {
                Calls++;Last=inner.Encode(input);
                Assert.That(Last.SoftTokens.backendType,Is.EqualTo(BackendType.GPUCompute));
                Assert.That(Last.SoftTokens.shape,Is.EqualTo(new TensorShape(Profile.FeatureCount,512)));return Last;
            }
        }
        static string Hash(string path)=>QualifiedImageBackboneSquare.Hash(path);
        static float[] Floats(string path){var bytes=File.ReadAllBytes(path);var values=new float[bytes.Length/4];Buffer.BlockCopy(bytes,0,values,0,bytes.Length);return values;}
        static int[] Ints(string path){var bytes=File.ReadAllBytes(path);var values=new int[bytes.Length/4];Buffer.BlockCopy(bytes,0,values,0,bytes.Length);return values;}
        static void Check(string path,string sha)=>Assert.That(Hash(path),Is.EqualTo(sha),path);
        static void AuditWorker(object owner,Model model,bool backbone)
        {
            var worker=(Worker)owner.GetType().GetField("worker",Fields).GetValue(owner);
            Assert.That(worker.backendType,Is.EqualTo(BackendType.GPUCompute));
            var cpu=(HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallback",Fields).GetValue(worker);
            var flush=(HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallbackShouldFlushGPU",Fields).GetValue(worker);
            var data=model.layers.SelectMany(layer=>layer.outputs).Concat(model.inputs.Where(input=>!backbone||input.name=="audio_features").Select(input=>input.index));
            Assert.That(cpu.Intersect(data),Is.Empty,"No model computation or nonempty feature input may fall back to CPU.");
            Assert.That(flush,Is.Empty);
        }
        public static void Run(PartitionedAudioFeatureEncoder encoder,string signal,JObject report,string reportPath,Action save)
        {
            var root=Path.GetFullPath(Path.Combine(Application.dataPath,"../../k/artifacts/audio-real-chain"));
            var ppRoot=Path.Combine(root,"audio-runtime-preprocessing");
            var backboneRoot=Path.GetFullPath(Path.Combine(Application.dataPath,"../../k/artifacts/audio-final-backbone"));
            Check(Path.Combine(ppRoot,"audio-validation.json"),"4e325f708e8d103a9de358a411380ddbb51cfe953984e9dcb73e291d919630fa");
            var ppAudit=JObject.Parse(File.ReadAllText(Path.Combine(ppRoot,"audio-validation.json")));
            string PcmFile(string name)
            {
                var path=Path.Combine(ppRoot,name);var spec=ppAudit["files"][name];Assert.That(spec,Is.Not.Null);
                Assert.That(new FileInfo(path).Length,Is.EqualTo(spec.Value<long>("bytes")));Check(path,spec.Value<string>("sha256"));return path;
            }
            var waveform=Floats(PcmFile(signal+"-pcm-fp32.bin"));var sampleMask=Ints(PcmFile(signal+"-samples-int32.bin"));
            var count=sampleMask.Count(value=>value==1);
            Assert.That(waveform.Length,Is.EqualTo(179200));Assert.That(sampleMask,Is.EqualTo(Enumerable.Repeat(1,count).Concat(Enumerable.Repeat(0,179200-count)).ToArray()));
            Assert.That(waveform.Skip(count),Is.All.EqualTo(0));
            var input=new AudioPcmInput(waveform.Take(count).ToArray());Assert.That(input.ValidFrameCount,Is.EqualTo(encoder.Profile.ValidFrameCount));
            Check(Path.Combine(backboneRoot,"audio-embedding-reference.json"),"281d12573b431667e8841033db1c6a10dce4382ef98c9a5345c18044a1194001");
            Check(Path.Combine(backboneRoot,"saved-models-receipt.json"),"f79df6aa24fed7f154598a8035bd6cb104dd9d6168db74901be636beb9b9a047");
            var reference=JObject.Parse(File.ReadAllText(Path.Combine(backboneRoot,"audio-embedding-reference.json")));
            var row=((JArray)reference["cases"]).Single(value=>value.Value<string>("id")==signal);
            var profileName=row.Value<string>("profile");Assert.That(profileName,Is.EqualTo("frames-1119-valid-"+input.ValidFrameCount));
            var metadata=reference["metadata"];
            var profile=new MultimodalBackboneProfile(1024,metadata.Value<int>("image_token_id"),metadata.Value<int>("audio_token_id"),0,encoder.Profile.FeatureCount,metadata.Value<int>("vocabulary_size"),metadata.Value<int>("pad_token_id"));
            var tokenizerRoot=Path.Combine(root,"tokenizer");
            foreach(var name in new[]{"tokenizer.json","tokenizer_config.json"})Check(Path.Combine(tokenizerRoot,name),metadata["input_sha256"].Value<string>(name));
            var builder=new MultimodalPromptBuilder(File.ReadAllText(Path.Combine(tokenizerRoot,"tokenizer.json")),File.ReadAllText(Path.Combine(tokenizerRoot,"tokenizer_config.json")),profile);
            var prompt=builder.BuildAudio(input.ValidFrameCount,TextRole.Document);
            Assert.That(prompt.InputIds,Is.EqualTo(row["input_ids"].ToObject<int[]>()));Assert.That(prompt.AttentionMask,Is.EqualTo(row["attention_mask"].ToObject<int[]>()));
            var ppPath=Path.Combine(root,"preprocessor-96d31a4831db40699e7210196e2acead-fp32.sentis");
            Check(ppPath,"da7dd8f65ebb2789360235d657646f47e315f062fc8ccc3b10e3285ecaf6ed8d");
            var backbonePath=Path.Combine(backboneRoot,profileName+"-fp32.sentis");
            Check(backbonePath,input.ValidFrameCount==839?"4f30639749fa7e54ad0067a99d1b28c1f8a85df23726c19adfe6d7baf38f63ad":"484b88de0c8d77b9f36ceab79b2ffd5772444d97a3b2f4571380f159d7c389fa");
            report["scope"]="real_pcm_accepted_gpu_encoder_original_document_final768_api";report["phase"]="load_original_preprocessor_and_backbone";save();
            var ppModel=TextModelFile.Load(ppPath);var backboneModel=TextModelFile.Load(backbonePath);
            using var pp=new AudioPreprocessor(ppModel,BackendType.GPUCompute);
            using var backbone=new MultimodalBackbone(backboneModel,profile,BackendType.GPUCompute);
            AuditWorker(pp,ppModel,false);AuditWorker(backbone,backboneModel,true);
            var pre=new ObservedPreprocessor(pp);var enc=new ObservedEncoder(encoder);
            using var api=new AudioEmbedder(pre,enc,backbone,builder);
            if(signal=="impulse")
            {
                Assert.Throws<ArgumentNullException>(()=>api.Embed(null));
                Assert.Throws<ArgumentOutOfRangeException>(()=>api.Embed(input,(TextRole)99));
                Assert.Throws<ArgumentException>(()=>api.Embed(new AudioPcmInput(new float[161])));
                Assert.That(pre.Calls,Is.Zero);Assert.That(enc.Calls,Is.Zero);
                report["invalid_input_recovery_checks"]=new JArray("null_pcm","invalid_role","different_fixed_profile");
            }
            report["phase"]="execute_pcm_document_api";save();
            var timer=System.Diagnostics.Stopwatch.StartNew();var actual=api.Embed(input,TextRole.Document);
            report["execute_and_final_download_ms"]=timer.Elapsed.TotalMilliseconds;
            Assert.That(pre.Calls,Is.EqualTo(1));Assert.That(enc.Calls,Is.EqualTo(1));
            Assert.Throws<ObjectDisposedException>(()=>{var ignored=pre.Last.Features;});
            Assert.Throws<ObjectDisposedException>(()=>{var ignored=enc.Last.SoftTokens;});
            var expected=row["embedding"].ToObject<float[]>();Assert.That(actual.Length,Is.EqualTo(768));Assert.That(expected.Length,Is.EqualTo(768));
            double dot=0,aa=0,bb=0,maximum=0;var mismatches=0;
            for(var i=0;i<actual.Length;i++)
            {
                Assert.That(float.IsNaN(actual[i])||float.IsInfinity(actual[i]),Is.False);
                dot+=(double)actual[i]*expected[i];aa+=(double)actual[i]*actual[i];bb+=(double)expected[i]*expected[i];
                var diff=Math.Abs((double)actual[i]-expected[i]);maximum=Math.Max(maximum,diff);if(diff>1e-5+1e-4*Math.Abs((double)expected[i]))mismatches++;
            }
            var cosine=dot/Math.Sqrt(aa*bb);report["cosine"]=cosine;report["norm_squared"]=aa;
            report["elementwise_diagnostic_mismatches"]=mismatches;report["maximum_absolute_error"]=maximum;
            var capture=Path.ChangeExtension(reportPath,".bin");Assert.That(File.Exists(capture),Is.False);
            var bytes=new byte[actual.Length*4];Buffer.BlockCopy(actual,0,bytes,0,bytes.Length);File.WriteAllBytes(capture,bytes);
            report["capture_sha256"]=Hash(capture);report["profile"]=profileName;report["role"]="document";
            report["source_sample_count"]=count;report["valid_frame_count"]=input.ValidFrameCount;
            report["actual_pcm_preprocessing_validated"]=true;report["owned_intermediates_disposed"]=true;
            report["cpu_computation_or_input_count"]=0;report["readback_flush_count"]=0;report["intermediate_readback_count"]=0;
            save();Assert.That(cosine,Is.GreaterThanOrEqualTo(0.999));Assert.That(Math.Abs(aa-1),Is.LessThanOrEqualTo(0.002));
            api.Dispose();Assert.Throws<ObjectDisposedException>(()=>api.Embed(input));
            report["api_dispose_rejected_reuse"]=true;report["success"]=true;report["phase"]="completed";save();
        }
    }
}
