using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.ExceptionServices;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEditor;
using UnityEngine;

namespace EmbeddingGemma.Tests
{
    // Candidate PCM -> real preprocessing -> real encoder -> real backbone -> public API.
    // Final-vector acceptance never erases the independently failed encoder gates.
    public sealed class AudioRealChainReferenceTests
    {
        const string Producer="4edbe17a8105233c5aaa1514faa2862b08a4f7e0";
        string root,backboneRoot,resultPath;JObject reference,encoder,encoderAudit,preprocessingAudit,savedReceipt,report;
        JArray conditions;Type encoderType,profileType,embedderType;
        readonly Dictionary<string,Lazy<string[]>> saved=new Dictionary<string,Lazy<string[]>>();
        readonly Dictionary<string,MultimodalPromptBuilder> prompts=new Dictionary<string,MultimodalPromptBuilder>();
        readonly HashSet<string> preprocessingChecked=new HashSet<string>();
        readonly JArray preprocessingCases=new JArray();
        [OneTimeSetUp]
        public void AuditImmutableChainBundle()
        {
            root=Path.GetFullPath(Path.Combine(Application.dataPath,"../artifacts/audio-real-chain"));
            backboneRoot=Path.GetFullPath(Path.Combine(Application.dataPath,"../artifacts/audio-final-backbone"));
            resultPath=Path.Combine(root,"../audio-real-chain-results/results.json");conditions=new JArray();
            report=new JObject{["success"]=false,["scope"]="real_document_pcm_candidate_chain_final_vectors",
                ["end_to_end_validated"]=false,["encoder_sentis_gate_status"]="not_executed_for_current_producer",
                ["encoder_sentis_gate_passed"]=false,["reference_source"]=Producer,["conditions"]=conditions,
                ["unity"]=Application.unityVersion,["gpu"]=SystemInfo.graphicsDeviceName,["graphics_api"]=SystemInfo.graphicsDeviceType.ToString(),
                ["preprocessing_precision"]="fp32",["preprocessing_stage_cases"]=preprocessingCases,["started_utc"]=DateTime.UtcNow.ToString("O"),["hash_backends"]=new JObject(),
                ["gpu_transfer_evidence"]="Fixed Runtime device contracts, original stage output backends and Worker CPU fallback/flush sets; GPU profiler capture is a separate gate."};Save();
            Assert.That(File.ReadAllText(Resolve(root,"expected-source.txt")).Trim(),Is.EqualTo(Producer));
            Assert.That(Read(root,"staging-receipt.json").Value<bool>("success"),Is.True,"Complete audited staging first.");
            AuditHash(Resolve(root,"audio-runtime-reference/audio-encoder-validation.json"),"68d29c96012ef0a44114f2d3edbbff8d14e116ac697abee1a36d0b212155cfa1");
            AuditHash(Resolve(root,"audio-runtime-preprocessing/audio-validation.json"),"4e325f708e8d103a9de358a411380ddbb51cfe953984e9dcb73e291d919630fa");
            AuditHash(Resolve(backboneRoot,"audio-embedding-validation.json"),"9173afd593113769ed9fb9518c6092ac532d0cb055fc8f97eea2de811248e413");
            AuditHash(Resolve(backboneRoot,"saved-models-receipt.json"),"f79df6aa24fed7f154598a8035bd6cb104dd9d6168db74901be636beb9b9a047");
            encoderAudit=Read(root,"audio-runtime-reference/audio-encoder-validation.json");preprocessingAudit=Read(root,"audio-runtime-preprocessing/audio-validation.json");
            var finalAudit=Read(backboneRoot,"audio-embedding-validation.json");savedReceipt=Read(backboneRoot,"saved-models-receipt.json");
            AuditFile(backboneRoot,"audio-embedding-reference.json",finalAudit["files"]["audio-embedding-reference.json"]);
            reference=Read(backboneRoot,"audio-embedding-reference.json");
            AuditFile(root,"audio-runtime-reference/audio-encoder-reference.json",encoderAudit["files"]["audio-encoder-reference.json"]);
            encoder=Read(root,"audio-runtime-reference/audio-encoder-reference.json");
            AudioBackboneReferenceGuard.ValidateContext(reference,finalAudit,Producer);
            AudioChainReferenceGuard.ValidateProducers((JObject)reference["metadata"],(JObject)encoder["metadata"],(JObject)preprocessingAudit["metadata"],Producer);
            Assert.That(encoderAudit["files"]["audio-encoder-reference.json"].Value<string>("sha256"),Is.EqualTo(reference["metadata"].Value<string>("encoder_reference_sha256")));
            Assert.That(JToken.DeepEquals(encoder["metadata"],encoderAudit["metadata"]),Is.True);
            Assert.That(savedReceipt.Value<string>("reference_producer"),Is.EqualTo(Producer));
            foreach(var file in (JObject)encoderAudit["files"])
                if(!file.Key.EndsWith(".pt2",StringComparison.Ordinal))AuditFile(root,"audio-runtime-reference/"+file.Key,file.Value);
            foreach(var file in (JObject)preprocessingAudit["files"])AuditFile(root,"audio-runtime-preprocessing/"+file.Key,file.Value);
            foreach(var file in (JObject)savedReceipt["saved_models"])AuditFile(backboneRoot,file.Key,file.Value);
            foreach(var name in new[]{"tokenizer.json","tokenizer_config.json","config_sentence_transformers.json"})
                AuditHash(Resolve(root,"tokenizer/"+name),reference["metadata"]["input_sha256"].Value<string>(name));
            var runtime=typeof(AudioPcmInput).Assembly;
            encoderType=runtime.GetType("EmbeddingGemma.AudioFeatureEncoder");profileType=runtime.GetType("EmbeddingGemma.AudioEncoderProfile");embedderType=runtime.GetType("EmbeddingGemma.AudioEmbedder");
            Assert.That(encoderType,Is.Not.Null,"Use the fixed candidate Runtime API, not a source overlay.");Assert.That(profileType,Is.Not.Null);Assert.That(embedderType,Is.Not.Null);
            Assert.That(typeof(MultimodalPromptBuilder).GetMethod("BuildAudio"),Is.Not.Null);
            report["source_encoder_python_success"]=encoderAudit["success"];report["metadata_and_files_audited"]=true;Save();
        }
        [TestCase("fp32",BackendType.CPU)] [TestCase("fp32",BackendType.GPUCompute)]
        [TestCase("float16",BackendType.CPU)] [TestCase("float16",BackendType.GPUCompute)]
        [Timeout(1500000)]
        public void DocumentPcmCandidateChainMatchesOfficialFinalVectors(string precision,BackendType backend)
        {
            var condition=new JObject{["precision"]=precision,["requested_backend"]=backend.ToString(),["success"]=false,["cases"]=new JArray(),
                ["cpu_fallback_count"]=0,["readback_flush_count"]=0,["intermediate_readback_count"]=0};conditions.Add(condition);Save();
            try
            {
                if(backend==BackendType.GPUCompute)Assert.That(SystemInfo.supportsComputeShaders,Is.True,"Actual GPU required; never skip.");
                foreach(var entry in (JObject)reference["profiles"])
                {
                    var spec=(JObject)entry.Value;var encoderSpec=encoder["profiles"][entry.Key];
                    condition["phase"]="load_"+entry.Key;Save();
                    var frameMask=Ints(Resolve(root,"audio-runtime-reference/"+encoderSpec.Value<string>("frame_mask")));var validFrames=frameMask.Count(x=>x==1);
                    Assert.That(frameMask.Length,Is.EqualTo(1119));Assert.That(frameMask,Is.EqualTo(Enumerable.Repeat(1,validFrames).Concat(Enumerable.Repeat(0,1119-validFrames)).ToArray()));
                    var metadata=reference["metadata"];var profile=new MultimodalBackboneProfile(1024,metadata.Value<int>("image_token_id"),metadata.Value<int>("audio_token_id"),
                        0,spec.Value<int>("audio_feature_count"),metadata.Value<int>("vocabulary_size"),metadata.Value<int>("pad_token_id"));
                    Assert.That(profile.AudioFeatureCount,Is.EqualTo(encoderSpec.Value<int>("feature_count")));
                    if(!prompts.TryGetValue(entry.Key,out var builder))prompts.Add(entry.Key,builder=new MultimodalPromptBuilder(
                        File.ReadAllText(Resolve(root,"tokenizer/tokenizer.json")),File.ReadAllText(Resolve(root,"tokenizer/tokenizer_config.json")),profile));
                    if(!saved.TryGetValue("preprocessor",out var ppSaved))saved.Add("preprocessor",ppSaved=new Lazy<string[]>(()=>ImportAndSave("preprocessor","audio-runtime-preprocessing/audio-preprocessor.pt2",false)));
                    if(!saved.TryGetValue(entry.Key,out var encSaved))saved.Add(entry.Key,encSaved=new Lazy<string[]>(()=>ImportAndSave(entry.Key,"audio-runtime-reference/"+encoderSpec.Value<string>("graph"),true)));
                    var ppModel=TextModelFile.Load(ppSaved.Value[0]);var encModel=TextModelFile.Load(encSaved.Value[precision=="fp32"?0:1]);
                    var backboneModel=TextModelFile.Load(Resolve(backboneRoot,entry.Key+"-"+precision+".sentis"));
                    using(var preprocessing=new AudioPreprocessor(ppModel,backend))using(var backbone=new MultimodalBackbone(backboneModel,profile,backend))
                    {
                        var encoderProfile=Activator.CreateInstance(profileType,validFrames);
                        using var featureEncoder=(IDisposable)Activator.CreateInstance(encoderType,encModel,backend,encoderProfile);
                        using var api=(IDisposable)Activator.CreateInstance(embedderType,preprocessing,featureEncoder,backbone,builder);
                        var workers=new[]{WorkerOf(preprocessing),WorkerOf(featureEncoder),WorkerOf(backbone)};
                        var models=new[]{ppModel,encModel,backboneModel};
                        for(var index=0;index<workers.Length;index++)
                        {
                            Assert.That(workers[index].backendType,Is.EqualTo(backend));
                            var cpu=CpuFallback(workers[index],models[index],index==2);var flush=Flush(workers[index]);
                            if(backend==BackendType.GPUCompute){condition["cpu_fallback_count"]=(int)condition["cpu_fallback_count"]+cpu;condition["readback_flush_count"]=(int)condition["readback_flush_count"]+flush;Assert.That(cpu,Is.Zero);Assert.That(flush,Is.Zero);}
                        }
                        foreach(var row in ((JArray)reference["cases"]).Where(row=>row.Value<string>("profile")==entry.Key))
                        {
                            var id=row.Value<string>("id");condition["phase"]="evaluate_"+id;Save();
                            var input=AudioChainReferenceGuard.FromPaddedPcm(Floats(Resolve(root,"audio-runtime-preprocessing/"+id+"-pcm-fp32.bin")),
                                Ints(Resolve(root,"audio-runtime-preprocessing/"+id+"-samples-int32.bin")),validFrames);
                            VerifyPreprocessing(preprocessing,input,id,backend);
                            var prompt=(MultimodalPrompt)Invoke(typeof(MultimodalPromptBuilder).GetMethod("BuildAudio"),builder,new object[]{validFrames,TextRole.Document,null});
                            Assert.That(prompt.InputIds,Is.EqualTo(row["input_ids"].ToObject<int[]>()),"Official full Document prompt IDs.");
                            Assert.That(prompt.AttentionMask,Is.EqualTo(row["attention_mask"].ToObject<int[]>()));
                            float[] actual;
                            try{actual=(float[])Invoke(embedderType.GetMethod("Embed"),api,new object[]{input,TextRole.Document,null});}
                            catch(Exception error){condition["failed_case_id"]=id;condition["original_api_error"]=error.ToString();Save();CaptureFailedStages(condition,workers,id,precision,backend);throw;}
                            var devices=workers.All(worker=>worker.PeekOutput().backendType==backend) && workers[0].PeekOutput(ppModel.outputs[1].name).backendType==backend;
                            var result=AudioBackboneReferenceGuard.CompareVectors(actual,row["embedding"].ToObject<float[]>(),precision=="float16");
                            result["id"]=id;result["profile"]=entry.Key;result["actual_backend"]=workers[2].PeekOutput().backendType.ToString();
                            result["embedding_dimension"]=actual.Length;result["norm_squared"]=actual.Sum(value=>(double)value*value);result["intermediate_devices_matched"]=devices;
                            ((JArray)condition["cases"]).Add(result);Save();Assert.That(devices,Is.True);
                        }
                    }
                    ppModel=null;encModel=null;backboneModel=null;GC.Collect();GC.WaitForPendingFinalizers();
                }
                Assert.That(((JArray)condition["cases"]).Count,Is.EqualTo(6));Assert.That(((JArray)condition["cases"]).All(row=>row.Value<bool>("success")),Is.True);
                condition["success"]=true;condition["phase"]="completed";
            }
            catch(Exception error){condition["error"]=error.ToString();throw;}finally{Save();}
        }
        string[] ImportAndSave(string id,string graph,bool quantize)
        {
            report["import_phase"]=id;Save();var assetPath="Assets/AudioRealChainGenerated/"+id+Path.GetExtension(graph);
            Directory.CreateDirectory(Path.GetDirectoryName(assetPath));File.Copy(Resolve(root,graph),assetPath,true);AssetDatabase.ImportAsset(assetPath,ImportAssetOptions.ForceSynchronousImport);
            var asset=AssetDatabase.LoadAssetAtPath<ModelAsset>(assetPath);Assert.That(asset,Is.Not.Null);var model=ModelLoader.Load(asset);
            var fp32=Resolve(root,id+"-fp32.sentis");var half=Resolve(root,id+"-float16.sentis");
            try{TextModelFile.Save(model,fp32);if(quantize){ModelQuantizer.QuantizeWeights(QuantizationType.Float16,ref model);TextModelFile.Save(model,half);}}
            finally{model=null;Resources.UnloadAsset(asset);GC.Collect();GC.WaitForPendingFinalizers();}
            return new[]{fp32,quantize?half:fp32};
        }
        [OneTimeTearDown]public void Complete()
        {
            if(report==null)return;var summary=AudioChainReferenceGuard.Summarize(conditions,encoderAudit?.Value<bool>("success")==true,false);
            report["preprocessing_stage_validated"]=preprocessingCases.Count==12 && preprocessingCases.All(row=>row.Value<bool>("success"));
            foreach(var property in summary)report[property.Key]=property.Value;report["completed_utc"]=DateTime.UtcNow.ToString("O");Save();prompts.Clear();
        }
        void VerifyPreprocessing(AudioPreprocessor preprocessor,AudioPcmInput input,string id,BackendType backend)
        {
            var key=backend+"/"+id;if(preprocessingChecked.Contains(key))return;
            // Explicit standalone stage diagnostic. It ends before the public API's
            // GPU chain; no diagnostic readback is performed inside AudioEmbedder.
            using var diagnostic=preprocessor.Process(input);
            Assert.That(diagnostic.Features.backendType,Is.EqualTo(backend));Assert.That(diagnostic.FrameMask.backendType,Is.EqualTo(backend));
            var actual=diagnostic.Features.DownloadToArray();var mask=diagnostic.FrameMask.DownloadToArray();
            var expected=Floats(Resolve(root,"audio-runtime-preprocessing/"+id+"-features-fp32.bin"));
            var expectedMask=Ints(Resolve(root,"audio-runtime-preprocessing/"+id+"-frames-int32.bin"));
            Assert.That(actual.Length,Is.EqualTo(expected.Length));int mismatches=0;double maximum=0,ratio=0;
            for(var index=0;index<actual.Length;index++)
            {
                var difference=Math.Abs((double)actual[index]-expected[index]);var bound=0.004+0.0001*Math.Abs((double)expected[index]);
                if(float.IsNaN(actual[index])||float.IsInfinity(actual[index])||difference>bound)mismatches++;
                maximum=Math.Max(maximum,difference);ratio=Math.Max(ratio,difference/bound);
            }
            var result=new JObject{["id"]=id,["actual_backend"]=backend.ToString(),["elementwise_mismatches"]=mismatches,
                ["mask_exact"]=mask.SequenceEqual(expectedMask),["maximum_absolute_error"]=maximum,["maximum_tolerance_ratio"]=ratio,
                ["diagnostic_readback_count"]=backend==BackendType.GPUCompute?2:0,["success"]=mismatches==0&&mask.SequenceEqual(expectedMask)};
            preprocessingCases.Add(result);Save();Assert.That(result.Value<bool>("success"),Is.True,"Separate official PCM preprocessing stage criterion.");
            preprocessingChecked.Add(key);
        }
        static Worker WorkerOf(object value)=>(Worker)value.GetType().GetField("worker",BindingFlags.Instance|BindingFlags.NonPublic).GetValue(value);
        void CaptureFailedStages(JObject condition,Worker[] workers,string id,string precision,BackendType backend)
        {
            var diagnostics=new JObject();condition["failure_stage_diagnostics"]=diagnostics;
            var names=new[]{"preprocessing_features","encoder_features","final_embedding"};
            for(var index=0;index<workers.Length;index++)
            {
                condition["phase"]="diagnose_"+id+"_"+names[index];Save();
                try
                {
                    var tensor=workers[index].PeekOutput() as Tensor<float>;Assert.That(tensor,Is.Not.Null);
                    var values=tensor.DownloadToArray();var stats=AudioChainReferenceGuard.DescribeValues(values);
                    stats["actual_backend"]=tensor.backendType.ToString();stats["readback_scope"]="failure_diagnostic_after_api_exception";
                    var filename="failure-"+precision+"-"+backend+"-"+id+"-"+names[index]+".bin";
                    var bytes=new byte[values.Length*4];Buffer.BlockCopy(values,0,bytes,0,bytes.Length);
                    File.WriteAllBytes(Path.Combine(Path.GetDirectoryName(resultPath),filename),bytes);
                    stats["local_binary"]=filename;diagnostics[names[index]]=stats;
                }
                catch(Exception error){diagnostics[names[index]]=new JObject{["diagnostic_error"]=error.ToString()};}
                Save();
            }
            condition["phase"]="failed_"+id;Save();
        }
        static int CpuFallback(Worker worker,Model model,bool backbone)
        {
            var fallback=(HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallback",BindingFlags.Instance|BindingFlags.NonPublic).GetValue(worker);
            return fallback.Intersect(model.layers.SelectMany(layer=>layer.outputs).Concat(model.inputs.Where(input=>!backbone||input.name=="audio_features").Select(input=>input.index))).Count();
        }
        static int Flush(Worker worker)=>((HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallbackShouldFlushGPU",BindingFlags.Instance|BindingFlags.NonPublic).GetValue(worker)).Count;
        static object Invoke(MethodInfo method,object instance,object[] args)
        {try{return method.Invoke(instance,args);}catch(TargetInvocationException error){ExceptionDispatchInfo.Capture(error.InnerException).Throw();throw;}}
        void AuditFile(string directory,string name,JToken spec)
        {Assert.That(spec,Is.Not.Null);var path=Resolve(directory,name);Assert.That(new FileInfo(path).Length,Is.EqualTo(spec.Value<long>("bytes")));AuditHash(path,spec.Value<string>("sha256"));}
        void AuditHash(string path,string expected)
        {
            var type=AppDomain.CurrentDomain.GetAssemblies().Select(assembly=>assembly.GetType("EmbeddingGemma.Validation.PlayerFileHash")).FirstOrDefault(value=>value!=null);
            Assert.That(type,Is.Not.Null,"Use the unchanged validated platform SHA-256 helper.");using var stream=File.OpenRead(path);var args=new object[]{stream,null,null};
            var digest=(string)Invoke(type.GetMethod("Compute"),null,args);Assert.That(digest,Is.EqualTo(expected));
            ((JObject)report["hash_backends"])[Path.GetFileName(path)]=(string)args[1];
        }
        static string Resolve(string directory,string name)
        {var path=Path.GetFullPath(Path.Combine(directory,name));Assert.That(path.StartsWith(directory+Path.DirectorySeparatorChar,StringComparison.Ordinal),Is.True);return path;}
        static JObject Read(string directory,string name)=>JObject.Parse(File.ReadAllText(Resolve(directory,name)));
        static float[] Floats(string path){var bytes=File.ReadAllBytes(path);Assert.That(bytes.Length%4,Is.Zero);var values=new float[bytes.Length/4];Buffer.BlockCopy(bytes,0,values,0,bytes.Length);return values;}
        static int[] Ints(string path){var bytes=File.ReadAllBytes(path);Assert.That(bytes.Length%4,Is.Zero);var values=new int[bytes.Length/4];Buffer.BlockCopy(bytes,0,values,0,bytes.Length);return values;}
        void Save(){Directory.CreateDirectory(Path.GetDirectoryName(resultPath));File.WriteAllText(resultPath,report.ToString());}
    }
}
