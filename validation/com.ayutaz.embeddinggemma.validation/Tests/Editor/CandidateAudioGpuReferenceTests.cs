using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using EmbeddingGemma.Tests;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using UnityEditor;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.TestTools;

namespace EmbeddingGemma.Validation.Tests
{
    // New generation acceptance is separate from every retained-model test.
    public sealed class CandidateAudioGpuReferenceTests
    {
        [UnityTest]
        public IEnumerator FullProfileMustMatchAllFiveOriginalSignalsOnGpu()
            =>Wait(Run("frames-1119-valid-1119"));
        [UnityTest]
        public IEnumerator PaddedProfileMustMatchOriginalSignalOnGpu()
            =>Wait(Run("frames-1119-valid-839"));
        static IEnumerator Wait(Task task)
        {
            while(!task.IsCompleted)yield return null;
            task.GetAwaiter().GetResult();
        }
        static float[] Floats(string path)
        {
            if(!BitConverter.IsLittleEndian)throw new NotSupportedException("This fixed validation reads little-endian FP32 coefficients.");
            var bytes=File.ReadAllBytes(path);Assert.That(bytes.Length%4,Is.Zero);
            var values=new float[bytes.Length/4];Buffer.BlockCopy(bytes,0,values,0,bytes.Length);return values;
        }
        static IGpuTensorOperation Operation(string root,JObject operation)
        {
            switch(operation.Value<string>("kind"))
            {
                case "silu":return new GpuSilu();
                case "tanh":return new GpuTanh();
                case "scalar_divide":return new GpuScalarDivide(operation.Value<float>("divisor"));
                case "softmax":return new GpuSoftmax();
                case "glu":return new GpuGlu();
                case "batched_matmul":return new GpuBatchedMatmul();
                case "sequential_scale":return new GpuSequentialScale(operation.Value<float>("first_scale"),Floats(Path.Combine(root,operation.Value<string>("file"))));
                case "matmul":
                    var shape=operation["weight_shape"].ToObject<int[]>();
                    return new GpuBlockedMatmul(shape[0],shape[1],Floats(Path.Combine(root,operation.Value<string>("file"))));
                default:throw new InvalidDataException("An unqualified GPU operation kind was selected.");
            }
        }
        static void AuditGitSource(string source)
        {
            Assert.That(source,Does.Match("^[0-9a-f]{40}$"));
            var root=Path.GetFullPath(Path.Combine(Application.dataPath,".."));
            var manifest=JObject.Parse(File.ReadAllText(Path.Combine(root,"Packages/manifest.json")));
            var locked=JObject.Parse(File.ReadAllText(Path.Combine(root,"Packages/packages-lock.json")));
            foreach(var name in new[]{"com.ayutaz.embeddinggemma","com.ayutaz.embeddinggemma.validation"})
            {
                Assert.That(manifest["dependencies"].Value<string>(name).EndsWith("#"+source,StringComparison.Ordinal),Is.True);
                Assert.That(locked["dependencies"][name].Value<string>("source"),Is.EqualTo("git"));
                Assert.That(locked["dependencies"][name].Value<string>("hash"),Is.EqualTo(source));
            }
        }
        static async Task Run(string profile)
        {
            var source=Environment.GetEnvironmentVariable("EG_ACCEPTANCE_SOURCE");
            var original=Environment.GetEnvironmentVariable("EG_ORIGINAL_AUDIO_ROOT");
            var output=Environment.GetEnvironmentVariable("EG_CANDIDATE_GPU_REPORT");
            Assert.That(output,Is.Not.Null.And.Not.Empty);Assert.That(original,Is.Not.Null.And.Not.Empty);
            Assert.That(File.Exists(output),Is.False);
            var report=new JObject
            {
                ["success"]=false,["scope"]="saved_candidate_encoder_original_reference_gpu_gate_only",
                ["runtime_source"]=source,["candidate_producer"]=CandidateAudioPlanGuard.Producer,
                ["profile"]=profile,["unity"]=Application.unityVersion,["sentis"]="2.6.1",
                ["gpu"]=SystemInfo.graphicsDeviceName,["graphics_api"]=SystemInfo.graphicsDeviceType.ToString(),
                ["phase"]="audit",["generation_qualified"]=false,["final_roles_validated"]=false,
                ["consumer_player_validated"]=false,["original_golden_changed"]=false
            };
            void Save()
            {
                report["scope"]="saved_candidate_encoder_original_reference_gpu_gate_only";
                Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(output)));File.WriteAllText(output,report.ToString());
            }
            var operations=new List<IGpuTensorOperation>();Save();
            try
            {
                try
                {
                    Assert.That(SystemInfo.supportsComputeShaders,Is.True,"Actual GPUCompute required; never skip.");
                    Assert.That(Application.unityVersion,Is.EqualTo("6000.3.21f1"));AuditGitSource(source);
                    var relative="Assets/CandidateGpuAudio/"+profile;
                    var root=Path.GetFullPath(Path.Combine(Application.dataPath,"CandidateGpuAudio",profile));
                    var plan=CandidateAudioPlanGuard.Load(root,profile);
                    report["plan_sha256"]=CandidateAudioPlanGuard.Hash(Path.Combine(root,"partition-plan.json"));
                    report["preparation_receipt_sha256"]=CandidateAudioPlanGuard.Hash(Path.Combine(root,"audio-gpu-refinement.json"));
                    report["all_415_model_coefficient_hashes_verified"]=true;
                    Assert.That(CandidateAudioPlanGuard.Hash(Path.Combine(original,"audio-encoder-reference.json")),Is.EqualTo("ac23f39904b8149dd5ccb6485afcc93d96047f0977b5faf1d640c8bd018c13c9"));
                    Assert.That(CandidateAudioPlanGuard.Hash(Path.Combine(original,"audio-encoder-validation.json")),Is.EqualTo("68d29c96012ef0a44114f2d3edbbff8d14e116ac697abee1a36d0b212155cfa1"));
                    var reference=JObject.Parse(File.ReadAllText(Path.Combine(original,"audio-encoder-reference.json")));
                    var audit=JObject.Parse(File.ReadAllText(Path.Combine(original,"audio-encoder-validation.json")));
                    report["phase"]="load_fixed_segments";Save();
                    var models=new List<Model>();
                    foreach(var segment in (JArray)plan["segments"])
                    {
                        var asset=AssetDatabase.LoadAssetAtPath<ModelAsset>(relative+"/"+segment.Value<string>("file"));
                        Assert.That(asset,Is.Not.Null);var model=ModelLoader.Load(asset);var aliases=(JObject)segment["output_aliases"];
                        for(var index=0;index<model.outputs.Count;index++)
                        {
                            var boundary=model.outputs[index];
                            if(aliases?.Property(boundary.name)!=null){boundary.name=aliases.Value<string>(boundary.name);model.outputs[index]=boundary;}
                        }
                        model=CompensatedModel.CreateWithDynamicMatMuls(model,16);model=CompensatedModel.Create(model,64);models.Add(model);
                        await Task.Yield();
                    }
                    foreach(JObject row in (JArray)plan["operations"])operations.Add(Operation(root,row));
                    var inputs=((JArray)plan["operations"]).Select(row=>row["inputs"].ToObject<string[]>()).ToArray();
                    var outputs=((JArray)plan["operations"]).Select(row=>row.Value<string>("output")).ToArray();
                    var frames=profile=="frames-1119-valid-839"?839:1119;
                    report["phase"]="construct_audited_gpu_pipeline";Save();
                    using var encoder=new PartitionedAudioFeatureEncoder(models.ToArray(),inputs,outputs,operations.ToArray(),new AudioEncoderProfile(frames));
                    report["operation_count"]=operations.Count;report["segment_count"]=models.Count;
                    await QualifiedStagedAudioEncoderGate.Run(encoder,original,reference,audit,report,output,Save);
                    report["scope"]="saved_candidate_encoder_original_reference_gpu_gate_only";
                }
                finally
                {
                    for(var index=operations.Count-1;index>=0;index--)operations[index].Dispose();
                }
                report["success"]=true;report["phase"]="completed_and_disposed";
            }
            catch(Exception error){report["success"]=false;report["error_type"]=error.GetType().Name;throw;}
            finally{Save();}
        }
    }
}
