using System;
using System.Collections.Generic;
using System.IO;
using System.Text.RegularExpressions;
using EmbeddingGemma.Tests;
using Newtonsoft.Json.Linq;
using UnityEditor;
using Unity.InferenceEngine;
using UnityEngine;

namespace EmbeddingGemma.Validation.Tests
{
    // Serializes the exact pinned candidate partitions. This never executes a
    // model and never declares the generated bundle or generation qualified.
    public static class CandidateAudioBundlePreparation
    {
        static void AuditGitSource(string source)
        {
            var root=Path.GetFullPath(Path.Combine(Application.dataPath,".."));
            var manifest=JObject.Parse(File.ReadAllText(Path.Combine(root,"Packages/manifest.json")));
            var locked=JObject.Parse(File.ReadAllText(Path.Combine(root,"Packages/packages-lock.json")));
            foreach(var name in new[]{"com.ayutaz.embeddinggemma","com.ayutaz.embeddinggemma.validation"})
                if(manifest["dependencies"]?.Value<string>(name)?.EndsWith("#"+source,StringComparison.Ordinal)!=true
                    || locked["dependencies"]?[name]?.Value<string>("source")!="git"
                    || locked["dependencies"]?[name]?.Value<string>("hash")!=source)
                    throw new InvalidDataException("Both preparation packages must resolve from the fixed execution commit.");
        }
        public static JObject Prepare(string source,string output)
        {
            if(!Regex.IsMatch(source??"","\\A[0-9a-f]{40}\\z") || string.IsNullOrWhiteSpace(output))
                throw new ArgumentException("A fixed execution source and new bundle directory are required.");
            output=Path.GetFullPath(output);
            if(Directory.Exists(output)||File.Exists(output))throw new IOException("Candidate preparation never overwrites existing evidence.");
            AuditGitSource(source);
            var plans=new Dictionary<string,JObject>();
            foreach(var profile in new[]{"frames-1119-valid-1119","frames-1119-valid-839"})
                plans.Add(profile,CandidateAudioPlanGuard.Load(Path.Combine(Application.dataPath,"CandidateGpuAudio",profile),profile));
            var writer=new GpuMediaBundleWriter(output,source);
            var results=new JArray();var report=new JObject
            {
                ["schema"]=1,["success"]=false,["scope"]="pinned_saved_candidate_audio_bundle_serialization_only",
                ["runtime_source"]=source,["partition_preparation_source"]=CandidateAudioPlanGuard.Producer,
                ["profiles"]=results,["model_executed"]=false,["gpu_validated"]=false,
                ["generation_qualified"]=false,["original_golden_changed"]=false,["phase"]="serialize_models"
            };
            var receipt=Path.Combine(output,"candidate-audio-preparation.json");
            void Save(){File.WriteAllText(receipt,report.ToString());}
            Save();
            try
            {
                foreach(var entry in plans)
                {
                    var profile=entry.Key;var plan=entry.Value;
                    var relative="Assets/CandidateGpuAudio/"+profile;
                    var root=Path.Combine(Application.dataPath,"CandidateGpuAudio",profile);
                    report["phase"]="serialize_"+profile;Save();var models=new List<Model>();
                    foreach(var segment in (JArray)plan["segments"])
                    {
                        var asset=AssetDatabase.LoadAssetAtPath<ModelAsset>(relative+"/"+segment.Value<string>("file"));
                        if(asset==null)throw new InvalidDataException("A pinned imported candidate segment is missing.");
                        var model=ModelLoader.Load(asset);var aliases=(JObject)segment["output_aliases"];
                        for(var index=0;index<model.outputs.Count;index++)
                        {
                            var boundary=model.outputs[index];
                            if(aliases?.Property(boundary.name)!=null){boundary.name=aliases.Value<string>(boundary.name);model.outputs[index]=boundary;}
                        }
                        model=CompensatedModel.CreateWithDynamicMatMuls(model,16);model=CompensatedModel.Create(model,64);models.Add(model);
                    }
                    var operations=new JArray();
                    foreach(JObject operation in (JArray)plan["operations"])
                        operations.Add(GpuMediaBundlePreparation.AudioOperation(writer,root,operation));
                    var frames=profile=="frames-1119-valid-839"?839:1119;
                    GpuMediaBundlePreparation.CommitAudio(writer,frames,models.ToArray(),operations);
                    results.Add(new JObject
                    {
                        ["profile"]=profile,["segment_count"]=models.Count,["operation_count"]=operations.Count,
                        ["source_plan_sha256"]=CandidateAudioPlanGuard.Hash(Path.Combine(root,"partition-plan.json")),
                        ["source_preparation_receipt_sha256"]=CandidateAudioPlanGuard.Hash(Path.Combine(root,"audio-gpu-refinement.json")),
                        ["source_original_initializer_sha256"]=plan["original_initializer_sha256"].DeepClone(),
                        ["existing_reference_values_changed"]=false,["gpu_validated"]=false
                    });
                    Save();
                }
                report["bundle_descriptor_sha256"]=CandidateAudioPlanGuard.Hash(Path.Combine(output,"bundle.json"));
                report["success"]=true;report["phase"]="prepared_not_bundle_validated";
            }
            catch(Exception error){report["error_type"]=error.GetType().Name;Save();throw;}
            Save();return report;
        }
        public static void Run()
        {
            try
            {
                Prepare(Environment.GetEnvironmentVariable("EG_ACCEPTANCE_SOURCE"),Environment.GetEnvironmentVariable("EG_CANDIDATE_BUNDLE_OUTPUT"));
                Debug.Log("Pinned candidate audio bundle preparation completed; model execution and bundle validation remain separate.");
            }
            catch(Exception error)
            {
                Debug.LogError("Candidate bundle preparation failed: "+error.GetType().Name);EditorApplication.Exit(1);
            }
        }
    }
}
