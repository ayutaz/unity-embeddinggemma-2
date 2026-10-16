using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEngine;

namespace EmbeddingGemma.Tests
{
    public sealed class BoundedBackboneReferenceTests
    {
        const string RuntimeSource="66c72fb5277dc5c3b2897cf71da583413411bc77";
        const string ReferenceSource="4edbe17a8105233c5aaa1514faa2862b08a4f7e0";
        string root,resultPath;
        JObject reference,validation,receipt,report;
        JArray conditions;
        readonly HashSet<string> audited=new HashSet<string>();
        [OneTimeSetUp] public void AuditSavedModelsAndOriginalFinalReference()
        {
            root=Path.GetFullPath(Path.Combine(Application.dataPath,"../artifacts/audio-final-backbone"));
            resultPath=Path.GetFullPath(Path.Combine(Application.dataPath,"../artifacts/bounded-backbone-results/results.json"));
            conditions=new JArray();report=new JObject {["success"]=false,["scope"]="bounded_upload_native_audio_features_to_final_backbone_only",
                ["runtime_source"]=RuntimeSource,["reference_source"]=ReferenceSource,["runtime_overlay"]=false,
                ["encoder_validated"]=false,["preprocessing_validated"]=false,["end_to_end_validated"]=false,
                ["unity"]=Application.unityVersion,["sentis"]="2.6.1",["gpu"]=SystemInfo.graphicsDeviceName,
                ["graphics_api"]=SystemInfo.graphicsDeviceType.ToString(),["started_utc"]=DateTime.UtcNow.ToString("O"),["conditions"]=conditions};Save();
            var locked=JObject.Parse(File.ReadAllText(Path.Combine(Application.dataPath,"../Packages/packages-lock.json")));
            Assert.That(locked["dependencies"]["com.ayutaz.embeddinggemma"].Value<string>("hash"),Is.EqualTo(RuntimeSource));
            Audit(Resolve("audio-embedding-validation.json"),"9173afd593113769ed9fb9518c6092ac532d0cb055fc8f97eea2de811248e413");
            Audit(Resolve("audio-embedding-reference.json"),"281d12573b431667e8841033db1c6a10dce4382ef98c9a5345c18044a1194001");
            Audit(Resolve("saved-models-receipt.json"),"f79df6aa24fed7f154598a8035bd6cb104dd9d6168db74901be636beb9b9a047");
            validation=JObject.Parse(File.ReadAllText(Resolve("audio-embedding-validation.json")));
            reference=JObject.Parse(File.ReadAllText(Resolve("audio-embedding-reference.json")));
            receipt=JObject.Parse(File.ReadAllText(Resolve("saved-models-receipt.json")));
            AudioBackboneReferenceGuard.ValidateContext(reference,validation,ReferenceSource);
            foreach(var row in (JArray)reference["cases"])
            {var file=row.Value<string>("features_file");var identity=validation["files"][file];Assert.That(new FileInfo(Resolve(file)).Length,Is.EqualTo(identity.Value<long>("bytes")));Audit(Resolve(file),identity.Value<string>("sha256"));}
            report["source_encoder_success"]=validation["source_encoder_validation_success"];Save();
        }
        [TestCase("fp32",BackendType.CPU)] [TestCase("fp32",BackendType.GPUCompute)]
        [TestCase("float16",BackendType.CPU)] [TestCase("float16",BackendType.GPUCompute)]
        [Timeout(1200000)] public void SavedNativeFeaturesKeepEveryOriginalFinalVector(string precision,BackendType backend)
        {
            var condition=new JObject {["precision"]=precision,["requested_backend"]=backend.ToString(),["success"]=false,["cases"]=new JArray(),["uploads"]=new JArray()};conditions.Add(condition);Save();
            try
            {
                if(backend==BackendType.GPUCompute)Assert.That(SystemInfo.supportsComputeShaders,Is.True,"Actual GPU required; never skip.");
                foreach(var entry in (JObject)reference["profiles"])
                {
                    condition["phase"]="audit_saved_"+entry.Key;Save();var file=entry.Key+"-"+precision+".sentis";
                    var identity=receipt["saved_models"][file];Assert.That(identity,Is.Not.Null);
                    Assert.That(new FileInfo(Resolve(file)).Length,Is.EqualTo(identity.Value<long>("bytes")));Audit(Resolve(file),identity.Value<string>("sha256"));
                    var model=TextModelFile.Load(Resolve(file));
                    try
                    {
                        var spec=entry.Value;var meta=reference["metadata"];
                        var profile=new MultimodalBackboneProfile(1024,meta.Value<int>("image_token_id"),meta.Value<int>("audio_token_id"),0,
                            spec.Value<int>("audio_feature_count"),meta.Value<int>("vocabulary_size"),meta.Value<int>("pad_token_id"));
                        condition["phase"]="construct_and_upload_"+entry.Key;Save();
                        using var backbone=new MultimodalBackbone(model,profile,backend);
                        var owner=(GpuConstantWorker)typeof(MultimodalBackbone).GetField("workerOwner",BindingFlags.Instance|BindingFlags.NonPublic).GetValue(backbone);
                        var worker=owner.Worker;Assert.That(worker.backendType,Is.EqualTo(backend));
                        ((JArray)condition["uploads"]).Add(new JObject {["profile"]=entry.Key,["uploaded_constants"]=owner.UploadedConstantCount,
                            ["upload_calls"]=owner.UploadCallCount,["maximum_upload_bytes"]=owner.MaximumUploadBytes,["skipped_cpu_constants"]=owner.SkippedCpuConstantCount});Save();
                        Assert.That(owner.MaximumUploadBytes,Is.LessThanOrEqualTo(GpuConstantWorker.DefaultChunkBytes));
                        if(backend==BackendType.GPUCompute)Assert.That(owner.UploadedConstantCount,Is.GreaterThan(0));
                        var resident=(HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallback",BindingFlags.Instance|BindingFlags.NonPublic).GetValue(worker);
                        var cpu=resident.Intersect(model.layers.SelectMany(layer=>layer.outputs).Concat(model.inputs.Where(input=>input.name=="audio_features").Select(input=>input.index))).ToArray();
                        var flush=(HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallbackShouldFlushGPU",BindingFlags.Instance|BindingFlags.NonPublic).GetValue(worker);
                        condition["cpu_computation_or_nonempty_feature_inputs"]=new JArray(cpu);condition["readback_flush"]=new JArray(flush);
                        if(backend==BackendType.GPUCompute){Assert.That(cpu,Is.Empty);Assert.That(flush,Is.Empty);}
                        foreach(var row in ((JArray)reference["cases"]).Where(row=>row.Value<string>("profile")==entry.Key))
                        {
                            condition["phase"]="evaluate_"+row.Value<string>("id");Save();
                            using var empty=new Tensor<float>(new TensorShape(0,512));
                            using var features=new Tensor<float>(new TensorShape(profile.AudioFeatureCount,512),Floats(Resolve(row.Value<string>("features_file"))));
                            if(backend==BackendType.GPUCompute)ComputeTensorData.Pin(features);
                            var actual=backbone.Embed(row["input_ids"].ToObject<int[]>(),row["attention_mask"].ToObject<int[]>(),empty,features);
                            var result=AudioBackboneReferenceGuard.CompareVectors(actual,row["embedding"].ToObject<float[]>(),precision=="float16");
                            result["id"]=row["id"];result["profile"]=entry.Key;result["actual_backend"]=worker.PeekOutput().backendType.ToString();
                            ((JArray)condition["cases"]).Add(result);Save();
                        }
                    }
                    finally{model=null;GC.Collect();GC.WaitForPendingFinalizers();}
                }
                Assert.That(((JArray)condition["cases"]).Count,Is.EqualTo(6));Assert.That(((JArray)condition["cases"]).All(row=>row.Value<bool>("success")),Is.True);
                condition["success"]=true;condition["phase"]="completed";
            }
            catch(Exception error){condition["error"]=error.ToString();throw;}finally{Save();}
        }
        [OneTimeTearDown] public void Complete(){report["success"]=conditions.Count==4&&conditions.All(row=>row.Value<bool>("success"));report["completed_utc"]=DateTime.UtcNow.ToString("O");Save();}
        string Resolve(string name){var path=Path.GetFullPath(Path.Combine(root,name));Assert.That(path.StartsWith(root+Path.DirectorySeparatorChar,StringComparison.Ordinal),Is.True);return path;}
        void Audit(string path,string expected)
        {if(audited.Contains(path))return;var type=AppDomain.CurrentDomain.GetAssemblies().Select(assembly=>assembly.GetType("EmbeddingGemma.Validation.PlayerFileHash")).FirstOrDefault(value=>value!=null);Assert.That(type,Is.Not.Null);using var stream=File.OpenRead(path);var args=new object[] {stream,null,null};Assert.That((string)type.GetMethod("Compute").Invoke(null,args),Is.EqualTo(expected),path);audited.Add(path);report["hash_backend"]=args[1].ToString();Save();}
        static float[] Floats(string path){var bytes=File.ReadAllBytes(path);var values=new float[bytes.Length/4];Buffer.BlockCopy(bytes,0,values,0,bytes.Length);return values;}
        void Save(){Directory.CreateDirectory(Path.GetDirectoryName(resultPath));File.WriteAllText(resultPath,report.ToString());}
    }
}
