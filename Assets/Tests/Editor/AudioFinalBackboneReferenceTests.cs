using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Security.Cryptography;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEditor;
using UnityEngine;

namespace EmbeddingGemma.Tests
{
    // Real native audio features -> zero-image backbone. Not encoder / preprocessing / full chain acceptance.
    public sealed class AudioFinalBackboneReferenceTests
    {
        string root,resultPath;JObject reference,validation,report;JArray conditions;
        readonly Dictionary<string,Lazy<string[]>> saved=new Dictionary<string,Lazy<string[]>>();
        [OneTimeSetUp]
        public void AuditFinalNativeReference()
        {
            root=Path.GetFullPath(Path.Combine(Application.dataPath,"../artifacts/audio-final-backbone"));
            resultPath=Path.Combine(root,"../audio-final-backbone-results/results.json");conditions=new JArray();
            report=new JObject{["success"]=false,["scope"]="real_native_audio_features_to_final_backbone_only",
                ["encoders_validated"]=false,["gpu_preprocessing_validated"]=false,["end_to_end_validated"]=false,
                ["unity"]=Application.unityVersion,["gpu"]=SystemInfo.graphicsDeviceName,
                ["graphics_api"]=SystemInfo.graphicsDeviceType.ToString(),["conditions"]=conditions,["started_utc"]=DateTime.UtcNow.ToString("O")};Save();
            validation=JObject.Parse(File.ReadAllText(Resolve("audio-embedding-validation.json")));
            Audit("audio-embedding-reference.json");reference=JObject.Parse(File.ReadAllText(Resolve("audio-embedding-reference.json")));
            var source=File.ReadAllText(Resolve("expected-source.txt")).Trim();Assert.That(source,Does.Match("^[0-9a-f]{40}$"));
            AudioBackboneReferenceGuard.ValidateContext(reference,validation,source);
            foreach(var profile in (JObject)reference["profiles"])Audit(profile.Value.Value<string>("graph"));
            foreach(var row in (JArray)reference["cases"])Audit(row.Value<string>("features_file"));
            report["reference_source"]=source;report["source_encoder_success"]=validation["source_encoder_validation_success"];
            report["onnx_encoder_numeric"]=validation["onnx_encoder_numerically_validated"];Save();
        }
        [TestCase("fp32",BackendType.CPU)] [TestCase("fp32",BackendType.GPUCompute)]
        [TestCase("float16",BackendType.CPU)] [TestCase("float16",BackendType.GPUCompute)]
        [Timeout(1200000)]
        public void SavedAudioOnlyBackboneMatchesAllSixFinalVectors(string precision,BackendType backend)
        {
            var condition=new JObject{["precision"]=precision,["requested_backend"]=backend.ToString(),["success"]=false,["cases"]=new JArray()};
            conditions.Add(condition);Save();
            try
            {
                if(backend==BackendType.GPUCompute)Assert.That(SystemInfo.supportsComputeShaders,Is.True,"Actual GPU required; never skip.");
                foreach(var entry in (JObject)reference["profiles"])
                {
                    condition["phase"]="load_"+entry.Key;Save();var spec=(JObject)entry.Value;
                    if(!saved.TryGetValue(entry.Key,out var cache))saved.Add(entry.Key,cache=new Lazy<string[]>(()=>ImportAndSave(entry.Key,spec)));
                    var model=TextModelFile.Load(cache.Value[precision=="fp32"?0:1]);
                    var metadata=reference["metadata"];
                    var profile=new MultimodalBackboneProfile(1024,metadata.Value<int>("image_token_id"),metadata.Value<int>("audio_token_id"),
                        0,spec.Value<int>("audio_feature_count"),metadata.Value<int>("vocabulary_size"),metadata.Value<int>("pad_token_id"));
                    using(var backbone=new MultimodalBackbone(model,profile,backend))
                    {
                        var worker=(Worker)typeof(MultimodalBackbone).GetField("worker",BindingFlags.Instance|BindingFlags.NonPublic).GetValue(backbone);
                        const BindingFlags flags=BindingFlags.Instance|BindingFlags.NonPublic;
                        var resident=(HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallback",flags).GetValue(worker);
                        var cpu=resident.Intersect(model.layers.SelectMany(layer=>layer.outputs).Concat(model.inputs.Where(input=>input.name=="audio_features").Select(input=>input.index))).ToArray();
                        var flush=(HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallbackShouldFlushGPU",flags).GetValue(worker);
                        condition["cpu_computation_or_nonempty_feature_inputs"]=new JArray(cpu);condition["readback_flush"]=new JArray(flush);
                        if(backend==BackendType.GPUCompute){Assert.That(cpu,Is.Empty);Assert.That(flush,Is.Empty);}
                        foreach(var row in ((JArray)reference["cases"]).Where(row=>row.Value<string>("profile")==entry.Key))
                        {
                            condition["phase"]="evaluate_"+row.Value<string>("id");Save();
                            using var image=new Tensor<float>(new TensorShape(0,512));
                            using var audio=new Tensor<float>(new TensorShape(profile.AudioFeatureCount,512),Floats(Resolve(row.Value<string>("features_file"))));
                            if(backend==BackendType.GPUCompute)ComputeTensorData.Pin(audio);
                            var actual=backbone.Embed(row["input_ids"].ToObject<int[]>(),row["attention_mask"].ToObject<int[]>(),image,audio);
                            Assert.That(worker.PeekOutput().backendType,Is.EqualTo(backend));Assert.That(audio.backendType,Is.EqualTo(backend));
                            var result=AudioBackboneReferenceGuard.CompareVectors(actual,row["embedding"].ToObject<float[]>(),precision=="float16");
                            result["id"]=row["id"];result["profile"]=entry.Key;result["actual_backend"]=worker.PeekOutput().backendType.ToString();
                            ((JArray)condition["cases"]).Add(result);Save();
                        }
                    }
                    model=null;GC.Collect();GC.WaitForPendingFinalizers();
                }
                Assert.That(((JArray)condition["cases"]).Count,Is.EqualTo(6));
                Assert.That(((JArray)condition["cases"]).All(row=>row.Value<bool>("success")),Is.True,"Every native-feature final vector must meet the original final-vector criterion.");
                condition["success"]=true;condition["phase"]="completed";
            }
            catch(Exception error){condition["error"]=error.ToString();throw;}finally{Save();}
        }
        string[] ImportAndSave(string id,JObject spec)
        {
            report["import_phase"]=id;Save();var assetPath="Assets/AudioFinalGenerated/"+id+".pt2";
            Directory.CreateDirectory(Path.GetDirectoryName(assetPath));File.Copy(Resolve(spec.Value<string>("graph")),assetPath,true);
            AssetDatabase.ImportAsset(assetPath,ImportAssetOptions.ForceSynchronousImport);
            var asset=AssetDatabase.LoadAssetAtPath<ModelAsset>(assetPath);Assert.That(asset,Is.Not.Null,"Import the real zero-image profile.");
            var model=ModelLoader.Load(asset);var fp32=Path.Combine(root,id+"-fp32.sentis");var half=Path.Combine(root,id+"-float16.sentis");
            try{TextModelFile.Save(model,fp32);ModelQuantizer.QuantizeWeights(QuantizationType.Float16,ref model);TextModelFile.Save(model,half);}
            finally{model=null;Resources.UnloadAsset(asset);GC.Collect();GC.WaitForPendingFinalizers();}
            return new[] {fp32,half};
        }
        [OneTimeTearDown]public void Complete()
        {if(report==null)return;report["success"]=conditions.Count==4&&conditions.All(row=>row.Value<bool>("success"));report["completed_utc"]=DateTime.UtcNow.ToString("O");Save();}
        string Resolve(string name)
        {var path=Path.GetFullPath(Path.Combine(root,name));Assert.That(path.StartsWith(root+Path.DirectorySeparatorChar,StringComparison.Ordinal),Is.True);return path;}
        void Audit(string name)
        {var spec=validation["files"][name];Assert.That(spec,Is.Not.Null);var path=Resolve(name);Assert.That(new FileInfo(path).Length,Is.EqualTo(spec.Value<long>("bytes")));Assert.That(Hash(path),Is.EqualTo(spec.Value<string>("sha256")));}
        static float[] Floats(string path)
        {var raw=File.ReadAllBytes(path);Assert.That(raw.Length%4,Is.Zero);var values=new float[raw.Length/4];Buffer.BlockCopy(raw,0,values,0,raw.Length);return values;}
        static string Hash(string path)
        {using var stream=File.OpenRead(path);using var sha=SHA256.Create();return BitConverter.ToString(sha.ComputeHash(stream)).Replace("-","").ToLowerInvariant();}
        void Save(){Directory.CreateDirectory(Path.GetDirectoryName(resultPath));File.WriteAllText(resultPath,report.ToString());}
    }
}
