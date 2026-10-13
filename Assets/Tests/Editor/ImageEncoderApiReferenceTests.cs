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
    // Acceptance of the TDD-tested public API against cached real models; never re-import weights here.
    public sealed class ImageEncoderApiReferenceTests
    {
        const string RuntimeSource="10fa49cd4a7db231cb3e27434dfabaf47b54fb1f";
        const string ReferenceSource="e2a6dc01fbca8c9be1f3bbe875325ad338de0ca0";
        static string Root => Path.GetFullPath(Path.Combine(Application.dataPath,"../artifacts/vision-encoder"));
        static string Output => Path.GetFullPath(Path.Combine(Application.dataPath,"../artifacts/image-encoder-api-real/results.json"));
        JObject reference,report,prior;
        JArray conditions;
        readonly Dictionary<string,string> audited=new Dictionary<string,string>();
        [OneTimeSetUp] public void AuditFixedReferencesAndSavedModels()
        {
            conditions=new JArray(); report=new JObject {
                ["scope"]="public_image_encoder_api_real_features_only",["success"]=false,["expected_conditions"]=12,
                ["runtime_source"]=RuntimeSource,["reference_source"]=ReferenceSource,["runtime_overlay"]=false,
                ["preprocessing_validated"]=false,["end_to_end_validated"]=false,
                ["unity"]=Application.unityVersion,["sentis"]="2.6.1",["gpu"]=SystemInfo.graphicsDeviceName,
                ["graphics_api"]=SystemInfo.graphicsDeviceType.ToString(),["started_utc"]=DateTime.UtcNow.ToString("O"),["conditions"]=conditions}; Save();
            var locked=JObject.Parse(File.ReadAllText(Path.Combine(Application.dataPath,"../Packages/packages-lock.json")));
            Assert.That(locked["dependencies"]["com.ayutaz.embeddinggemma"].Value<string>("hash"),Is.EqualTo(RuntimeSource));
            Audit(Path.Combine(Root,"vision-reference.json"),"3dfccd1a487ce04df3f8d640483ca045f7a7a56231bc1096ad0f845c923a29c6");
            Audit(Path.Combine(Root,"vision-validation.json"),"ab1aa775b553f33a5e2892abed2cdcef7768e35fbad6fa5af9ec3286a7ba90ba");
            var priorPath=Path.GetFullPath(Path.Combine(Application.dataPath,"../artifacts/m3-vision/results.json"));
            Audit(priorPath,"720e4b27049d04005108c75525ce7aa2c895b33cdcfc2021915d3c3440e9ae91");
            prior=JObject.Parse(File.ReadAllText(priorPath)); Assert.That(prior.Value<string>("reference_source"),Is.EqualTo(ReferenceSource));
            reference=JObject.Parse(File.ReadAllText(Path.Combine(Root,"vision-reference.json")));
            Assert.That(reference["metadata"].Value<string>("source_commit"),Is.EqualTo(ReferenceSource));
            Assert.That(reference["metadata"].Value<string>("model_revision"),Is.EqualTo("914f7f89142e33e77833254d9c9b90c3cef7303b"));
            var validation=JObject.Parse(File.ReadAllText(Path.Combine(Root,"vision-validation.json")));
            Assert.That(validation.Value<bool>("success"),Is.True); Assert.That(validation.Value<int>("case_count"),Is.EqualTo(6));
            foreach(var file in (JObject)validation["files"])
            {var path=Resolve(file.Key); Assert.That(new FileInfo(path).Length,Is.EqualTo(file.Value.Value<long>("bytes"))); Audit(path,file.Value.Value<string>("sha256"));}
            report["reference_audited"]=true; Save();
        }
        [TestCase("square","fp32",BackendType.CPU)] [TestCase("square","fp32",BackendType.GPUCompute)]
        [TestCase("square","float16",BackendType.CPU)] [TestCase("square","float16",BackendType.GPUCompute)]
        [TestCase("landscape","fp32",BackendType.CPU)] [TestCase("landscape","fp32",BackendType.GPUCompute)]
        [TestCase("landscape","float16",BackendType.CPU)] [TestCase("landscape","float16",BackendType.GPUCompute)]
        [TestCase("portrait","fp32",BackendType.CPU)] [TestCase("portrait","fp32",BackendType.GPUCompute)]
        [TestCase("portrait","float16",BackendType.CPU)] [TestCase("portrait","float16",BackendType.GPUCompute)]
        [Timeout(1200000)]
        public void OwnedApiSoftTokensMatchEveryNativeFeature(string profileName,string precision,BackendType backend)
        {
            var condition=new JObject {["profile"]=profileName,["precision"]=precision,["requested_backend"]=backend.ToString(),
                ["success"]=false,["phase"]="audit_saved_model",["cases"]=new JArray()}; conditions.Add(condition); Save();
            Model model=null;
            try
            {
                if(backend==BackendType.GPUCompute) Assert.That(SystemInfo.supportsComputeShaders,Is.True,"Actual GPU required; never skip.");
                var original=((JArray)prior["conditions"]).Where(row=>row.Value<string>("profile")==profileName&&row.Value<string>("precision")==precision).ToArray();
                Assert.That(original.Length,Is.EqualTo(2)); Assert.That(original[0].Value<string>("saved_sha256"),Is.EqualTo(original[1].Value<string>("saved_sha256")));
                var modelPath=Resolve(profileName+"-"+precision+".sentis"); Assert.That(new FileInfo(modelPath).Length,Is.EqualTo(original[0].Value<long>("saved_bytes")));
                Audit(modelPath,original[0].Value<string>("saved_sha256")); condition["saved_sha256"]=original[0]["saved_sha256"];
                var positions=Read<int>(Resolve(reference["profiles"][profileName].Value<string>("positions")));
                Assert.That(positions.Length,Is.EqualTo(5040)); var width=(positions.Where((value,index)=>index%2==0).Max()+1)*16;
                var height=(positions.Where((value,index)=>index%2==1).Max()+1)*16; var profile=new ImageEncoderProfile(width,height);
                Assert.That(profile.FeatureCount,Is.EqualTo(reference["profiles"][profileName].Value<int>("feature_count")));
                model=TextModelFile.Load(modelPath); condition["model_inputs"]=new JArray(model.inputs.Select(input=>input.name));
                condition["largest_constant_bytes"]=model.constants.Max(value=>value.lengthBytes); condition["phase"]="construct_api"; Save();
                using(var encoder=new ImageFeatureEncoder(model,backend,profile))
                {
                    var worker=(Worker)typeof(ImageFeatureEncoder).GetField("worker",BindingFlags.NonPublic|BindingFlags.Instance).GetValue(encoder);
                    Assert.That(worker.backendType,Is.EqualTo(backend)); condition["actual_worker_backend"]=worker.backendType.ToString();
                    var resident=(HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallback",BindingFlags.NonPublic|BindingFlags.Instance).GetValue(worker);
                    var actualCpu=resident.Intersect(model.layers.SelectMany(layer=>layer.outputs).Concat(model.inputs.Select(input=>input.index))).ToArray();
                    var flush=(HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallbackShouldFlushGPU",BindingFlags.NonPublic|BindingFlags.Instance).GetValue(worker);
                    condition["cpu_computation_or_input_count"]=actualCpu.Length; condition["readback_flush_count"]=flush.Count;
                    if(backend==BackendType.GPUCompute){Assert.That(actualCpu,Is.Empty);Assert.That(flush,Is.Empty);}
                    foreach(var row in ((JArray)reference["cases"]).Where(row=>row.Value<string>("profile")==profileName))
                    {
                        condition["phase"]="evaluate_"+row.Value<string>("id"); Save();
                        var tensor=new Tensor<float>(new TensorShape(1,2520,768),Read<float>(Resolve(row.Value<string>("pixels"))));
                        if(backend==BackendType.GPUCompute) ComputeTensorData.Pin(tensor);
                        using var input=(ImagePreprocessingResult)typeof(ImagePreprocessingResult).GetConstructors(BindingFlags.NonPublic|BindingFlags.Instance).Single().Invoke(
                            new object[] {tensor,width,height,profile.FeatureCount,positions,new List<Tensor> {tensor},new List<ComputeBuffer>()});
                        using var encoded=encoder.Encode(input); Assert.That(encoded.SoftTokens.backendType,Is.EqualTo(backend));
                        Assert.That(encoded.SoftTokens.shape,Is.EqualTo(new TensorShape(profile.FeatureCount,512)));
                        var actual=encoded.SoftTokens.DownloadToArray(); var expected=row["features"].ToObject<float[][]>();
                        double minimumCosine=1,maximumAbsolute=0,maximumRatio=0;
                        for(var feature=0;feature<profile.FeatureCount;feature++)
                        {
                            Assert.That(expected[feature].Length,Is.EqualTo(512)); double dot=0,leftSquared=0,rightSquared=0;
                            for(var channel=0;channel<512;channel++)
                            {
                                var left=actual[feature*512+channel];var right=expected[feature][channel];
                                Assert.That(float.IsNaN(left)||float.IsInfinity(left),Is.False);
                                dot+=(double)left*right;leftSquared+=(double)left*left;rightSquared+=(double)right*right;
                                var difference=Math.Abs((double)left-right);maximumAbsolute=Math.Max(maximumAbsolute,difference);
                                maximumRatio=Math.Max(maximumRatio,difference/(1e-5+1e-4*Math.Abs(right)));
                            }
                            Assert.That(leftSquared*rightSquared,Is.GreaterThan(0));minimumCosine=Math.Min(minimumCosine,dot/Math.Sqrt(leftSquared*rightSquared));
                        }
                        var passed=minimumCosine>=(precision=="fp32"?0.999:0.99)&&(precision!="fp32"||maximumRatio<=1);
                        ((JArray)condition["cases"]).Add(new JObject {["id"]=row.Value<string>("id"),["success"]=passed,["minimum_cosine"]=minimumCosine,
                            ["maximum_absolute_error"]=maximumAbsolute,["maximum_fp32_tolerance_ratio"]=maximumRatio,["actual_backend"]=encoded.SoftTokens.backendType.ToString()}); Save();
                    }
                }
                var cases=(JArray)condition["cases"]; Assert.That(cases.Count,Is.EqualTo(2));
                Assert.That(cases.All(row=>row.Value<bool>("success")),Is.True,"Keep the original FP32 elementwise and Float16 cosine gates.");
                condition["success"]=true;condition["phase"]="completed";
            }
            catch(Exception error){condition["error"]=error.ToString();throw;}
            finally{model=null;GC.Collect();GC.WaitForPendingFinalizers();Save();}
        }
        [OneTimeTearDown] public void CompleteReport()
        {report["completed_utc"]=DateTime.UtcNow.ToString("O");report["success"]=conditions.Count==12&&conditions.All(row=>row.Value<bool>("success"));Save();}
        void Audit(string path,string expected)
        {
            if(audited.TryGetValue(path,out var existing)){Assert.That(existing,Is.EqualTo(expected));return;}
            var type=AppDomain.CurrentDomain.GetAssemblies().Select(assembly=>assembly.GetType("EmbeddingGemma.Validation.PlayerFileHash")).FirstOrDefault(value=>value!=null);
            Assert.That(type,Is.Not.Null,"Use the unchanged audited Windows CNG hash helper.");
            using var stream=File.OpenRead(path);var args=new object[] {stream,null,null};
            var actual=(string)type.GetMethod("Compute").Invoke(null,args);Assert.That(actual,Is.EqualTo(expected),path);
            audited.Add(path,actual);report["hash_backend"]=args[1].ToString();Save();
        }
        static string Resolve(string relative)
        {var path=Path.GetFullPath(Path.Combine(Root,relative));Assert.That(path.StartsWith(Root+Path.DirectorySeparatorChar,StringComparison.Ordinal),Is.True);return path;}
        static T[] Read<T>(string path) where T:struct
        {var bytes=File.ReadAllBytes(path);Assert.That(bytes.Length%4,Is.Zero);var result=new T[bytes.Length/4];Buffer.BlockCopy(bytes,0,result,0,bytes.Length);return result;}
        void Save(){Directory.CreateDirectory(Path.GetDirectoryName(Output));File.WriteAllText(Output,report.ToString());}
    }
}
