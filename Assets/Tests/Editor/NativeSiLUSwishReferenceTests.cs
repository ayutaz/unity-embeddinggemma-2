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
    public sealed class NativeSiLUSwishReferenceTests
    {
        string root, resultPath;
        JObject reference,report;
        JArray conditions;
        readonly Dictionary<string,string> saved=new Dictionary<string,string>();

        [OneTimeSetUp]
        public void AuditNativeReferenceAndImportStandardSwishOnce()
        {
            root=Path.GetFullPath(Path.Combine(Application.dataPath,"../artifacts/native-silu"));
            resultPath=Path.Combine(root,"../native-silu-results/results.json");
            conditions=new JArray(); report=new JObject { ["stage"]="native_silu_swish_probe_only",["success"]=false,
                ["audio_encoder_validated"]=false,["unity"]=Application.unityVersion,["conditions"]=conditions };
            Save();
            var validation=JObject.Parse(File.ReadAllText(Path.Combine(root,"silu-validation.json")));
            Assert.That(validation.Value<bool>("success"),Is.True);
            Assert.That(validation.Value<string>("source_commit"),Is.EqualTo(File.ReadAllText(Path.Combine(root,"expected-source.txt")).Trim()));
            foreach(var item in (JObject)validation["files"])
            {
                var path=Path.GetFullPath(Path.Combine(root,item.Key));
                Assert.That(path.StartsWith(root+Path.DirectorySeparatorChar,StringComparison.Ordinal),Is.True);
                Assert.That(new FileInfo(path).Length,Is.EqualTo(item.Value.Value<long>("bytes")));
                Assert.That(Hash(path),Is.EqualTo(item.Value.Value<string>("sha256")));
            }
            reference=JObject.Parse(File.ReadAllText(Path.Combine(root,"silu-reference.json")));
            Assert.That(reference["metadata"].Value<string>("source_commit"),Is.EqualTo(validation.Value<string>("source_commit")));
            report["reference_source"]=validation["source_commit"];
            Assert.That(((JArray)reference["cases"]).Count,Is.EqualTo(2));
            foreach(var item in (JArray)reference["cases"])
            {
                Assert.That(item.Value<bool>("native_saved_exact"),Is.True);
                var id=item.Value<string>("id");
                var assetPath="Assets/NativeSiLUGenerated/"+id+".onnx";
                Directory.CreateDirectory(Path.GetDirectoryName(assetPath));
                File.Copy(Path.Combine(root,item.Value<string>("onnx_graph")),assetPath,true);
                AssetDatabase.ImportAsset(assetPath,ImportAssetOptions.ForceSynchronousImport);
                var asset=AssetDatabase.LoadAssetAtPath<ModelAsset>(assetPath);
                Assert.That(asset,Is.Not.Null);
                var model=ModelLoader.Load(asset);
                Assert.That(model.layers.Count(layer=>layer.GetType().Name=="Swish"),Is.EqualTo(1),"Import must keep the native Sentis Swish layer.");
                var path=Path.Combine(root,id+".sentis"); TextModelFile.Save(model,path);saved.Add(id,path);
            }
        }

        [TestCase(BackendType.CPU)]
        [TestCase(BackendType.GPUCompute)]
        public void StandardSwishMatchesNativeVectorsAtTheFixedTolerance(BackendType backend)
        {
            var condition=new JObject { ["requested_backend"]=backend.ToString(),["success"]=false,["cases"]=new JArray() };
            conditions.Add(condition);Save();
            try
            {
                if(backend==BackendType.GPUCompute)Assert.That(SystemInfo.supportsComputeShaders,Is.True,"Do not skip GPU.");
                foreach(var item in (JArray)reference["cases"])
                {
                    var model=TextModelFile.Load(saved[item.Value<string>("id")]);
                    using var worker=new Worker(model,backend);
                    Assert.That(worker.backendType,Is.EqualTo(backend));
                    const BindingFlags flags=BindingFlags.Instance|BindingFlags.NonPublic;
                    var resident=(HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallback",flags).GetValue(worker);
                    var cpu=resident.Intersect(model.layers.SelectMany(layer=>layer.outputs).Concat(model.inputs.Select(input=>input.index))).ToArray();
                    var flush=(HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallbackShouldFlushGPU",flags).GetValue(worker);
                    if(backend==BackendType.GPUCompute){Assert.That(cpu,Is.Empty);Assert.That(flush,Is.Empty);}
                    var shape=new TensorShape(item["shape"].ToObject<int[]>());
                    using var input=new Tensor<float>(shape,item["input"].ToObject<float[]>());
                    if(backend==BackendType.GPUCompute)ComputeTensorData.Pin(input);
                    worker.Schedule(input);
                    var output=worker.PeekOutput() as Tensor<float>;
                    Assert.That(output.shape,Is.EqualTo(shape));Assert.That(output.backendType,Is.EqualTo(backend));
                    var values=output.DownloadToArray();var expected=item["expected"].ToObject<float[]>();
                    Assert.That(values.Length,Is.EqualTo(expected.Length));
                    int mismatches=0,exact=0;double maximum=0,ratio=0;
                    for(var index=0;index<values.Length;index++)
                    {
                        Assert.That(float.IsNaN(values[index])||float.IsInfinity(values[index]),Is.False);
                        var difference=Math.Abs((double)values[index]-expected[index]);var bound=1e-5+1e-4*Math.Abs((double)expected[index]);
                        maximum=Math.Max(maximum,difference);ratio=Math.Max(ratio,difference/bound);
                        if(difference>bound)mismatches++;if(values[index]!=expected[index])exact++;
                    }
                    ((JArray)condition["cases"]).Add(new JObject { ["id"]=item["id"],["actual_backend"]=output.backendType.ToString(),
                        ["mismatched_elements"]=mismatches,["exact_value_mismatches"]=exact,["maximum_absolute_error"]=maximum,
                        ["maximum_tolerance_ratio"]=ratio,["cpu_computation_or_inputs"]=new JArray(cpu),["readback_flush"]=new JArray(flush) });
                    Save();Assert.That(mismatches,Is.Zero);
                }
                condition["success"]=true;
            }
            catch(Exception error){condition["error"]=error.ToString();throw;}
            finally{Save();}
        }
        [OneTimeTearDown]
        public void Complete()
        {
            if(report==null)return;
            report["success"]=conditions.Count==2&&conditions.All(item=>item.Value<bool>("success"));
            report["completed_utc"]=DateTime.UtcNow.ToString("O");Save();
        }
        static string Hash(string path)
        {
            using var file=File.OpenRead(path);using var hash=SHA256.Create();
            return BitConverter.ToString(hash.ComputeHash(file)).Replace("-","").ToLowerInvariant();
        }
        void Save(){Directory.CreateDirectory(Path.GetDirectoryName(resultPath));File.WriteAllText(resultPath,report.ToString());}
    }
}
