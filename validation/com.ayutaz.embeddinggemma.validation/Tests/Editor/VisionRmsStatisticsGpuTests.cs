using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using EmbeddingGemma.Tests;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using UnityEditor;
using Unity.InferenceEngine;
using UnityEngine;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class VisionRmsStatisticsGpuTests
    {
        [Test]
        public void RefinedVisionRmsStatisticsPreserveNormalGpuOutput()
        {
            var source=Environment.GetEnvironmentVariable("EG_ACCEPTANCE_SOURCE");
            var output=Path.GetFullPath(Path.Combine(Application.dataPath,"../artifacts/vision-rms-statistics-gpu-"+source+".json"));Assert.That(File.Exists(output),Is.False);
            var report=new JObject{["success"]=false,["scope"]="vision_rms_small_statistics_gpu_observation_only",["runtime_source"]=source,
                ["native_reference_validated"]=false,["full_encoder_validated"]=false,["end_to_end_validated"]=false,["diagnostic_readbacks_only"]=true};
            void Save(){Directory.CreateDirectory(Path.GetDirectoryName(output));File.WriteAllText(output,report.ToString());}
            void Check(string path,string sha){using var stream=File.OpenRead(path);Assert.That(PlayerFileHash.Compute(stream,out _),Is.EqualTo(sha));}
            Save();
            try
            {
                Assert.That(SystemInfo.supportsComputeShaders,Is.True,"Actual GPU required; never skip.");
                var locked=JObject.Parse(File.ReadAllText(Path.Combine(Application.dataPath,"../Packages/packages-lock.json")));
                foreach(var package in new[]{"com.ayutaz.embeddinggemma","com.ayutaz.embeddinggemma.validation"})Assert.That(locked["dependencies"][package].Value<string>("hash"),Is.EqualTo(source));
                Check(Path.Combine(Application.dataPath,"GpuVisionFirstRmsCandidate/first-rms.onnx"),"724605434efc0b70bb28b59b5dc639f8b0aeb765a4170ec782397506f6549976");
                var model=ModelLoader.Load(AssetDatabase.LoadAssetAtPath<ModelAsset>("Assets/GpuVisionFirstRmsCandidate/first-rms.onnx"));
                var analysis=typeof(Worker).Assembly.GetType("Unity.InferenceEngine.Compiler.Analyser.PartialInferenceAnalysis");
                var context=analysis.GetMethod("InferModelPartialTensors").Invoke(null,new object[]{model});var get=context.GetType().GetMethod("GetPartialTensor");
                var selected=new List<(Layer layer,TensorShape shape)>();
                foreach(var layer in model.layers)
                {
                    if(!new[]{"ScalarMad","Sqrt","Div","Reciprocal"}.Contains(layer.GetType().Name))continue;
                    var partial=get.Invoke(context,new object[]{layer.outputs[0]});var shape=(DynamicTensorShape)partial.GetType().GetProperty("shape").GetValue(partial);
                    if(shape.IsStatic()&&shape.ToTensorShape().length==2520)selected.Add((layer,shape.ToTensorShape()));
                }
                Assert.That(selected.Count,Is.GreaterThan(0));
                var inputPath=Path.GetFullPath(Path.Combine(Application.dataPath,"../../k/artifacts/vision-diagnostics/native-stage-0.bin"));Check(inputPath,"bc4c19c9aaf9688801c99ed073fdde22f0daf6b822dc2c9c4b7a3f7c9aded636");
                using var input=new Tensor<float>(new TensorShape(1,2520,768),Floats(inputPath));ComputeTensorData.Pin(input);
                var observed=WorkerObservation.ObserveIndexes(model,BackendType.GPUCompute,input,new HashSet<int>(selected.Select(row=>row.layer.outputs[0])));
                Assert.That(observed.FinalMatchesBaselineBits,Is.True);
                // Prior capture is independently hash-pinned rather than inferred from a current source label.
                var baselineBytes=new byte[observed.Final.Length*4];Buffer.BlockCopy(observed.Final,0,baselineBytes,0,baselineBytes.Length);
                using(var stream=new MemoryStream(baselineBytes))Assert.That(PlayerFileHash.Compute(stream,out _),Is.EqualTo("687a68f9956e6c1587cdb0fd5359dd75840355ecb80c6f74c6819c90377109c9"));
                var folder=Path.ChangeExtension(output,null)+"-captures";Assert.That(Directory.Exists(folder),Is.False);Directory.CreateDirectory(folder);
                var rows=new JArray();report["statistics"]=rows;
                foreach(var entry in selected)
                {
                    var snapshot=observed.Snapshots.Single(value=>value.TensorIndex==entry.layer.outputs[0]);Assert.That(snapshot.Backend,Is.EqualTo(BackendType.GPUCompute));
                    var row=new JObject{["tensor_index"]=snapshot.TensorIndex,["type"]=entry.layer.GetType().Name,["shape"]=JArray.FromObject(Enumerable.Range(0,entry.shape.rank).Select(i=>entry.shape[i]).ToArray())};
                    foreach(var field in entry.layer.GetType().GetFields().Where(field=>field.FieldType==typeof(float)))row[field.Name]=Convert.ToDouble(field.GetValue(entry.layer));
                    var bytes=new byte[snapshot.Values.Length*4];Buffer.BlockCopy(snapshot.Values,0,bytes,0,bytes.Length);var path=Path.Combine(folder,"tensor-"+snapshot.TensorIndex+".bin");File.WriteAllBytes(path,bytes);
                    using(var stream=File.OpenRead(path))row["sha256"]=PlayerFileHash.Compute(stream,out _);row["file"]=path;rows.Add(row);
                }
                report["normal_final_matches_observed_bits"]=true;report["actual_backend"]="GPUCompute";report["success"]=true;
            }
            catch(Exception error){report["error"]=error.ToString();throw;}
            finally{Save();}
        }
        static float[] Floats(string path){var bytes=File.ReadAllBytes(path);var data=new float[bytes.Length/4];Buffer.BlockCopy(bytes,0,data,0,bytes.Length);return data;}
    }
}
