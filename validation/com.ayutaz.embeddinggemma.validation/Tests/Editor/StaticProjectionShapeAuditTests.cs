using System;
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
    public sealed class StaticProjectionShapeAuditTests
    {
        [Test]
        public void AuditPartitionBoundaryShapesWithoutRunningInference()
        {
            var source=Environment.GetEnvironmentVariable("EG_ACCEPTANCE_SOURCE");
            var root=Path.Combine(Application.dataPath,"GpuAudioBoundaryShapeAudit");
            var output=Path.GetFullPath(Path.Combine(Application.dataPath,"../artifacts/partition-boundary-shapes-"+source+".json"));
            Assert.That(File.Exists(output),Is.False);
            var report=new JObject { ["success"]=false,["scope"]="fixed_boundary_metadata_only_no_inference",
                ["runtime_source"]=source,["gpu_validated"]=false,["full_encoder_validated"]=false,["end_to_end_validated"]=false };
            void Save(){Directory.CreateDirectory(Path.GetDirectoryName(output));File.WriteAllText(output,report.ToString());}
            Save();
            try
            {
                var locked=JObject.Parse(File.ReadAllText(Path.Combine(Application.dataPath,"../Packages/packages-lock.json")));
                foreach(var package in new[]{"com.ayutaz.embeddinggemma","com.ayutaz.embeddinggemma.validation"})
                    Assert.That(locked["dependencies"][package].Value<string>("hash"),Is.EqualTo(source));
                var manifest=JObject.Parse(File.ReadAllText(Path.Combine(root,"manifest.json")));
                Assert.That(manifest.Value<string>("source_graph_sha256"),Is.EqualTo("c050c2efa87d83aea45587b25f7b8d1d24dc471f33cc732dcf17d89588c70c13"));
                using(var stream=File.OpenRead(Path.Combine(root,"boundaries.onnx")))
                    Assert.That(PlayerFileHash.Compute(stream,out _),Is.EqualTo(manifest.Value<string>("audit_graph_sha256")));
                report["source_graph_sha256"]=manifest["source_graph_sha256"];report["audit_graph_sha256"]=manifest["audit_graph_sha256"];
                var model=ModelLoader.Load(AssetDatabase.LoadAssetAtPath<ModelAsset>("Assets/GpuAudioBoundaryShapeAudit/boundaries.onnx"));
                var analyser=typeof(Worker).Assembly.GetType("Unity.InferenceEngine.Compiler.Analyser.PartialInferenceAnalysis");
                Assert.That(analyser,Is.Not.Null);
                var context=analyser.GetMethod("InferModelPartialTensors").Invoke(null,new object[]{model});
                var get=context.GetType().GetMethod("GetPartialTensor");
                int[] Shape(int index)
                {
                    var partial=get.Invoke(context,new object[]{index});
                    var shape=(DynamicTensorShape)partial.GetType().GetProperty("shape").GetValue(partial);
                    Assert.That(shape.IsStatic(),Is.True,"Every actual partition boundary must resolve a fixed shape.");
                    var tensor=shape.ToTensorShape();return Enumerable.Range(0,tensor.rank).Select(axis=>tensor[axis]).ToArray();
                }
                var boundaries=new JObject();
                foreach(var input in model.inputs){Assert.That(input.dataType,Is.EqualTo(DataType.Float));boundaries[input.name]=JArray.FromObject(Shape(input.index));}
                foreach(var boundary in model.outputs)boundaries[boundary.name]=JArray.FromObject(Shape(boundary.index));
                report["boundaries"]=boundaries;Save();
                Assert.That(boundaries.Properties().Select(p=>p.Name),Is.EquivalentTo(manifest["boundary_names"].ToObject<string[]>()));
                Assert.That(boundaries.Count,Is.EqualTo(411));report["success"]=true;
            }
            catch(Exception error){report["error"]=error.ToString();throw;}
            finally { Save(); }
        }

        [Test]
        public void AuditPinnedStaticProjectionShapesWithoutRunningInference()
        {
            var source=Environment.GetEnvironmentVariable("EG_ACCEPTANCE_SOURCE");
            var output=Path.GetFullPath(Path.Combine(Application.dataPath,"../artifacts/static-projection-shapes-"+source+".json"));
            Assert.That(File.Exists(output),Is.False);
            var report=new JObject { ["success"]=false,["scope"]="fixed_model_metadata_only_no_inference",
                ["runtime_source"]=source,["gpu_validated"]=false,["full_encoder_validated"]=false,["end_to_end_validated"]=false };
            void Save(){Directory.CreateDirectory(Path.GetDirectoryName(output));File.WriteAllText(output,report.ToString());}
            Save();
            try
            {
                var locked=JObject.Parse(File.ReadAllText(Path.Combine(Application.dataPath,"../Packages/packages-lock.json")));
                foreach(var package in new[]{"com.ayutaz.embeddinggemma","com.ayutaz.embeddinggemma.validation"})
                    Assert.That(locked["dependencies"][package].Value<string>("hash"),Is.EqualTo(source));
                const string graphSha="c050c2efa87d83aea45587b25f7b8d1d24dc471f33cc732dcf17d89588c70c13";
                using(var stream=File.OpenRead(Path.Combine(Application.dataPath,"GpuAudioScalarMeanCandidate/frames-1119-valid-1119.onnx")))
                    Assert.That(PlayerFileHash.Compute(stream,out _),Is.EqualTo(graphSha));
                report["source_graph_sha256"]=graphSha;
                var model=ModelLoader.Load(AssetDatabase.LoadAssetAtPath<ModelAsset>("Assets/GpuAudioScalarMeanCandidate/frames-1119-valid-1119.onnx"));
                var analyser=typeof(Worker).Assembly.GetType("Unity.InferenceEngine.Compiler.Analyser.PartialInferenceAnalysis");
                Assert.That(analyser,Is.Not.Null);
                var context=analyser.GetMethod("InferModelPartialTensors").Invoke(null,new object[]{model});
                var get=context.GetType().GetMethod("GetPartialTensor");
                int[] Shape(int index)
                {
                    var partial=get.Invoke(context,new object[]{index});
                    var shape=(DynamicTensorShape)partial.GetType().GetProperty("shape").GetValue(partial);
                    Assert.That(shape.IsStatic(),Is.True,"The actual pinned model must resolve a fixed shape.");
                    var tensor=shape.ToTensorShape();return Enumerable.Range(0,tensor.rank).Select(axis=>tensor[axis]).ToArray();
                }
                var constants=model.constants.ToDictionary(c=>c.index);var rows=new JArray();
                foreach(var layer in model.layers.Where(l=>new[]{"MatMul","MatMul2D","Dense","DenseBatched"}.Contains(l.GetType().Name) && l.inputs.Length>=2 && l.outputs.Length==1))
                {
                    if(!constants.TryGetValue(layer.inputs[1],out var weight)||weight.shape.rank!=2)continue;
                    Assert.That(weight.dataType,Is.EqualTo(DataType.Float));
                    var bytes=(ArraySegment<byte>)typeof(Constant).GetField("array",BindingFlags.NonPublic|BindingFlags.Instance).GetValue(weight);
                    using var stream=new MemoryStream(bytes.Array,bytes.Offset,bytes.Count,false);
                    var sha=PlayerFileHash.Compute(stream,out _);
                    rows.Add(new JObject { ["kind"]=layer.GetType().Name,["weight_sha256"]=sha,["input_shape"]=JArray.FromObject(Shape(layer.inputs[0])),
                        ["output_shape"]=JArray.FromObject(Shape(layer.outputs.Single())),["weight_shape"]=JArray.FromObject(Shape(layer.inputs[1])) });
                }
                report["projections"]=rows;report["projection_count"]=rows.Count;Save();
                Assert.That(rows.Count,Is.EqualTo(135));report["success"]=true;
            }
            catch(Exception error){report["error"]=error.ToString();throw;}
            finally { Save(); }
        }
    }
}
