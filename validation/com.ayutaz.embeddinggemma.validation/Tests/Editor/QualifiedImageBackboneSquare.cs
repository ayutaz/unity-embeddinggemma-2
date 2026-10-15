using System;
using System.IO;
using System.Linq;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEditor;
using UnityEngine;

namespace EmbeddingGemma.Validation.Tests
{
    public static class QualifiedImageBackboneSquare
    {
        public const string OriginalGraphSha="ad45dd7d0b7be3f0d94f0391b674d8e95d09ea3f72edbf2408e9fae89ac7f03f";
        public static string Root=>Path.GetFullPath(Path.Combine(Application.dataPath,"../../numerical-acceptance-state/ib11830367896"));
        public static string Hash(string path){using var stream=File.OpenRead(path);return PlayerFileHash.Compute(stream,out _);}
        public static void Validate(Model model,int imageFeatureCount=256)
        {
            Assert.That(model.inputs.Count,Is.EqualTo(4));Assert.That(model.outputs.Count,Is.EqualTo(1));
            foreach(var row in new[]{("input_ids",DataType.Int,new TensorShape(1,1024)),("attention_mask",DataType.Int,new TensorShape(1,1024)),
                ("image_features",DataType.Float,new TensorShape(imageFeatureCount,512)),("audio_features",DataType.Float,new TensorShape(0,512))})
            {
                var input=model.inputs.Single(value=>value.name==row.Item1);Assert.That(input.dataType,Is.EqualTo(row.Item2));Assert.That(input.shape.ToTensorShape(),Is.EqualTo(row.Item3));
            }
        }
        public static Model Load()
        {
            var receiptPath=Path.Combine(Root,"square-sentis-staging.json");
            Assert.That(Hash(receiptPath),Is.EqualTo("05238e775d41b282120e98160b71d7afd8334d9c4d748b377b3b3afafff3a600"));
            var receipt=JObject.Parse(File.ReadAllText(receiptPath));
            Assert.That(receipt.Value<bool>("success"),Is.True);Assert.That(receipt.Value<string>("original_graph_sha256"),Is.EqualTo(OriginalGraphSha));
            Assert.That(receipt.Value<bool>("inference_executed"),Is.False);
            Assert.That(receipt.Value<string>("saved_sha256"),Is.EqualTo("f6f4ba9d036e70c3d26b5932daca3facecbc0c45753a66b2da72e7b8f02519ce"));
            Assert.That(receipt.Value<string>("original_coefficients_sha256"),Is.EqualTo("c234c173d9a1f15fbe4bf7a3adbb7e2e5f17fdb8c6f4b85c0befe4d5fb2b097b"));
            var path=Path.Combine(Root,"square-fp32.sentis");Assert.That(Hash(path),Is.EqualTo(receipt.Value<string>("saved_sha256")));
            var model=TextModelFile.Load(path);Validate(model);return model;
        }
    }
    public sealed class OriginalImageBackboneStagingTests
    {
        [Test]
        public void OriginalSquareImageBackboneCanBeSavedAndReloadedWithoutInference()
        {
            var source=Environment.GetEnvironmentVariable("EG_ACCEPTANCE_SOURCE");
            var root=QualifiedImageBackboneSquare.Root;var saved=Path.Combine(root,"square-fp32.sentis");var output=Path.Combine(root,"square-sentis-staging.json");
            Assert.That(File.Exists(saved)||File.Exists(output),Is.False,"Never overwrite fixed model staging evidence.");
            var original=Path.Combine(Application.dataPath,"GpuImageBackboneSquare/backbone.pt2");
            Assert.That(QualifiedImageBackboneSquare.Hash(original),Is.EqualTo(QualifiedImageBackboneSquare.OriginalGraphSha));
            var asset=AssetDatabase.LoadAssetAtPath<ModelAsset>("Assets/GpuImageBackboneSquare/backbone.pt2");Assert.That(asset,Is.Not.Null);
            var model=ModelLoader.Load(asset);QualifiedImageBackboneSquare.Validate(model);
            var coefficients=ModelCoefficientHash.Compute(model,out _);TextModelFile.Save(model,saved);
            Assert.That(ModelCoefficientHash.Compute(model,out _),Is.EqualTo(coefficients));var reloaded=TextModelFile.Load(saved);QualifiedImageBackboneSquare.Validate(reloaded);
            Assert.That(ModelCoefficientHash.Compute(reloaded,out _),Is.EqualTo(coefficients));
            var report=new JObject{["success"]=true,["source"]=source,["scope"]="fixed_original_image_square_backbone_serialization_only",
                ["original_graph_sha256"]=QualifiedImageBackboneSquare.OriginalGraphSha,["saved_sha256"]=QualifiedImageBackboneSquare.Hash(saved),
                ["original_coefficients_sha256"]=coefficients,["coefficient_bytes_preserved"]=true,["inference_executed"]=false,["gpu_validated"]=false,["end_to_end_validated"]=false};
            File.WriteAllText(output,report.ToString());
        }
    }
}
