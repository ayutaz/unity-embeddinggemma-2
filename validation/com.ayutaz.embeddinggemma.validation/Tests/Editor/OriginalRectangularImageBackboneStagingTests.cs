using System;
using System.IO;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEditor;
using UnityEngine;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class OriginalRectangularImageBackboneStagingTests
    {
        [Test]
        public void OriginalRectangularImageBackboneCanBeSavedAndReloadedWithoutInference()
        {
            var root=Path.GetFullPath(Path.Combine(Application.dataPath,"../../numerical-acceptance-state/il11830367896"));
            var saved=Path.Combine(root,"landscape-fp32.sentis");var output=Path.Combine(root,"landscape-sentis-staging.json");
            Assert.That(File.Exists(saved)||File.Exists(output),Is.False,"Never overwrite fixed staging evidence.");
            const string originalSha="56955cba4fba156149786fdaa90bfd045021085090f08398c13ace603efe4f20";
            Assert.That(QualifiedImageBackboneSquare.Hash(Path.Combine(Application.dataPath,"GpuImageBackboneRectangular/backbone.pt2")),Is.EqualTo(originalSha));
            var asset=AssetDatabase.LoadAssetAtPath<ModelAsset>("Assets/GpuImageBackboneRectangular/backbone.pt2");Assert.That(asset,Is.Not.Null);
            var model=ModelLoader.Load(asset);QualifiedImageBackboneSquare.Validate(model,252);var coefficients=ModelCoefficientHash.Compute(model,out _);
            TextModelFile.Save(model,saved);Assert.That(ModelCoefficientHash.Compute(model,out _),Is.EqualTo(coefficients));
            var reloaded=TextModelFile.Load(saved);QualifiedImageBackboneSquare.Validate(reloaded,252);Assert.That(ModelCoefficientHash.Compute(reloaded,out _),Is.EqualTo(coefficients));
            File.WriteAllText(output,new JObject{["success"]=true,["source"]=Environment.GetEnvironmentVariable("EG_ACCEPTANCE_SOURCE"),
                ["scope"]="fixed_original_rectangular_image_backbone_serialization_only",["original_graph_sha256"]=originalSha,
                ["saved_sha256"]=QualifiedImageBackboneSquare.Hash(saved),["original_coefficients_sha256"]=coefficients,["coefficient_bytes_preserved"]=true,
                ["inference_executed"]=false,["gpu_validated"]=false,["end_to_end_validated"]=false}.ToString());
        }
    }
}
