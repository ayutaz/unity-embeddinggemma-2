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
    /// <summary>Validation-only standard-FP32 templates; original learned scales are rebound by the existing graft.</summary>
    public static class QualifiedVisionMidpointRmsTemplates
    {
        public static (Model model,string sha) Load(string kind)
        {
            if(!new[]{"hidden","head-scaled","head-unscaled","tail252","tail256"}.Contains(kind))throw new ArgumentException("Unknown audited midpoint RMS template.",nameof(kind));
            var directory=Path.Combine(Application.dataPath,"GpuVisionMidpointRmsV2");
            string Hash(string path){using var stream=File.OpenRead(path);return PlayerFileHash.Compute(stream,out _);}
            var path=Path.Combine(directory,"midpoint-rms-reference.json");
            Assert.That(Hash(path),Is.EqualTo("9e87b0af2e504ce8fd08b5434fd4a778eb72031504a07f84ba543536e61b9209"));
            var reference=JObject.Parse(File.ReadAllText(path));Assert.That(reference.Value<bool>("original_reference_unchanged"),Is.True);
            var row=reference["models"].Single(value=>value.Value<string>("kind")==kind);var file=row.Value<string>("file");var sha=row.Value<string>("sha256");
            Assert.That(Hash(Path.Combine(directory,file)),Is.EqualTo(sha));
            var model=ModelLoader.Load(AssetDatabase.LoadAssetAtPath<ModelAsset>("Assets/GpuVisionMidpointRmsV2/"+file));
            Assert.That(model.inputs.Count,Is.EqualTo(1));Assert.That(model.outputs.Count,Is.EqualTo(1));
            Assert.That(model.inputs[0].shape.ToTensorShape(),Is.EqualTo(new TensorShape(row["shape"].Values<int>().ToArray())));
            return(model,sha);
        }
    }
}
