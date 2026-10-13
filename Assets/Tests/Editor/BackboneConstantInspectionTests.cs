using System;
using System.IO;
using System.Linq;
using System.Reflection;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEngine;

namespace EmbeddingGemma.Tests
{
    public sealed class BackboneConstantInspectionTests
    {
        [TestCase("fp32")] [TestCase("float16")]
        public void InspectAuditedBackboneWithoutGpuExecution(string precision)
        {
            var root=Path.GetFullPath(Path.Combine(Application.dataPath,"../artifacts/audio-final-backbone"));
            var receiptPath=Path.Combine(root,"saved-models-receipt.json");
            Assert.That(Hash(receiptPath),Is.EqualTo("f79df6aa24fed7f154598a8035bd6cb104dd9d6168db74901be636beb9b9a047"));
            var receipt=JObject.Parse(File.ReadAllText(receiptPath));
            var file="frames-1119-valid-1119-"+precision+".sentis"; var path=Path.Combine(root,file); var identity=receipt["saved_models"][file];
            Assert.That(identity,Is.Not.Null); Assert.That(new FileInfo(path).Length,Is.EqualTo(identity.Value<long>("bytes")));
            Assert.That(Hash(path),Is.EqualTo(identity.Value<string>("sha256")));
            var model=TextModelFile.Load(path);
            try
            {
                var constants=model.constants.OrderByDescending(value=>value.lengthBytes).Take(8).ToArray();
                Assert.That(constants.Length,Is.GreaterThan(0));
                var report=new JObject {["scope"]="saved_backbone_cpu_metadata_inspection_only",["precision"]=precision,
                    ["gpu_execution"]=false,["d3d12_failure_resolved"]=false,["saved_sha256"]=identity["sha256"],
                    ["unity"]=Application.unityVersion,["sentis"]="2.6.1",["completed_utc"]=DateTime.UtcNow.ToString("O"),
                    ["largest_constants"]=new JArray(constants.Select(value=>new JObject {["index"]=value.index,["bytes"]=value.lengthBytes,
                        ["shape"]=value.shape.ToString(),["data_type"]=value.dataType.ToString()})),
                    ["recorded_d3d12_upload_buffer_bytes"]=16777216,["recorded_d3d12_requested_bytes"]=536870912};
                var destination=Path.GetFullPath(Path.Combine(Application.dataPath,"../artifacts/backbone-constant-inspection"));
                Directory.CreateDirectory(destination);File.WriteAllText(Path.Combine(destination,precision+".json"),report.ToString());
            }
            finally {model=null;GC.Collect();GC.WaitForPendingFinalizers();}
        }
        static string Hash(string path)
        {
            var type=AppDomain.CurrentDomain.GetAssemblies().Select(assembly=>assembly.GetType("EmbeddingGemma.Validation.PlayerFileHash")).FirstOrDefault(value=>value!=null);
            Assert.That(type,Is.Not.Null);using var stream=File.OpenRead(path);var args=new object[] {stream,null,null};return (string)type.GetMethod("Compute").Invoke(null,args);
        }
    }
}
