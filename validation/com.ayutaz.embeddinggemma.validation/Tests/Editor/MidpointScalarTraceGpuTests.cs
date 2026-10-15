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
    public sealed class MidpointScalarTraceGpuTests
    {
        [Test]
        public void QualifiedQueryReciprocalAndNeighborsMustMatchScalarOracleOnGpu()
        {
            Assert.That(SystemInfo.supportsComputeShaders,Is.True,"Actual GPU required; never skip.");
            var source=Environment.GetEnvironmentVariable("EG_ACCEPTANCE_SOURCE");
            var root=Path.Combine(Application.dataPath,"GpuMidpointScalarTrace");
            var output=Path.GetFullPath(Path.Combine(Application.dataPath,"../artifacts/midpoint-scalar-trace-gpu-"+source+".json"));
            Assert.That(File.Exists(output),Is.False);
            var report=new JObject{["success"]=false,["runtime_source"]=source,["scope"]="qualified_query_statistic_and_32_neighbors_scalar_trace_only",
                ["full_encoder_validated"]=false,["end_to_end_validated"]=false};
            void Save(){Directory.CreateDirectory(Path.GetDirectoryName(output));File.WriteAllText(output,report.ToString());}
            string Hash(string path){using var stream=File.OpenRead(path);return PlayerFileHash.Compute(stream,out _);}
            Save();
            try
            {
                var packages=JObject.Parse(File.ReadAllText(Path.Combine(Application.dataPath,"../Packages/packages-lock.json")));
                foreach(var package in new[]{"com.ayutaz.embeddinggemma","com.ayutaz.embeddinggemma.validation"})Assert.That(packages["dependencies"][package].Value<string>("hash"),Is.EqualTo(source));
                Assert.That(Hash(Path.Combine(root,"trace-reference.json")),Is.EqualTo("bfad44cce4f6ac1163183c0daa1443b93a466bf3531f6d8676c7fa53f6fcf32d"));
                var reference=JObject.Parse(File.ReadAllText(Path.Combine(root,"trace-reference.json")));
                foreach(var entry in ((JObject)reference["files"]).Properties())Assert.That(Hash(Path.Combine(root,entry.Value.Value<string>("file"))),Is.EqualTo(entry.Value.Value<string>("sha256")));
                var values=Floats(Path.Combine(root,"input.bin"));var expected=Floats(Path.Combine(root,"expected.bin"));
                var model=ModelLoader.Load(AssetDatabase.LoadAssetAtPath<ModelAsset>("Assets/GpuMidpointScalarTrace/trace.onnx"));
                using var pipeline=new GpuProjectionPipeline(new[]{model},Array.Empty<string>(),Array.Empty<string>(),Array.Empty<IGpuTensorOperation>());
                using var input=new Tensor<float>(new TensorShape(1,65),values);ComputeTensorData.Pin(input);
                using var result=pipeline.Execute(input);Assert.That(result.backendType,Is.EqualTo(BackendType.GPUCompute));
                var actual=result.DownloadToArray();Assert.That(actual.Length,Is.EqualTo(65*reference["names"].Count()));
                var bytes=new byte[actual.Length*4];Buffer.BlockCopy(actual,0,bytes,0,bytes.Length);File.WriteAllBytes(Path.ChangeExtension(output,"bin"),bytes);
                report["capture_sha256"]=Hash(Path.ChangeExtension(output,"bin"));report["names"]=reference["names"].DeepClone();
                var traces=new JArray();report["center_traces"]=traces;
                for(var i=0;i<reference["names"].Count();i++)traces.Add(new JObject{["name"]=reference["names"][i].Value<string>(),["value"]=actual[i*65+32],["bits"]=BitConverter.SingleToInt32Bits(actual[i*65+32])});
                int differences=Enumerable.Range(0,65).Count(i=>BitConverter.SingleToInt32Bits(actual[i])!=BitConverter.SingleToInt32Bits(expected[i]));
                report["final_bit_differences"]=differences;report["actual_backend"]="GPUCompute";report["cpu_fallback_count"]=0;report["intermediate_readback_count"]=0;report["phase"]="completed";Save();
                Assert.That(differences,Is.Zero,"Keep independent two-step FP32 sqrt / reciprocal rounding at the actual query statistic.");report["success"]=true;
            }
            finally{Save();}
        }
        static float[] Floats(string path){var bytes=File.ReadAllBytes(path);var data=new float[bytes.Length/4];Buffer.BlockCopy(bytes,0,data,0,bytes.Length);return data;}
    }
}
