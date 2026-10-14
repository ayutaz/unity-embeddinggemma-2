using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using EmbeddingGemma.Tests;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using UnityEditor;
using Unity.InferenceEngine;
using UnityEngine;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class PartitionedAudioGpuReferenceTests
    {
        [TestCase("silence")]
        [TestCase("impulse")]
        [TestCase("sine")]
        [TestCase("nyquist")]
        [TestCase("noise")]
        [TestCase("padded_noise")]
        public void FirstFeedForwardPartitionGpuMatchesOriginalGolden(string signal)
        {
            Assert.That(SystemInfo.supportsComputeShaders,Is.True,"Actual GPU required; never skip.");
            var source=Environment.GetEnvironmentVariable("EG_ACCEPTANCE_SOURCE");
            var root=Path.Combine(Application.dataPath,"GpuAudioPartitionedFirstFf");
            var originalRoot=Path.GetFullPath(Path.Combine(Application.dataPath,"../../k/artifacts/audio-real-chain/audio-runtime-reference"));
            var output=Path.GetFullPath(Path.Combine(Application.dataPath,"../artifacts/partitioned-first-ff-gpu-"+source+"-"+signal+".json"));
            Assert.That(File.Exists(output),Is.False);
            var report=new JObject { ["success"]=false,["scope"]="one_partitioned_original_audio_encoder_gpu_numeric_gate",
                ["runtime_source"]=source,["signal"]=signal,["requested_backend"]="GPUCompute",
                ["gpu"]=SystemInfo.graphicsDeviceName,["phase"]="audit",["end_to_end_validated"]=false,["all_original_signals_validated"]=false };
            void Save(){Directory.CreateDirectory(Path.GetDirectoryName(output));File.WriteAllText(output,report.ToString());}
            void Check(string path,string sha){using var stream=File.OpenRead(path);Assert.That(PlayerFileHash.Compute(stream,out _),Is.EqualTo(sha));}
            Save();
            var kernels=new List<GpuBlockedMatmul>();
            try
            {
                var locked=JObject.Parse(File.ReadAllText(Path.Combine(Application.dataPath,"../Packages/packages-lock.json")));
                foreach(var package in new[]{"com.ayutaz.embeddinggemma","com.ayutaz.embeddinggemma.validation"})
                    Assert.That(locked["dependencies"][package].Value<string>("hash"),Is.EqualTo(source));
                var plan=JObject.Parse(File.ReadAllText(Path.Combine(root,"partition-plan.json")));
                Assert.That(plan.Value<string>("source_graph_sha256"),Is.EqualTo("63f24d4883047ebe1f30572d7d45ad5bead33891bbe79a256f9ee10867a564b2"));
                Assert.That(plan.Value<string>("reference_source"),Is.EqualTo("4edbe17a8105233c5aaa1514faa2862b08a4f7e0"));
                Assert.That(plan.Value<bool>("source_initializer_bytes_preserved"),Is.True);
                foreach(var file in (JObject)plan["files"])
                {
                    var path=Path.Combine(root,file.Key);Assert.That(new FileInfo(path).Length,Is.EqualTo(file.Value.Value<long>("bytes")));
                    Check(path,file.Value.Value<string>("sha256"));
                }
                var referencePath=Path.Combine(originalRoot,"audio-encoder-reference.json");
                Check(referencePath,"ac23f39904b8149dd5ccb6485afcc93d96047f0977b5faf1d640c8bd018c13c9");
                var reference=JObject.Parse(File.ReadAllText(referencePath));
                var audit=JObject.Parse(File.ReadAllText(Path.Combine(originalRoot,"audio-encoder-validation.json")));
                var row=((JArray)reference["cases"]).Single(c=>c.Value<string>("id")==signal);
                Assert.That(row.Value<string>("profile"),Is.EqualTo(plan.Value<string>("profile")),"The exact padded profile is required; never substitute full profile.");
                var inputName=row.Value<string>("input_features");Check(Path.Combine(originalRoot,inputName),audit["files"][inputName].Value<string>("sha256"));
                report["phase"]="prepare";report["partition_producer"]=plan["producer"];Save();
                var models=new List<Model>();var layers=new JArray();var timer=System.Diagnostics.Stopwatch.StartNew();
                foreach(var segment in (JArray)plan["segments"])
                {
                    var asset=AssetDatabase.LoadAssetAtPath<ModelAsset>("Assets/GpuAudioPartitionedFirstFf/"+segment.Value<string>("file"));
                    Assert.That(asset,Is.Not.Null);var model=ModelLoader.Load(asset);
                    model=CompensatedModel.CreateWithDynamicMatMuls(model,16);model=CompensatedModel.Create(model,64);
                    models.Add(model);layers.Add(model.layers.Count);
                }
                var inputs=new List<string>();var outputs=new List<string>();
                foreach(var projection in (JArray)plan["projections"])
                {
                    var shape=projection["weight_shape"].ToObject<int[]>();var weightPath=Path.Combine(root,projection.Value<string>("file"));
                    Check(weightPath,projection.Value<string>("weight_sha256"));
                    kernels.Add(new GpuBlockedMatmul(shape[0],shape[1],Floats(weightPath)));
                    inputs.Add(projection.Value<string>("input"));outputs.Add(projection.Value<string>("output"));
                }
                report["segment_layer_counts"]=layers;report["prepare_ms"]=timer.Elapsed.TotalMilliseconds;
                using var pipeline=new GpuProjectionPipeline(models.ToArray(),inputs.ToArray(),outputs.ToArray(),kernels.ToArray());
                report["cpu_computation_or_input_count"]=0;report["readback_flush_count"]=0;
                using var input=new Tensor<float>(new TensorShape(1,1119,128),Floats(Path.Combine(originalRoot,inputName)));ComputeTensorData.Pin(input);
                report["phase"]="execute";Save();timer.Restart();
                using var encoded=pipeline.Execute(input);Assert.That(encoded.backendType,Is.EqualTo(BackendType.GPUCompute));
                var actual=encoded.DownloadToArray();report["execute_and_final_download_ms"]=timer.Elapsed.TotalMilliseconds;
                var expected=row["features"].ToObject<float[][]>().SelectMany(v=>v).ToArray();Assert.That(actual.Length,Is.EqualTo(expected.Length));
                int mismatch=0;double maximum=0,ratio=0;
                for(var i=0;i<actual.Length;i++)
                {
                    Assert.That(float.IsNaN(actual[i])||float.IsInfinity(actual[i]),Is.False);
                    var difference=Math.Abs((double)actual[i]-expected[i]);var allowed=1e-5+1e-4*Math.Abs((double)expected[i]);
                    maximum=Math.Max(maximum,difference);ratio=Math.Max(ratio,difference/allowed);if(difference>allowed)mismatch++;
                }
                report["actual_backend"]=encoded.backendType.ToString();report["mismatched_elements"]=mismatch;
                report["maximum_absolute_error"]=maximum;report["maximum_tolerance_ratio"]=ratio;report["phase"]="completed";Save();
                Assert.That(mismatch,Is.Zero,"Keep the original full FP32 audio encoder gate.");report["success"]=true;
            }
            catch(Exception error){report["error"]=error.ToString();throw;}
            finally { foreach(var kernel in kernels)kernel.Dispose();Save(); }
        }
        static float[] Floats(string path){var bytes=File.ReadAllBytes(path);var values=new float[bytes.Length/4];Buffer.BlockCopy(bytes,0,values,0,bytes.Length);return values;}
    }
}
