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
    public sealed class SavedOnnxAudioSubsampleGpuObservationTests
    {
        [Test]
        public void ObservePinnedSubsampleOnGpuWithoutChangingNormalFinalBits()
            => Observe(false);

        [Test]
        public void ObserveCompactSubsampleOnGpuWithoutChangingNormalFinalBits()
            => Observe(true);

        void Observe(bool compact)
        {
            Assert.That(SystemInfo.supportsComputeShaders,Is.True,"Actual GPU required; no skip.");
            var source=Environment.GetEnvironmentVariable("EG_ACCEPTANCE_SOURCE");
            var root=Path.GetFullPath(Path.Combine(Application.dataPath,"../../numerical-acceptance-state/subsample-native-replay"));
            var folder=compact?"GpuAudioCompactOuterCandidate":"GpuAudioCandidate";
            var output=Path.GetFullPath(Path.Combine(Application.dataPath,"../artifacts/"+(compact?"compact-subsample-gpu-":"saved-onnx-subsample-gpu-")+source+".json"));
            Assert.That(File.Exists(output),Is.False);
            var rows=new JArray();
            var report=new JObject { ["success"]=false,["scope"]="one_saved_gpu_subsample_observation_only",
                ["runtime_source"]=source,["candidate_source"]="6bceed7f63f74b8df012f41b853eddaca0e10702",
                ["requested_backend"]="GPUCompute",["stages"]=rows,["phase"]="audit",
                ["full_encoder_validated"]=false,["end_to_end_validated"]=false };
            void Save(){Directory.CreateDirectory(Path.GetDirectoryName(output));File.WriteAllText(output,report.ToString());}
            void Check(string path,string sha){using var stream=File.OpenRead(path);Assert.That(PlayerFileHash.Compute(stream,out _),Is.EqualTo(sha));}
            string ConstantHash(Constant value)
            {
                var bytes=(ArraySegment<byte>)typeof(Constant).GetField("array",BindingFlags.NonPublic|BindingFlags.Instance).GetValue(value);
                using var stream=new MemoryStream(bytes.Array,bytes.Offset,bytes.Count,false);return PlayerFileHash.Compute(stream,out _);
            }
            Save();
            try
            {
                var locked=JObject.Parse(File.ReadAllText(Path.Combine(Application.dataPath,"../Packages/packages-lock.json")));
                foreach(var package in new[]{"com.ayutaz.embeddinggemma","com.ayutaz.embeddinggemma.validation"})
                    Assert.That(locked["dependencies"][package].Value<string>("hash"),Is.EqualTo(source));
                var replayPath=Path.Combine(root,"audio-layer-replay-report.json");
                Check(replayPath,"664a8242d079f53bf98d2032f9398de6d2c8d6bcd2312d1baaa35bb498ce5fb8");
                var replay=JObject.Parse(File.ReadAllText(replayPath));
                Assert.That(replay.Value<bool>("success") && replay.Value<bool>("subsample_native_final_exact_match"),Is.True);
                Assert.That(replay.Value<string>("reference_source"),Is.EqualTo("4edbe17a8105233c5aaa1514faa2862b08a4f7e0"));
                var modelPath=Path.Combine(Application.dataPath,folder+"/frames-1119-valid-1119.onnx");
                var graphSha=compact?"11607255142478773e621f5e9feebf2d3269457feb67d63215cecbc55fc61b67":"f43f3275c1ba48837b626fc09c738a8204dabd25e2ff3c74d7e75abcb3591899";
                Check(modelPath,graphSha);report["graph_sha256"]=graphSha;
                var model=ModelLoader.Load(AssetDatabase.LoadAssetAtPath<ModelAsset>("Assets/"+folder+"/frames-1119-valid-1119.onnx"));
                report["phase"]="bind";Save();
                var constants=model.constants.ToDictionary(c=>c.index);
                int Weighted(string kind,string sha)
                {
                    var matches=model.layers.Where(l=>l.GetType().Name==kind && l.inputs.Length>=2 &&
                        constants.TryGetValue(l.inputs[1],out var weight) && ConstantHash(weight)==sha).ToArray();
                    Assert.That(matches.Length,Is.EqualTo(1),"Pinned subsample weight path is missing or ambiguous.");
                    return matches[0].outputs.Single();
                }
                var normAnchors=new JArray(
                    new JObject { ["stage"]="layer0.norm",["coefficient_bytes"]=512,["coefficient_sha256"]="dbb0fc3431144ac4a6265e965a3bac5bb1c0ee9e87b266bc4cb13d2b7d4d8c24" },
                    new JObject { ["stage"]="layer1.norm",["coefficient_bytes"]=128,["coefficient_sha256"]="7b744a9e296e2331f5553570fdb993412b8f760f1a000179f366554cc84b1434" });
                var map=AudioCoefficientLayerMap.Resolve(model,normAnchors);
                map["layer0.conv"]=Weighted("Conv","e4d4ee18dc86a9945d38c159a861784a2c3d9467d6e0be9c48a414e60e2b92fd");
                map["layer1.conv"]=Weighted("Conv","68309f01ea3e738fccb24a0db5db608841a3a8e59b0a4fa5d0fb521ed60bf43c");
                map["layer0.act"]=AudioCoefficientLayerMap.ResolveReluOutput(model,map["layer0.norm"]);
                map["layer1.act"]=AudioCoefficientLayerMap.ResolveReluOutput(model,map["layer1.norm"]);
                var ffRoot=Path.GetFullPath(Path.Combine(Application.dataPath,"../../k/artifacts/audio-native-feed-forward-replay"));
                var anchorPath=Path.Combine(ffRoot,"feed-forward-norm-anchors.json");
                Check(anchorPath,"95f3e32536dce9b3b3820050dac4bc837c0f460c75e4e56735381bf924007c33");
                if(compact)map["input_proj_linear"]=Weighted("MatMul","a59650e614c2746ee18bbb23c475f65677beeb9ffe49350854af2aad873052a1");
                else
                {
                    var beforeNorm=AudioCoefficientLayerMap.ResolveNormInputs(model,(JArray)JObject.Parse(File.ReadAllText(anchorPath))["anchors"])["pre_layer_norm"];
                    var producer=model.layers.Single(l=>l.outputs.Contains(beforeNorm));
                    if(producer.GetType().Name=="Clip") beforeNorm=producer.inputs[0];
                    producer=model.layers.Single(l=>l.outputs.Contains(beforeNorm));
                    Assert.That(producer.GetType().Name,Is.EqualTo("MatMul"));
                    Assert.That(producer.inputs.Count(constants.ContainsKey),Is.EqualTo(1));
                    map["input_proj_linear"]=beforeNorm;
                }
                if(compact)
                {
                    model=CompensatedModel.CreateWithDynamicMatMuls(model,16);
                    model=CompensatedModel.Create(model,64);report["layer_count"]=model.layers.Count;
                    using var worker=new Worker(model,BackendType.GPUCompute);
                    var cpu=(HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallback",BindingFlags.NonPublic|BindingFlags.Instance).GetValue(worker);
                    var flush=(HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallbackShouldFlushGPU",BindingFlags.NonPublic|BindingFlags.Instance).GetValue(worker);
                    var count=cpu.Intersect(model.layers.SelectMany(l=>l.outputs).Concat(model.inputs.Select(i=>i.index))).Count();
                    report["cpu_computation_or_input_count"]=count;report["readback_flush_count"]=flush.Count;
                    Assert.That(count,Is.Zero);Assert.That(flush.Count,Is.Zero);
                }
                report["bound_outputs"]=JObject.FromObject(map);
                var inputPath=Path.GetFullPath(Path.Combine(Application.dataPath,"../../k/artifacts/audio-real-chain/audio-runtime-reference/frames-1119-valid-1119/silence-features-fp32.bin"));
                Check(inputPath,"366de71cb4e45e4feda58a605babe4ea9d8693a7e37e84eac0735a7c42d0ad24");
                using var input=new Tensor<float>(new TensorShape(1,1119,128),Floats(inputPath));ComputeTensorData.Pin(input);
                report["phase"]="observe";Save();
                var observed=WorkerObservation.ObserveIndexes(model,BackendType.GPUCompute,input,new HashSet<int>(map.Values));
                Assert.That(observed.FinalMatchesBaselineBits,Is.True);Assert.That(observed.Snapshots.Count,Is.EqualTo(7));
                report["normal_final_matches_observed_bits"]=true;
                var captures=Path.ChangeExtension(output,null)+"-captures";Directory.CreateDirectory(captures);report["capture_directory"]=captures;
                foreach(var name in new[]{"layer0.conv","layer0.norm","layer0.act","layer1.conv","layer1.norm","layer1.act","input_proj_linear"})
                {
                    var snapshot=observed.Snapshots.Single(s=>s.TensorIndex==map[name]);Assert.That(snapshot.Backend,Is.EqualTo(BackendType.GPUCompute));
                    var part=replay["parts"]["subsample."+name];var path=Path.Combine(root,"audio-layer-replay",part.Value<string>("file"));Check(path,part.Value<string>("sha256"));
                    var expected=Floats(path);var actual=snapshot.Values;Assert.That(actual.Length,Is.EqualTo(expected.Length));
                    var bytes=new byte[actual.Length*4];Buffer.BlockCopy(actual,0,bytes,0,bytes.Length);File.WriteAllBytes(Path.Combine(captures,name+"-actual.bin"),bytes);
                    int mismatch=0,exact=0;double maximum=0,ratio=0;
                    for(var i=0;i<actual.Length;i++){Assert.That(float.IsNaN(actual[i])||float.IsInfinity(actual[i]),Is.False);var d=Math.Abs((double)actual[i]-expected[i]);var allowed=1e-5+1e-4*Math.Abs((double)expected[i]);if(d>allowed)mismatch++;if(actual[i]!=expected[i])exact++;maximum=Math.Max(maximum,d);ratio=Math.Max(ratio,d/allowed);}
                    rows.Add(new JObject { ["stage"]=name,["mismatches"]=mismatch,["exact_difference_count"]=exact,["numeric_passed"]=mismatch==0,["maximum_absolute_error"]=maximum,["maximum_tolerance_ratio"]=ratio,["actual_backend"]=snapshot.Backend.ToString() });Save();
                }
                report["first_nonzero_stage"]=rows.FirstOrDefault(r=>r.Value<int>("exact_difference_count")>0)?["stage"];
                report["first_tolerance_exceeding_stage"]=rows.FirstOrDefault(r=>r.Value<int>("mismatches")>0)?["stage"];
                report["observation_completed"]=true;report["phase"]="completed";report["success"]=true;
            }
            catch(Exception error){report["error"]=error.ToString();throw;}
            finally{Save();}
        }
        static float[] Floats(string path){var bytes=File.ReadAllBytes(path);var values=new float[bytes.Length/4];Buffer.BlockCopy(bytes,0,values,0,bytes.Length);return values;}
    }
}
