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
    public sealed class SavedOnnxAudioGpuLayerObservationTests
    {
        [Test]
        public void ObserveFirstFeedForwardOnGpuWithoutChangingNormalFinalBits()
        {
            Assert.That(SystemInfo.supportsComputeShaders, Is.True, "GPU required; no skip.");
            var source=Environment.GetEnvironmentVariable("EG_ACCEPTANCE_SOURCE");
            var nativeRoot=Path.GetFullPath(Path.Combine(Application.dataPath,"../../k/artifacts/audio-native-feed-forward-replay"));
            var root=Path.GetFullPath(Path.Combine(Application.dataPath,"../../k/artifacts/audio-real-chain/audio-runtime-reference"));
            var output=Path.GetFullPath(Path.Combine(Application.dataPath,"../artifacts/saved-onnx-audio-layer-gpu-"+source+".json"));
            Assert.That(File.Exists(output),Is.False);
            var rows=new JArray();
            var report=new JObject { ["success"]=false,["scope"]="one_gpu_first_feed_forward_observation_only",
                ["runtime_source"]=source,["requested_backend"]="GPUCompute",["stages"]=rows,
                ["full_encoder_validated"]=false,["end_to_end_validated"]=false,["phase"]="audit" };
            void Save() { Directory.CreateDirectory(Path.GetDirectoryName(output)); File.WriteAllText(output,report.ToString()); }
            void Hash(string path,string expected) { using var stream=File.OpenRead(path); Assert.That(PlayerFileHash.Compute(stream,out _),Is.EqualTo(expected)); }
            Save();
            try
            {
                var locked=JObject.Parse(File.ReadAllText(Path.Combine(Application.dataPath,"../Packages/packages-lock.json")));
                foreach(var package in new[]{"com.ayutaz.embeddinggemma","com.ayutaz.embeddinggemma.validation"})
                    Assert.That(locked["dependencies"][package].Value<string>("hash"),Is.EqualTo(source));
                var replayPath=Path.Combine(nativeRoot,"audio-layer-replay-report.json");
                Hash(replayPath,"99529dc0fa6c74825b23fca653d82091797ff1e5267d26c5cdae723fcf86f501");
                var replay=JObject.Parse(File.ReadAllText(replayPath));
                Assert.That(replay.Value<bool>("success") && replay.Value<bool>("native_final_exact_match"),Is.True);
                var anchorPath=Path.Combine(nativeRoot,"feed-forward-norm-anchors.json");
                Hash(anchorPath,"95f3e32536dce9b3b3820050dac4bc837c0f460c75e4e56735381bf924007c33");
                var anchors=JObject.Parse(File.ReadAllText(anchorPath));
                var modelPath=Path.Combine(Application.dataPath,"GpuAudioCandidate/frames-1119-valid-1119.onnx");
                Hash(modelPath,"f43f3275c1ba48837b626fc09c738a8204dabd25e2ff3c74d7e75abcb3591899");
                var asset=AssetDatabase.LoadAssetAtPath<ModelAsset>("Assets/GpuAudioCandidate/frames-1119-valid-1119.onnx");
                Assert.That(asset,Is.Not.Null); var model=ModelLoader.Load(asset);
                report["phase"]="resolve_anchors"; Save();
                var map=AudioCoefficientLayerMap.ResolveFeedForward(model,(JArray)anchors["anchors"]);
                var binaryRoot=Path.ChangeExtension(output,null)+"-captures";
                Directory.CreateDirectory(binaryRoot);
                var scaleLayer=model.layers.Single(l=>l.outputs.Contains(map["post_layer_norm"]));
                var scaleId=scaleLayer.inputs.Single(i=>model.constants.Any(c=>c.index==i));
                var scale=model.constants.Single(c=>c.index==scaleId);
                Assert.That(scale.dataType,Is.EqualTo(DataType.Float));
                var scaleBytes=(ArraySegment<byte>)typeof(Constant).GetField("array",BindingFlags.Instance|BindingFlags.NonPublic).GetValue(scale);
                using(var stream=File.Create(Path.Combine(binaryRoot,"post-norm-folded-weight.bin"))) stream.Write(scaleBytes.Array,scaleBytes.Offset,scaleBytes.Count);
                report["capture_directory"]=binaryRoot;
                var inputPath=Path.Combine(root,"frames-1119-valid-1119/silence-features-fp32.bin");
                Hash(inputPath,"366de71cb4e45e4feda58a605babe4ea9d8693a7e37e84eac0735a7c42d0ad24");
                using var input=new Tensor<float>(new TensorShape(1,1119,128),Floats(inputPath)); ComputeTensorData.Pin(input);
                report["bound_outputs"]=JObject.FromObject(map); report["phase"]="observe"; Save();
                var observed=WorkerObservation.ObserveIndexes(model,BackendType.GPUCompute,input,new HashSet<int>(map.Values));
                Assert.That(observed.FinalMatchesBaselineBits,Is.True,"Diagnostic scheduling changed original final bits.");
                report["normal_final_matches_observed_bits"]=observed.FinalMatchesBaselineBits;
                Assert.That(observed.Snapshots.Count,Is.EqualTo(5));
                foreach(var name in new[]{"pre_layer_norm","ffw_layer_1","act_fn","ffw_layer_2","post_layer_norm"})
                {
                    var snapshot=observed.Snapshots.Single(s=>s.TensorIndex==map[name]);
                    Assert.That(snapshot.Backend,Is.EqualTo(BackendType.GPUCompute));
                    var part=replay["parts"]["feed_forward1."+name];
                    var path=Path.Combine(nativeRoot,"audio-layer-replay",part.Value<string>("file"));
                    Hash(path,part.Value<string>("sha256")); var expected=Floats(path);
                    var actual=name=="post_layer_norm"?AudioCoefficientLayerMap.RestoreHalfScale(snapshot.Values):snapshot.Values;
                    var actualBytes=new byte[actual.Length*4];Buffer.BlockCopy(actual,0,actualBytes,0,actualBytes.Length);
                    File.WriteAllBytes(Path.Combine(binaryRoot,name+"-actual.bin"),actualBytes);
                    Assert.That(actual.Length,Is.EqualTo(expected.Length));
                    int mismatches=0; double max=0,ratio=0;
                    for(var i=0;i<actual.Length;i++) { var d=Math.Abs((double)actual[i]-expected[i]);var a=1e-5+1e-4*Math.Abs((double)expected[i]);if(d>a)mismatches++;max=Math.Max(max,d);ratio=Math.Max(ratio,d/a); }
                    rows.Add(new JObject { ["stage"]=name,["numeric_passed"]=mismatches==0,["mismatches"]=mismatches,
                        ["maximum_absolute_error"]=max,["maximum_tolerance_ratio"]=ratio,["tensor_index"]=snapshot.TensorIndex,
                        ["actual_backend"]=snapshot.Backend.ToString(),["restore_multiplier"]=name=="post_layer_norm"?2:1 }); Save();
                }
                report["phase"]="completed";report["observation_completed"]=true;report["success"]=true;
            }
            catch(Exception error) { report["error"]=error.ToString();throw; }
            finally { Save(); }
        }
        static float[] Floats(string path) { var bytes=File.ReadAllBytes(path);var v=new float[bytes.Length/4];Buffer.BlockCopy(bytes,0,v,0,bytes.Length);return v; }
    }
}
