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
    public sealed class RefinedAudioFullLayerObservationTests
    {
        [Test]
        public void ObserveStatic64LayersAgainstOriginalNativeWithoutChangingFinalBits()
            => Observe(false);

        [Test]
        public void ObserveCompactOuterLayersAgainstOriginalNativeWithoutChangingFinalBits()
            => Observe(true);

        void Observe(bool compact)
        {
            Assert.That(SystemInfo.supportsComputeShaders,Is.True,"Actual GPU required; no skip.");
            var source=Environment.GetEnvironmentVariable("EG_ACCEPTANCE_SOURCE");
            var root=Path.GetFullPath(Path.Combine(Application.dataPath,"../../k/artifacts/audio-current-stage-context"));
            var folder=compact?"GpuAudioCompactOuterCandidate":"GpuAudioRefinedCandidate";
            var stem=compact?"compact-outer-full-layers-":"refined-static64-full-layers-";
            var output=Path.GetFullPath(Path.Combine(Application.dataPath,"../artifacts/"+stem+source+".json"));
            Assert.That(File.Exists(output),Is.False);
            var rows=new JArray();var report=new JObject { ["success"]=false,["scope"]=compact?"silence_compact_outer_original_native_full_layer_observation_only":"silence_refined_static64_original_native_full_layer_observation_only",
                ["source"]=source,["requested_backend"]="GPUCompute",["stages"]=rows,["phase"]="audit",
                ["full_encoder_validated"]=false,["end_to_end_validated"]=false };
            void Save(){Directory.CreateDirectory(Path.GetDirectoryName(output));File.WriteAllText(output,report.ToString());}
            void Check(string path,string sha){using var stream=File.OpenRead(path);Assert.That(PlayerFileHash.Compute(stream,out _),Is.EqualTo(sha));}
            Save();
            try
            {
                var locked=JObject.Parse(File.ReadAllText(Path.Combine(Application.dataPath,"../Packages/packages-lock.json")));
                foreach(var p in new[]{"com.ayutaz.embeddinggemma","com.ayutaz.embeddinggemma.validation"})Assert.That(locked["dependencies"][p].Value<string>("hash"),Is.EqualTo(source));
                var referencePath=Path.Combine(root,"audio-stage-reference.json");var auditPath=Path.Combine(root,"audio-stage-validation.json");
                Check(referencePath,"6620f335563a91ccc484df7d9b89d4d889e58dab4ec3d7d0389b36abe36b454a");
                Check(auditPath,"3edee8acdd9ebebb819ae488b753a3a94949b953aa0906995e516764932b0f9b");
                var reference=JObject.Parse(File.ReadAllText(referencePath));var audit=JObject.Parse(File.ReadAllText(auditPath));
                Assert.That(audit.Value<bool>("success")&&audit.Value<bool>("context_validated"),Is.True);
                Assert.That(JToken.DeepEquals(reference["metadata"],audit["metadata"]),Is.True);
                Assert.That(reference["metadata"].Value<string>("source_commit"),Is.EqualTo("4edbe17a8105233c5aaa1514faa2862b08a4f7e0"));
                var originalCase=((JArray)reference["cases"]).Single(c=>c.Value<string>("id")=="silence");
                string ResolveExpected(string file)
                {
                    Assert.That(file.StartsWith("frames-1119-valid-1119/",StringComparison.Ordinal),Is.True);
                    var nested=Path.Combine(root,file);
                    // This immutable local cache retained the native binaries flat.
                    return File.Exists(nested)?nested:Path.Combine(root,Path.GetFileName(file));
                }
                // Refuse missing/corrupt references before model load or GPU work.
                foreach(var part in ((JArray)originalCase["stages"]).Where(s=>s.Value<string>("name").StartsWith("layer_",StringComparison.Ordinal)))
                {
                    var file=part.Value<string>("expected");Check(ResolveExpected(file),audit["files"][file].Value<string>("sha256"));
                }
                var anchors=JObject.Parse(File.ReadAllText(Path.Combine(Application.dataPath,"GpuAudioRefinedCandidate/full-layer-anchors.json")));
                Assert.That(anchors.Value<string>("native_reference_sha256"),Is.EqualTo("6620f335563a91ccc484df7d9b89d4d889e58dab4ec3d7d0389b36abe36b454a"));
                var modelPath=Path.Combine(Application.dataPath,folder+"/frames-1119-valid-1119.onnx");
                var graphSha=compact?"11607255142478773e621f5e9feebf2d3269457feb67d63215cecbc55fc61b67":"8aefc4b83fd0cc709544a0d6337b8c1837d2c864ad1d3c8cbd39fcd568abc222";
                Check(modelPath,graphSha);report["graph_sha256"]=graphSha;
                if(compact)
                {
                    var candidateAudit=JObject.Parse(File.ReadAllText(Path.Combine(Application.dataPath,folder+"/audio-encoder-validation.json")));
                    Assert.That(candidateAudit["rms_graph_transform"].Value<bool>("existing_initializer_bytes_preserved"),Is.True);
                    Assert.That(candidateAudit["rms_graph_transform"].Value<string>("source_graph_sha256"),Is.EqualTo(anchors.Value<string>("saved_candidate_graph_sha256")));
                }
                var model=ModelLoader.Load(AssetDatabase.LoadAssetAtPath<ModelAsset>("Assets/"+folder+"/frames-1119-valid-1119.onnx"));
                var map=AudioCoefficientLayerMap.Resolve(model,(JArray)anchors["anchors"]);Assert.That(map.Count,Is.EqualTo(12));
                report["bound_outputs"]=JObject.FromObject(map);report["phase"]="transform";Save();
                if(compact)model=CompensatedModel.CreateWithDynamicMatMuls(model,16);
                model=CompensatedModel.Create(model,64);
                report["layer_count"]=model.layers.Count;
                using(var worker=new Worker(model,BackendType.GPUCompute))
                {
                    var cpu=(HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallback",BindingFlags.NonPublic|BindingFlags.Instance).GetValue(worker);
                    var flush=(HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallbackShouldFlushGPU",BindingFlags.NonPublic|BindingFlags.Instance).GetValue(worker);
                    var count=cpu.Intersect(model.layers.SelectMany(l=>l.outputs).Concat(model.inputs.Select(i=>i.index))).Count();
                    report["cpu_computation_or_input_count"]=count;report["readback_flush_count"]=flush.Count;Assert.That(count,Is.Zero);Assert.That(flush.Count,Is.Zero);
                }
                var inputPath=Path.GetFullPath(Path.Combine(Application.dataPath,"../../k/artifacts/audio-real-chain/audio-runtime-reference/frames-1119-valid-1119/silence-features-fp32.bin"));
                Check(inputPath,"366de71cb4e45e4feda58a605babe4ea9d8693a7e37e84eac0735a7c42d0ad24");
                using var input=new Tensor<float>(new TensorShape(1,1119,128),Floats(inputPath));ComputeTensorData.Pin(input);
                report["phase"]="observe";Save();var observed=WorkerObservation.ObserveIndexes(model,BackendType.GPUCompute,input,new HashSet<int>(map.Values));
                Assert.That(observed.FinalMatchesBaselineBits,Is.True);Assert.That(observed.Snapshots.Count,Is.EqualTo(12));report["normal_final_matches_observed_bits"]=true;
                if(compact)
                {
                    var encoderPath=Path.GetFullPath(Path.Combine(Application.dataPath,"../../k/artifacts/audio-real-chain/audio-runtime-reference/audio-encoder-reference.json"));
                    Check(encoderPath,"ac23f39904b8149dd5ccb6485afcc93d96047f0977b5faf1d640c8bd018c13c9");
                    var encoderReference=JObject.Parse(File.ReadAllText(encoderPath));
                    var finalExpected=((JArray)encoderReference["cases"]).Single(c=>c.Value<string>("id")=="silence")["features"].ToObject<float[][]>().SelectMany(v=>v).ToArray();
                    Assert.That(observed.Final.Length,Is.EqualTo(finalExpected.Length));
                    int mismatch=0;double ratio=0;
                    for(var i=0;i<finalExpected.Length;i++)
                    {
                        Assert.That(float.IsNaN(observed.Final[i])||float.IsInfinity(observed.Final[i]),Is.False);
                        var d=Math.Abs((double)observed.Final[i]-finalExpected[i]);var allowed=1e-5+1e-4*Math.Abs((double)finalExpected[i]);
                        if(d>allowed)mismatch++;ratio=Math.Max(ratio,d/allowed);
                    }
                    report["final_original_mismatches"]=mismatch;report["final_original_maximum_tolerance_ratio"]=ratio;
                    // Preserve the known failed gate; success below means observation only.
                    Assert.That(mismatch,Is.EqualTo(201),"Observation must faithfully replay the latest failed compact candidate.");
                }
                var captureRoot=Path.ChangeExtension(output,null)+"-captures";Directory.CreateDirectory(captureRoot);report["capture_directory"]=captureRoot;
                for(var index=0;index<12;index++)
                {
                    var name="layer_"+index;var expectedPart=((JArray)originalCase["stages"]).Single(s=>s.Value<string>("name")==name);
                    var file=expectedPart.Value<string>("expected");var expectedPath=ResolveExpected(file);Check(expectedPath,audit["files"][file].Value<string>("sha256"));
                    var expected=Floats(expectedPath);var snapshot=observed.Snapshots.Single(s=>s.TensorIndex==map[name]);Assert.That(snapshot.Backend,Is.EqualTo(BackendType.GPUCompute));
                    var actual=snapshot.Values;Assert.That(actual.Length,Is.EqualTo(expected.Length));Assert.That(actual.All(v=>!float.IsNaN(v)&&!float.IsInfinity(v)),Is.True);
                    var bytes=new byte[actual.Length*4];Buffer.BlockCopy(actual,0,bytes,0,bytes.Length);File.WriteAllBytes(Path.Combine(captureRoot,name+"-actual.bin"),bytes);
                    int mismatch=0;double maximum=0,ratio=0;
                    for(var i=0;i<actual.Length;i++){var d=Math.Abs((double)actual[i]-expected[i]);var allowed=1e-5+1e-4*Math.Abs((double)expected[i]);if(d>allowed)mismatch++;maximum=Math.Max(maximum,d);ratio=Math.Max(ratio,d/allowed);}
                    rows.Add(new JObject { ["stage"]=name,["mismatches"]=mismatch,["numeric_passed"]=mismatch==0,["maximum_absolute_error"]=maximum,["maximum_tolerance_ratio"]=ratio,["actual_backend"]=snapshot.Backend.ToString() });Save();
                }
                report["first_tolerance_exceeding_layer"]=rows.FirstOrDefault(r=>r.Value<int>("mismatches")>0)?["stage"];
                report["phase"]="completed";report["observation_completed"]=true;report["success"]=true;
            }
            catch(Exception error){report["error"]=error.ToString();throw;}
            finally{Save();}
        }
        static float[] Floats(string path){var bytes=File.ReadAllBytes(path);var result=new float[bytes.Length/4];Buffer.BlockCopy(bytes,0,result,0,bytes.Length);return result;}
    }
}
