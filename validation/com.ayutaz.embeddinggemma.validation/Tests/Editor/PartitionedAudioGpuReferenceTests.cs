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
    public sealed class PartitionedAudioGpuReferenceTests
    {
        [Test]
        public void SmallBoundaryObservationPreservesFinalBitsAndBorrowedGpuResources()
        {
            Assert.That(SystemInfo.supportsComputeShaders,Is.True,"Actual GPU required; no skip.");
            var method=typeof(PartitionedAudioGpuReferenceTests).GetMethod("ObserveBoundaries",BindingFlags.Static|BindingFlags.NonPublic);
            Assert.That(method,Is.Not.Null,"Partition boundary observation is missing.");
            var a=new FunctionalGraph();var x=a.AddInput<float>(new TensorShape(1,2),"input");var prefix=a.Compile(x);
            var named=prefix.outputs[0];named.name="input";prefix.outputs[0]=named;
            var b=new FunctionalGraph();var residual=b.AddInput<float>(new TensorShape(1,2),"input");var projected=b.AddInput<float>(new TensorShape(1,2),"projected");
            var tail=b.Compile(residual+projected);named=tail.outputs[0];named.name="final";tail.outputs[0]=named;
            using var kernel=new GpuBlockedMatmul(2,2,new[]{2f,0f,0f,3f});
            using var input=new Tensor<float>(new TensorShape(1,2),new[]{-3f,4f});ComputeTensorData.Pin(input);
            var snapshots=(Dictionary<string,float[]>)method.Invoke(null,new object[]{new[]{prefix,tail},new[]{"input"},new[]{"projected"},new[]{kernel},input,new HashSet<string>{"input"}});
            Assert.That(snapshots["input"],Is.EqualTo(new[]{-3f,4f}));
            Assert.That(snapshots["__final"],Is.EqualTo(new[]{-9f,16f}));
            using var reused=kernel.Execute(input);Assert.That(reused.backendType,Is.EqualTo(BackendType.GPUCompute));
        }

        [TestCase("silence")]
        [TestCase("impulse")]
        [TestCase("sine")]
        [TestCase("nyquist")]
        [TestCase("noise")]
        [TestCase("padded_noise")]
        public void FirstFeedForwardPartitionGpuMatchesOriginalGolden(string signal)
            =>Execute(signal,false);

        [TestCase("silence")]
        [TestCase("impulse")]
        [TestCase("sine")]
        [TestCase("nyquist")]
        [TestCase("noise")]
        [TestCase("padded_noise")]
        public void AllStaticPartitionsGpuMatchesOriginalGolden(string signal)
            =>Execute(signal,true);

        [Test]
        public void ObserveAllStaticLayersAgainstOriginalNativeWithoutChangingFinalBits()
            =>Execute("silence",true,true);

        void Execute(string signal,bool allStatic,bool observe=false)
        {
            Assert.That(SystemInfo.supportsComputeShaders,Is.True,"Actual GPU required; never skip.");
            var source=Environment.GetEnvironmentVariable("EG_ACCEPTANCE_SOURCE");
            var folder=allStatic?"GpuAudioPartitionedAllStatic":"GpuAudioPartitionedFirstFf";
            var root=Path.Combine(Application.dataPath,folder);
            var originalRoot=Path.GetFullPath(Path.Combine(Application.dataPath,"../../k/artifacts/audio-real-chain/audio-runtime-reference"));
            var output=Path.GetFullPath(Path.Combine(Application.dataPath,"../artifacts/"+(observe?"partitioned-all-static-layer-observation-":allStatic?"partitioned-all-static-gpu-":"partitioned-first-ff-gpu-")+source+"-"+signal+".json"));
            Assert.That(File.Exists(output),Is.False);
            var report=new JObject { ["success"]=false,["scope"]=observe?"original_audio_layer_observation_only":"one_partitioned_original_audio_encoder_gpu_numeric_gate",
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
                Assert.That(plan.Value<string>("source_graph_sha256"),Is.EqualTo(allStatic?"c050c2efa87d83aea45587b25f7b8d1d24dc471f33cc732dcf17d89588c70c13":"63f24d4883047ebe1f30572d7d45ad5bead33891bbe79a256f9ee10867a564b2"));
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
                    var asset=AssetDatabase.LoadAssetAtPath<ModelAsset>("Assets/"+folder+"/"+segment.Value<string>("file"));
                    Assert.That(asset,Is.Not.Null);var model=ModelLoader.Load(asset);
                    var aliases=(JObject)segment["output_aliases"];
                    for(var index=0;index<model.outputs.Count;index++)
                    {
                        var boundary=model.outputs[index];
                        if(aliases?.Property(boundary.name)!=null){boundary.name=aliases.Value<string>(boundary.name);model.outputs[index]=boundary;}
                    }
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
                report["projection_count"]=kernels.Count;
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
                if(observe)
                {
                    var contextRoot=Path.GetFullPath(Path.Combine(Application.dataPath,"../../k/artifacts/audio-current-stage-context"));
                    var layerReferencePath=Path.Combine(contextRoot,"audio-stage-reference.json");var layerAuditPath=Path.Combine(contextRoot,"audio-stage-validation.json");
                    Check(layerReferencePath,"6620f335563a91ccc484df7d9b89d4d889e58dab4ec3d7d0389b36abe36b454a");
                    Check(layerAuditPath,"3edee8acdd9ebebb819ae488b753a3a94949b953aa0906995e516764932b0f9b");
                    var layerReference=JObject.Parse(File.ReadAllText(layerReferencePath));var layerAudit=JObject.Parse(File.ReadAllText(layerAuditPath));
                    Assert.That(layerAudit.Value<bool>("success")&&layerAudit.Value<bool>("context_validated"),Is.True);
                    Assert.That(layerReference["metadata"].Value<string>("source_commit"),Is.EqualTo("4edbe17a8105233c5aaa1514faa2862b08a4f7e0"));
                    var layerCase=((JArray)layerReference["cases"]).Single(c=>c.Value<string>("id")==signal);
                    var anchorPath=Path.Combine(Application.dataPath,"GpuAudioRefinedCandidate/full-layer-anchors.json");
                    Check(anchorPath,"34e8d2c011b0fb8f083f1a8c850dfc25193fd526696e186a833740d7980c4092");
                    var anchors=(JArray)JObject.Parse(File.ReadAllText(anchorPath))["anchors"];
                    var wanted=anchors.Select(a=>a.Value<string>("onnx_node")+"_output_0").ToHashSet();
                    report["phase"]="observe";Save();
                    var snapshots=ObserveBoundaries(models.ToArray(),inputs.ToArray(),outputs.ToArray(),kernels.ToArray(),input,wanted);
                    Assert.That(snapshots["__final"].Select(BitConverter.SingleToInt32Bits),Is.EqualTo(actual.Select(BitConverter.SingleToInt32Bits)));
                    report["normal_final_matches_observed_bits"]=true;var stages=new JArray();report["stages"]=stages;
                    foreach(var anchor in anchors)
                    {
                        var name=anchor.Value<string>("stage");var boundary=anchor.Value<string>("onnx_node")+"_output_0";
                        var part=((JArray)layerCase["stages"]).Single(p=>p.Value<string>("name")==name);
                        var file=part.Value<string>("expected");var path=Path.Combine(contextRoot,file);
                        if(!File.Exists(path))path=Path.Combine(contextRoot,Path.GetFileName(file));
                        Check(path,layerAudit["files"][file].Value<string>("sha256"));
                        var values=snapshots[boundary];var native=Floats(path);Assert.That(values.Length,Is.EqualTo(native.Length));
                        int bits=0,bad=0;double maxRatio=0;
                        for(var i=0;i<values.Length;i++)
                        {
                            Assert.That(float.IsNaN(values[i])||float.IsInfinity(values[i]),Is.False);
                            if(BitConverter.SingleToInt32Bits(values[i])!=BitConverter.SingleToInt32Bits(native[i]))bits++;
                            var difference=Math.Abs((double)values[i]-native[i]);var allowed=1e-5+1e-4*Math.Abs((double)native[i]);
                            if(difference>allowed)bad++;maxRatio=Math.Max(maxRatio,difference/allowed);
                        }
                        stages.Add(new JObject{["stage"]=name,["boundary"]=boundary,["bit_differences"]=bits,["mismatched_elements"]=bad,["maximum_tolerance_ratio"]=maxRatio});Save();
                    }
                    Assert.That(stages.Count,Is.EqualTo(12));report["observation_completed"]=true;
                    report["full_encoder_validated"]=false;report["success"]=true;report["phase"]="completed_observation";return;
                }
                Assert.That(mismatch,Is.Zero,"Keep the original full FP32 audio encoder gate.");report["success"]=true;
            }
            catch(Exception error){report["error"]=error.ToString();throw;}
            finally { foreach(var kernel in kernels)kernel.Dispose();Save(); }
        }
        // Diagnostic-only downloads; the product GPU pipeline is unchanged.
        static Dictionary<string,float[]> ObserveBoundaries(Model[] models,string[] inputs,string[] outputs,GpuBlockedMatmul[] kernels,Tensor<float> input,HashSet<string> wanted)
        {
            if(!wanted.IsSubsetOf(models.SelectMany(m=>m.outputs.Select(o=>o.name))))throw new ArgumentException("Observed boundary is missing.");
            float[] baseline;
            using(var normal=new GpuProjectionPipeline(models,inputs,outputs,kernels))
            using(var final=normal.Execute(input))baseline=final.DownloadToArray();
            var values=new Dictionary<string,Tensor<float>>{{models[0].inputs[0].name,input}};
            var owned=new HashSet<Tensor<float>>();var workers=new List<Worker>();var snapshots=new Dictionary<string,float[]>();
            try
            {
                for(var stage=0;stage<models.Length;stage++)
                {
                    var model=models[stage];var worker=new Worker(model,BackendType.GPUCompute);workers.Add(worker);
                    var cpu=(HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallback",BindingFlags.NonPublic|BindingFlags.Instance).GetValue(worker);
                    var flush=(HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallbackShouldFlushGPU",BindingFlags.NonPublic|BindingFlags.Instance).GetValue(worker);
                    Assert.That(cpu.Intersect(model.layers.SelectMany(l=>l.outputs).Concat(model.inputs.Select(i=>i.index))).Count(),Is.Zero);Assert.That(flush.Count,Is.Zero);
                    foreach(var boundary in model.inputs)worker.SetInput(boundary.name,values[boundary.name]);
                    worker.Schedule();
                    foreach(var boundary in model.outputs)
                    {
                        Tensor copy=null;worker.CopyOutput(boundary.name,ref copy);var value=copy as Tensor<float>;
                        if(value==null){copy?.Dispose();throw new InvalidOperationException("Expected a float GPU boundary.");}
                        owned.Add(value);Assert.That(value.backendType,Is.EqualTo(BackendType.GPUCompute));values[boundary.name]=value;
                        if(wanted.Contains(boundary.name))snapshots[boundary.name]=value.DownloadToArray();
                    }
                    if(stage<kernels.Length){var value=kernels[stage].Execute(values[inputs[stage]]);owned.Add(value);values[outputs[stage]]=value;}
                }
                snapshots["__final"]=values[models.Last().outputs.Single().name].DownloadToArray();
                Assert.That(snapshots["__final"].Select(BitConverter.SingleToInt32Bits),Is.EqualTo(baseline.Select(BitConverter.SingleToInt32Bits)),"Diagnostic downloads changed normal final bits.");
                Assert.That(snapshots.Keys.Where(k=>k!="__final"),Is.EquivalentTo(wanted));return snapshots;
            }
            finally {foreach(var value in owned)value.Dispose();foreach(var worker in workers)worker.Dispose();}
        }
        static float[] Floats(string path){var bytes=File.ReadAllBytes(path);var values=new float[bytes.Length/4];Buffer.BlockCopy(bytes,0,values,0,bytes.Length);return values;}
    }
}
