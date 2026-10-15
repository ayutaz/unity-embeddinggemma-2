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
        static Type RequiredSilu()
        {
            var type=typeof(GpuBlockedMatmul).Assembly.GetType("EmbeddingGemma.GpuSilu");
            Assert.That(type,Is.Not.Null,"Native-rounding GPU SiLU is missing.");return type;
        }
        [Test]
        public void GpuSiluOwnsOutputsRejectsCpuAndPreservesBorrowedInputAfterDispose()
        {
            Assert.That(SystemInfo.supportsComputeShaders,Is.True,"Actual GPU required; no skip.");
            var type=RequiredSilu();var operation=(IDisposable)Activator.CreateInstance(type);var execute=type.GetMethod("Execute");
            using var cpu=new Tensor<float>(new TensorShape(1,2),new[]{0f,0f});
            var error=Assert.Throws<TargetInvocationException>(()=>execute.Invoke(operation,new object[]{cpu}));Assert.That(error.InnerException,Is.InstanceOf<ArgumentException>());
            using var input=new Tensor<float>(new TensorShape(1,2),new[]{BitConverter.Int32BitsToSingle(unchecked((int)0x80000000)),0f});ComputeTensorData.Pin(input);
            using var output=(Tensor<float>)execute.Invoke(operation,new object[]{input});
            using var other=(Tensor<float>)execute.Invoke(operation,new object[]{input});operation.Dispose();operation.Dispose();
            Assert.That(output.backendType,Is.EqualTo(BackendType.GPUCompute));Assert.That(output.shape,Is.EqualTo(input.shape));
            Assert.That(output.DownloadToArray().Select(BitConverter.SingleToInt32Bits),Is.EqualTo(input.DownloadToArray().Select(BitConverter.SingleToInt32Bits)));
            Assert.That(other.DownloadToArray().Select(BitConverter.SingleToInt32Bits),Is.EqualTo(input.DownloadToArray().Select(BitConverter.SingleToInt32Bits)));
        }
        [Test]
        public void FrozenFirstActivationGpuMustMatchOriginalNativeBits()
        {
            Assert.That(SystemInfo.supportsComputeShaders,Is.True,"Actual GPU required; no skip.");
            var source=Environment.GetEnvironmentVariable("EG_ACCEPTANCE_SOURCE");
            var output=Path.GetFullPath(Path.Combine(Application.dataPath,"../artifacts/frozen-first-silu-gpu-"+source+".json"));Assert.That(File.Exists(output),Is.False);
            var report=new JObject{["success"]=false,["scope"]="one_frozen_original_silu_gpu_bit_gate",["runtime_source"]=source,["full_encoder_validated"]=false,["end_to_end_validated"]=false};
            void Save(){Directory.CreateDirectory(Path.GetDirectoryName(output));File.WriteAllText(output,report.ToString());}
            void Check(string path,string sha){using var stream=File.OpenRead(path);Assert.That(PlayerFileHash.Compute(stream,out _),Is.EqualTo(sha));}
            Save();
            try
            {
                var root=Path.GetFullPath(Path.Combine(Application.dataPath,"../../k/artifacts/audio-native-feed-forward-replay"));
                var path=Path.Combine(root,"audio-layer-replay-report.json");Check(path,"99529dc0fa6c74825b23fca653d82091797ff1e5267d26c5cdae723fcf86f501");
                var replay=JObject.Parse(File.ReadAllText(path));Assert.That(replay.Value<bool>("native_final_exact_match"),Is.True);
                var x=replay["parts"]["feed_forward1.ffw_layer_1"];var y=replay["parts"]["feed_forward1.act_fn"];
                var inputPath=Path.Combine(root,"audio-layer-replay",x.Value<string>("file"));var expectedPath=Path.Combine(root,"audio-layer-replay",y.Value<string>("file"));
                Check(inputPath,x.Value<string>("sha256"));Check(expectedPath,y.Value<string>("sha256"));
                var type=RequiredSilu();using var operation=(IDisposable)Activator.CreateInstance(type);
                using var input=new Tensor<float>(new TensorShape(1,280,4096),Floats(inputPath));ComputeTensorData.Pin(input);
                using var result=(Tensor<float>)type.GetMethod("Execute").Invoke(operation,new object[]{input});Assert.That(result.backendType,Is.EqualTo(BackendType.GPUCompute));
                var actual=result.DownloadToArray();var expected=Floats(expectedPath);Assert.That(actual.Length,Is.EqualTo(expected.Length));int bits=0;
                for(var index=0;index<actual.Length;index++)if(BitConverter.SingleToInt32Bits(actual[index])!=BitConverter.SingleToInt32Bits(expected[index]))bits++;
                report["actual_backend"]=result.backendType.ToString();report["count"]=actual.Length;report["bit_differences"]=bits;
                var bytes=new byte[actual.Length*4];Buffer.BlockCopy(actual,0,bytes,0,bytes.Length);File.WriteAllBytes(Path.ChangeExtension(output,"bin"),bytes);
                Save();Assert.That(bits,Is.Zero,"Preserve original native SiLU rounding.");report["success"]=true;
            }
            catch(Exception error){report["error"]=error.ToString();throw;}
            finally {Save();}
        }

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
            var snapshots=(Dictionary<string,float[]>)method.Invoke(null,new object[]{new[]{prefix,tail},new[]{"input"},new[]{"projected"},new[]{kernel},input,new HashSet<string>{"input","projected"}});
            Assert.That(snapshots["input"],Is.EqualTo(new[]{-3f,4f}));
            Assert.That(snapshots["projected"],Is.EqualTo(new[]{-6f,12f}));
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

        [Test]
        public void OriginalLayer10InputGpuMustMatchOriginalLayer10Output()
            =>Execute("silence",true,false,true);

        [Test]
        public void ObserveFirstFeedForwardBoundariesWithoutChangingNormalFinalBits()
            =>Execute("silence",true,false,false,true);

        [TestCase("silence")]
        [TestCase("impulse")]
        [TestCase("sine")]
        [TestCase("nyquist")]
        [TestCase("noise")]
        [TestCase("padded_noise")]
        public void NativeSiluPartitionsGpuMatchOriginalGolden(string signal)
            =>Execute(signal,true,false,false,false,true);

        [Test]
        public void ObserveNativeSiluFirstFeedForwardWithoutChangingNormalFinalBits()
            =>Execute("silence",true,false,false,true,true);

        void Execute(string signal,bool allStatic,bool observe=false,bool isolated=false,bool early=false,bool nativeSilu=false)
        {
            Assert.That(SystemInfo.supportsComputeShaders,Is.True,"Actual GPU required; never skip.");
            var source=Environment.GetEnvironmentVariable("EG_ACCEPTANCE_SOURCE");
            var folder=nativeSilu?"GpuAudioPartitionedNativeSilu":isolated?"GpuAudioPartitionedLayer10":allStatic?"GpuAudioPartitionedAllStatic":"GpuAudioPartitionedFirstFf";
            var root=Path.Combine(Application.dataPath,folder);
            var originalRoot=Path.GetFullPath(Path.Combine(Application.dataPath,"../../k/artifacts/audio-real-chain/audio-runtime-reference"));
            var stem=nativeSilu?(early?"partitioned-native-silu-early-observation-":"partitioned-native-silu-gpu-"):
                early?"partitioned-early-audio-observation-":isolated?"partitioned-layer10-original-input-":observe?"partitioned-all-static-layer-observation-":allStatic?"partitioned-all-static-gpu-":"partitioned-first-ff-gpu-";
            var output=Path.GetFullPath(Path.Combine(Application.dataPath,"../artifacts/"+stem+source+"-"+signal+".json"));
            Assert.That(File.Exists(output),Is.False);
            var report=new JObject { ["success"]=false,["scope"]=early?"original_early_audio_observation_only":isolated?"one_original_audio_layer_isolation_gpu_numeric_gate":observe?"original_audio_layer_observation_only":"one_partitioned_original_audio_encoder_gpu_numeric_gate",
                ["runtime_source"]=source,["signal"]=signal,["requested_backend"]="GPUCompute",
                ["gpu"]=SystemInfo.graphicsDeviceName,["phase"]="audit",["end_to_end_validated"]=false,["all_original_signals_validated"]=false };
            void Save(){Directory.CreateDirectory(Path.GetDirectoryName(output));File.WriteAllText(output,report.ToString());}
            void Check(string path,string sha){using var stream=File.OpenRead(path);Assert.That(PlayerFileHash.Compute(stream,out _),Is.EqualTo(sha));}
            Save();
            var kernels=new List<IGpuTensorOperation>();
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
                float[] inputValues,expected;int[] inputShape;
                if(isolated)
                {
                    Assert.That(plan.Value<int>("layer"),Is.EqualTo(10));Assert.That(((JArray)plan["projections"]).Count,Is.EqualTo(11));
                    var stageRoot=Path.GetFullPath(Path.Combine(Application.dataPath,"../../k/artifacts/audio-current-stage-context"));
                    var stageReferencePath=Path.Combine(stageRoot,"audio-stage-reference.json");var stageAuditPath=Path.Combine(stageRoot,"audio-stage-validation.json");
                    Check(stageReferencePath,"6620f335563a91ccc484df7d9b89d4d889e58dab4ec3d7d0389b36abe36b454a");
                    Check(stageAuditPath,"3edee8acdd9ebebb819ae488b753a3a94949b953aa0906995e516764932b0f9b");
                    var stageReference=JObject.Parse(File.ReadAllText(stageReferencePath));var stageAudit=JObject.Parse(File.ReadAllText(stageAuditPath));
                    Assert.That(stageAudit.Value<bool>("success")&&stageAudit.Value<bool>("context_validated"),Is.True);
                    Assert.That(stageReference["metadata"].Value<string>("source_commit"),Is.EqualTo("4edbe17a8105233c5aaa1514faa2862b08a4f7e0"));
                    var stageCase=((JArray)stageReference["cases"]).Single(c=>c.Value<string>("id")==signal);
                    float[] Native(string name)
                    {
                        var stage=((JArray)stageCase["stages"]).Single(s=>s.Value<string>("name")==name);var file=stage.Value<string>("expected");
                        var path=Path.Combine(stageRoot,file);if(!File.Exists(path))path=Path.Combine(stageRoot,Path.GetFileName(file));
                        Check(path,stageAudit["files"][file].Value<string>("sha256"));return Floats(path);
                    }
                    inputValues=Native("layer_9");expected=Native("layer_10");inputShape=new[]{1,280,1024};
                    Assert.That(inputValues.Length,Is.EqualTo(280*1024));Assert.That(expected.Length,Is.EqualTo(280*1024));
                    report["input_stage"]="layer_9";report["output_stage"]="layer_10";report["full_encoder_validated"]=false;
                }
                else
                {
                    inputValues=Floats(Path.Combine(originalRoot,inputName));inputShape=new[]{1,1119,128};
                    expected=row["features"].ToObject<float[][]>().SelectMany(v=>v).ToArray();
                }
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
                var operations=(JArray)plan[nativeSilu?"operations":"projections"];
                if(nativeSilu){Assert.That(plan.Value<int>("silu_count"),Is.EqualTo(36));Assert.That(operations.Count,Is.EqualTo(171));}
                foreach(var projection in operations)
                {
                    if(projection.Value<string>("kind")=="silu")kernels.Add(new GpuSilu());
                    else
                    {
                        var shape=projection["weight_shape"].ToObject<int[]>();var weightPath=Path.Combine(root,projection.Value<string>("file"));
                        Check(weightPath,projection.Value<string>("weight_sha256"));kernels.Add(new GpuBlockedMatmul(shape[0],shape[1],Floats(weightPath)));
                    }
                    inputs.Add(projection.Value<string>("input"));outputs.Add(projection.Value<string>("output"));
                }
                report["segment_layer_counts"]=layers;report["prepare_ms"]=timer.Elapsed.TotalMilliseconds;
                report["projection_count"]=((JArray)plan["projections"]).Count;report["operation_count"]=kernels.Count;report["silu_count"]=nativeSilu?36:0;
                using var pipeline=new GpuProjectionPipeline(models.ToArray(),inputs.ToArray(),outputs.ToArray(),kernels.ToArray());
                report["cpu_computation_or_input_count"]=0;report["readback_flush_count"]=0;
                using var input=new Tensor<float>(new TensorShape(inputShape),inputValues);ComputeTensorData.Pin(input);
                report["phase"]="execute";Save();timer.Restart();
                using var encoded=pipeline.Execute(input);Assert.That(encoded.backendType,Is.EqualTo(BackendType.GPUCompute));
                var actual=encoded.DownloadToArray();report["execute_and_final_download_ms"]=timer.Elapsed.TotalMilliseconds;
                Assert.That(actual.Length,Is.EqualTo(expected.Length));
                int mismatch=0;double maximum=0,ratio=0;
                for(var i=0;i<actual.Length;i++)
                {
                    Assert.That(float.IsNaN(actual[i])||float.IsInfinity(actual[i]),Is.False);
                    var difference=Math.Abs((double)actual[i]-expected[i]);var allowed=1e-5+1e-4*Math.Abs((double)expected[i]);
                    maximum=Math.Max(maximum,difference);ratio=Math.Max(ratio,difference/allowed);if(difference>allowed)mismatch++;
                }
                report["actual_backend"]=encoded.backendType.ToString();report["mismatched_elements"]=mismatch;
                report["maximum_absolute_error"]=maximum;report["maximum_tolerance_ratio"]=ratio;report["phase"]="completed";Save();
                if(early)
                {
                    var nativeRoot=Path.GetFullPath(Path.Combine(Application.dataPath,"../../k/artifacts/audio-native-feed-forward-replay"));
                    var replayPath=Path.Combine(nativeRoot,"audio-layer-replay-report.json");
                    Check(replayPath,"99529dc0fa6c74825b23fca653d82091797ff1e5267d26c5cdae723fcf86f501");
                    var replay=JObject.Parse(File.ReadAllText(replayPath));
                    Assert.That(replay.Value<bool>("success")&&replay.Value<bool>("native_final_exact_match"),Is.True);
                    Assert.That(replay.Value<string>("reference_source"),Is.EqualTo("4edbe17a8105233c5aaa1514faa2862b08a4f7e0"));
                    Assert.That(plan["projections"][1].Value<string>("weight_sha256"),Is.EqualTo("d598344ee23d0545a76fa6d95173fbc886e8f2dd46d5907e6bd50b6cfa4891df"));
                    Assert.That(plan["projections"][2].Value<string>("weight_sha256"),Is.EqualTo("50ff67bc3a9a8704f925183042667884c6a2114de99565c2fa93f79570bd77b6"));
                    var bindings=new[]{
                        (name:"input_projection",boundary:"/subsample_conv_projection/input_proj_linear/MatMul_output_0",part:"",minimum:float.NegativeInfinity,maximum:float.PositiveInfinity),
                        (name:"pre_layer_norm_input",boundary:"/feed_forward1/ffw_layer_1/Clip_output_0",part:"feed_forward1.pre_layer_norm",minimum:-12.875f,maximum:12.8125f),
                        (name:"activation_input",boundary:"/feed_forward1/ffw_layer_2/Clip_output_0",part:"feed_forward1.act_fn",minimum:-11.1875f,maximum:11.0625f),
                        (name:"feed_forward1_output",boundary:"/feed_forward1/Add_output_0",part:"feed_forward1",minimum:float.NegativeInfinity,maximum:float.PositiveInfinity),
                        (name:"norm_pre_attn",boundary:"/norm_pre_attn/Mul_1_output_0",part:"norm_pre_attn",minimum:float.NegativeInfinity,maximum:float.PositiveInfinity)};
                    report["phase"]="observe_early";Save();
                    var snapshots=ObserveBoundaries(models.ToArray(),inputs.ToArray(),outputs.ToArray(),kernels.ToArray(),input,bindings.Select(b=>b.boundary).ToHashSet());
                    Assert.That(snapshots["__final"].Select(BitConverter.SingleToInt32Bits),Is.EqualTo(actual.Select(BitConverter.SingleToInt32Bits)));
                    report["normal_final_matches_observed_bits"]=true;report["full_encoder_validated"]=false;
                    var captureRoot=Path.ChangeExtension(output,null)+"-captures";Assert.That(Directory.Exists(captureRoot),Is.False);Directory.CreateDirectory(captureRoot);
                    report["capture_directory"]=captureRoot;var stages=new JArray();report["stages"]=stages;
                    foreach(var binding in bindings)
                    {
                        string path,expectedSha;
                        if(binding.part.Length==0)
                        {
                            path=Path.GetFullPath(Path.Combine(Application.dataPath,"../../numerical-acceptance-state/subsample-native-replay/audio-layer-replay/subsample.input_proj_linear-fp32.bin"));
                            expectedSha="dc54774840fed958a310c6092e5c5c3483de74be52df7914d2bd128a99192969";
                        }
                        else
                        {
                            var part=replay["parts"][binding.part];path=Path.Combine(nativeRoot,"audio-layer-replay",part.Value<string>("file"));expectedSha=part.Value<string>("sha256");
                        }
                        Check(path,expectedSha);var native=Floats(path);var values=snapshots[binding.boundary];Assert.That(values.Length,Is.EqualTo(native.Length));
                        Assert.That(values.Any(v=>float.IsNaN(v)||float.IsInfinity(v)),Is.False);
                        int bits=0,bad=0;double maxRatio=0;
                        for(var index=0;index<values.Length;index++)
                        {
                            // Transform only the immutable expected values to the original
                            // quantized-linear input Clip boundary. Product computation stays GPU.
                            var target=Math.Max(binding.minimum,Math.Min(binding.maximum,native[index]));
                            if(BitConverter.SingleToInt32Bits(values[index])!=BitConverter.SingleToInt32Bits(target))bits++;
                            var difference=Math.Abs((double)values[index]-target);var allowed=1e-5+1e-4*Math.Abs((double)target);
                            if(difference>allowed)bad++;maxRatio=Math.Max(maxRatio,difference/allowed);
                        }
                        var bytes=new byte[values.Length*4];Buffer.BlockCopy(values,0,bytes,0,bytes.Length);var capture=Path.Combine(captureRoot,binding.name+"-fp32.bin");File.WriteAllBytes(capture,bytes);
                        using var stream=File.OpenRead(capture);var captureSha=PlayerFileHash.Compute(stream,out _);
                        stages.Add(new JObject{["stage"]=binding.name,["boundary"]=binding.boundary,["original_native_sha256"]=expectedSha,
                            ["expected_transform"]=binding.part=="feed_forward1.pre_layer_norm"||binding.part=="feed_forward1.act_fn"?"original_fp32_input_clip":"identity",
                            ["actual_file"]=capture,["actual_sha256"]=captureSha,["bit_differences"]=bits,["mismatched_elements"]=bad,["maximum_tolerance_ratio"]=maxRatio});Save();
                    }
                    Assert.That(stages.Count,Is.EqualTo(5));report["observation_completed"]=true;report["success"]=true;report["phase"]="completed_observation";return;
                }
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
        static Dictionary<string,float[]> ObserveBoundaries(Model[] models,string[] inputs,string[] outputs,IGpuTensorOperation[] kernels,Tensor<float> input,HashSet<string> wanted)
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
