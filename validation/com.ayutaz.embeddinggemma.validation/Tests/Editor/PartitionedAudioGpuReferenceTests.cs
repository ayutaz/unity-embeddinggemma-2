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

        [Test]
        public void BinaryBoundaryObservationPreservesNormalBitsAndBorrowedGpuResources()
        {
            Assert.That(SystemInfo.supportsComputeShaders,Is.True,"Actual GPU required; no skip.");
            var method=typeof(PartitionedAudioGpuReferenceTests).GetMethod("ObserveOperationBoundaries",BindingFlags.Static|BindingFlags.NonPublic);
            Assert.That(method,Is.Not.Null,"Binary boundary observation is missing.");
            var a=new FunctionalGraph();var x=a.AddInput<float>(new TensorShape(2,3),"input");
            var prefix=a.Compile(x,Functional.Transpose(x,0,1));
            var named=prefix.outputs[0];named.name="left";prefix.outputs[0]=named;
            named=prefix.outputs[1];named.name="right";prefix.outputs[1]=named;
            var b=new FunctionalGraph();var product=b.AddInput<float>(new TensorShape(2,2),"product");var tail=b.Compile(product);
            named=tail.outputs[0];named.name="final";tail.outputs[0]=named;
            using var kernel=new GpuBatchedMatmul();using var input=new Tensor<float>(new TensorShape(2,3),new[]{1f,2f,3f,4f,5f,6f});ComputeTensorData.Pin(input);
            var snapshots=(Dictionary<string,float[]>)method.Invoke(null,new object[]{new[]{prefix,tail},new[]{new[]{"left","right"}},new[]{"product"},new IGpuTensorOperation[]{kernel},input,new HashSet<string>{"right","product"}});
            Assert.That(snapshots["product"],Is.EqualTo(new[]{14f,32f,32f,77f}));
            Assert.That(snapshots["__final"],Is.EqualTo(snapshots["product"]));
            Assert.That(input.DownloadToArray(),Is.EqualTo(new[]{1f,2f,3f,4f,5f,6f}));
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

        [TestCase("silence")]
        [TestCase("impulse")]
        [TestCase("sine")]
        [TestCase("nyquist")]
        [TestCase("noise")]
        [TestCase("padded_noise")]
        public void NativeBmmPartitionsGpuMatchOriginalGolden(string signal)
            =>Execute(signal,true,nativeSilu:true,nativeBmm:true);

        [Test]
        public void ObserveConnectedNativeAttentionOperandsWithoutChangingNormalFinalBits()
            =>Execute("silence",true,nativeSilu:true,nativeBmm:true,observeBmm:true);

        [TestCase("silence")]
        [TestCase("impulse")]
        [TestCase("sine")]
        [TestCase("nyquist")]
        [TestCase("noise")]
        [TestCase("padded_noise")]
        public void NativeQueryScalePartitionsGpuMatchOriginalGolden(string signal)
            =>Execute(signal,true,nativeSilu:true,nativeBmm:true,nativeQueryScale:true);

        [Test]
        public void ObserveNativeQueryScaleConnectedAttentionOperandsWithoutChangingNormalFinalBits()
            =>Execute("silence",true,nativeSilu:true,nativeBmm:true,observeBmm:true,nativeQueryScale:true);

        [TestCase("silence")]
        [TestCase("impulse")]
        [TestCase("sine")]
        [TestCase("nyquist")]
        [TestCase("noise")]
        [TestCase("padded_noise")]
        public void NativeTanhPartitionsGpuMatchOriginalGolden(string signal)
            =>Execute(signal,true,nativeSilu:true,nativeBmm:true,nativeQueryScale:true,nativeTanh:true);

        [Test]
        public void ObserveNativeTanhConnectedAttentionOperandsWithoutChangingNormalFinalBits()
            =>Execute("silence",true,nativeSilu:true,nativeBmm:true,observeBmm:true,nativeQueryScale:true,nativeTanh:true);

        [TestCase("silence")]
        [TestCase("impulse")]
        [TestCase("sine")]
        [TestCase("nyquist")]
        [TestCase("noise")]
        [TestCase("padded_noise")]
        public void NativeDivisionPartitionsGpuMatchOriginalGolden(string signal)
            =>Execute(signal,true,nativeSilu:true,nativeBmm:true,nativeQueryScale:true,nativeTanh:true,nativeDivision:true);

        [Test]
        public void ObserveNativeDivisionConnectedAttentionOperandsWithoutChangingNormalFinalBits()
            =>Execute("silence",true,nativeSilu:true,nativeBmm:true,observeBmm:true,nativeQueryScale:true,nativeTanh:true,nativeDivision:true);

        [TestCase("silence")]
        [TestCase("impulse")]
        [TestCase("sine")]
        [TestCase("nyquist")]
        [TestCase("noise")]
        [TestCase("padded_noise")]
        public void NativeSoftmaxPartitionsGpuMatchOriginalGolden(string signal)
            =>Execute(signal,true,nativeSilu:true,nativeBmm:true,nativeQueryScale:true,nativeTanh:true,nativeDivision:true,nativeSoftmax:true);

        [Test]
        public void ObserveNativeSoftmaxConnectedAttentionOperandsWithoutChangingNormalFinalBits()
            =>Execute("silence",true,nativeSilu:true,nativeBmm:true,observeBmm:true,nativeQueryScale:true,nativeTanh:true,nativeDivision:true,nativeSoftmax:true);

        [Test]
        public void ObserveNativeSoftmaxPostAndConvolutionWithoutChangingNormalFinalBits()
            =>Execute("silence",true,early:true,nativeSilu:true,attention:true,nativeBmm:true,nativeQueryScale:true,nativeTanh:true,nativeDivision:true,nativeSoftmax:true);

        [Test]
        public void ObserveNativeConvolutionModulesWithoutChangingNormalFinalBits()
            =>Execute("silence",true,early:true,nativeSilu:true,attention:true,nativeBmm:true,nativeQueryScale:true,nativeTanh:true,nativeDivision:true,nativeSoftmax:true,convolutionDetail:true);

        [TestCase("silence")]
        [TestCase("impulse")]
        [TestCase("sine")]
        [TestCase("nyquist")]
        [TestCase("noise")]
        [TestCase("padded_noise")]
        public void NativeGluPartitionsGpuMatchOriginalGolden(string signal)
            =>Execute(signal,true,nativeSilu:true,nativeBmm:true,nativeQueryScale:true,nativeTanh:true,nativeDivision:true,nativeSoftmax:true,nativeGlu:true);

        [TestCase("silence")]
        [TestCase("impulse")]
        [TestCase("sine")]
        [TestCase("nyquist")]
        [TestCase("noise")]
        [TestCase("padded_noise")]
        public void FirstNormFmaPartitionsGpuMatchOriginalGolden(string signal)
            =>Execute(signal,true,nativeSilu:true,nativeBmm:true,nativeQueryScale:true,nativeTanh:true,nativeDivision:true,nativeSoftmax:true,nativeGlu:true,firstNormFma:true);

        [TestCase("silence")]
        [TestCase("impulse")]
        [TestCase("sine")]
        [TestCase("nyquist")]
        [TestCase("noise")]
        [TestCase("padded_noise")]
        public void ScalarMomentFmaPartitionsGpuMatchOriginalGolden(string signal)
            =>Execute(signal,true,nativeSilu:true,nativeBmm:true,nativeQueryScale:true,nativeTanh:true,nativeDivision:true,nativeSoftmax:true,nativeGlu:true,firstNormFma:true,scalarMomentFma:true);

        [Test]
        public void PartitionedAudioEncoderAdapterMatchesOriginalImpulseOnGpu()
            =>Execute("impulse",true,nativeSilu:true,nativeBmm:true,nativeQueryScale:true,nativeTanh:true,nativeDivision:true,nativeSoftmax:true,nativeGlu:true,firstNormFma:true,scalarMomentFma:true,adapterMode:true);

        [Test]
        public void ObserveNativeGluConvolutionModulesWithoutChangingNormalFinalBits()
            =>Execute("silence",true,early:true,nativeSilu:true,attention:true,nativeBmm:true,nativeQueryScale:true,nativeTanh:true,nativeDivision:true,nativeSoftmax:true,convolutionDetail:true,nativeGlu:true);

        [Test]
        public void ObserveNativeGluAllLayersWithoutChangingNormalFinalBits()
            =>Execute("silence",true,observe:true,nativeSilu:true,nativeBmm:true,nativeQueryScale:true,nativeTanh:true,nativeDivision:true,nativeSoftmax:true,nativeGlu:true);

        [Test]
        public void ObserveOriginalImpulseAllLayersWithoutChangingNormalFinalBits()
            =>Execute("impulse",true,observe:true,nativeSilu:true,nativeBmm:true,nativeQueryScale:true,nativeTanh:true,nativeDivision:true,nativeSoftmax:true,nativeGlu:true);

        [Test]
        public void ObserveOriginalLayerOnePartsWithoutChangingNormalFinalBits()
            =>Execute("silence",true,nativeSilu:true,nativeBmm:true,nativeQueryScale:true,nativeTanh:true,nativeDivision:true,nativeSoftmax:true,nativeGlu:true,layerObservation:1);

        [Test]
        public void ObserveOriginalLayerThreePartsWithoutChangingNormalFinalBits()
            =>Execute("silence",true,nativeSilu:true,nativeBmm:true,nativeQueryScale:true,nativeTanh:true,nativeDivision:true,nativeSoftmax:true,nativeGlu:true,layerObservation:3);

        [Test]
        public void ObserveNativeSiluFirstFeedForwardWithoutChangingNormalFinalBits()
            =>Execute("silence",true,false,false,true,true);

        [Test]
        public void ObserveFirstAttentionAndConvolutionWithoutChangingNormalFinalBits()
            =>Execute("silence",true,false,false,true,true,true);

        void Execute(string signal,bool allStatic,bool observe=false,bool isolated=false,bool early=false,bool nativeSilu=false,bool attention=false,bool nativeBmm=false,bool observeBmm=false,bool nativeQueryScale=false,bool nativeTanh=false,bool nativeDivision=false,bool nativeSoftmax=false,bool convolutionDetail=false,bool nativeGlu=false,int layerObservation=0,bool firstNormFma=false,bool scalarMomentFma=false,bool adapterMode=false)
        {
            Assert.That(SystemInfo.supportsComputeShaders,Is.True,"Actual GPU required; never skip.");
            var source=Environment.GetEnvironmentVariable("EG_ACCEPTANCE_SOURCE");
            var folder=nativeGlu?"GpuAudioPartitionedNativeGlu":nativeSoftmax?"GpuAudioPartitionedNativeSoftmax":nativeDivision?"GpuAudioPartitionedNativeDivision":nativeTanh?"GpuAudioPartitionedNativeTanh":nativeQueryScale?"GpuAudioPartitionedNativeQueryScale":nativeBmm?"GpuAudioPartitionedNativeBmm":nativeSilu?"GpuAudioPartitionedNativeSilu":isolated?"GpuAudioPartitionedLayer10":allStatic?"GpuAudioPartitionedAllStatic":"GpuAudioPartitionedFirstFf";
            var root=Path.Combine(Application.dataPath,folder);
            var originalRoot=Path.GetFullPath(Path.Combine(Application.dataPath,"../../k/artifacts/audio-real-chain/audio-runtime-reference"));
            var stem=nativeSoftmax?(observeBmm?"partitioned-native-softmax-operands-observation-":"partitioned-native-softmax-gpu-"):nativeDivision?(observeBmm?"partitioned-native-division-operands-observation-":"partitioned-native-division-gpu-"):nativeTanh?(observeBmm?"partitioned-native-tanh-operands-observation-":"partitioned-native-tanh-gpu-"):nativeQueryScale?(observeBmm?"partitioned-native-query-scale-operands-observation-":"partitioned-native-query-scale-gpu-"):observeBmm?"partitioned-native-bmm-operands-observation-":nativeBmm?"partitioned-native-bmm-gpu-":attention?"partitioned-native-silu-attention-observation-":nativeSilu?(early?"partitioned-native-silu-early-observation-":"partitioned-native-silu-gpu-"):
                early?"partitioned-early-audio-observation-":isolated?"partitioned-layer10-original-input-":observe?"partitioned-all-static-layer-observation-":allStatic?"partitioned-all-static-gpu-":"partitioned-first-ff-gpu-";
            if(nativeSoftmax&&early&&attention)stem="partitioned-native-softmax-post-convolution-observation-";
            if(convolutionDetail)stem="partitioned-native-convolution-modules-observation-";
            if(nativeGlu)stem=convolutionDetail?"partitioned-native-glu-convolution-observation-":"partitioned-native-glu-gpu-";
            if(nativeGlu&&observe)stem="partitioned-native-glu-layer-observation-";
            if(layerObservation>0)stem="partitioned-native-layer"+layerObservation+"-parts-observation-";
            if(firstNormFma)stem="partitioned-first-norm-fma-gpu-";
            if(scalarMomentFma)stem="partitioned-scalar-moment-fma-gpu-";
            if(adapterMode)stem="partitioned-audio-encoder-adapter-gpu-";
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
                    var segmentFile=segment.Value<string>("file");
                    if(firstNormFma&&segmentFile=="segment-0.onnx")
                    {
                        Assert.That(nativeGlu,Is.True);segmentFile=scalarMomentFma?"segment-0-scalar-moment-fma.onnx":"segment-0-first-norm-fma.onnx";
                        var segmentSha=scalarMomentFma?"650929620cdb4974eec0ee41d1d21c6dfe376c6703cb9cf5091d261171dbd949":"9bc00c121b589af7444e026532b4c01452a21b32b1b2b290e2377b55aa6fe19b";
                        Check(Path.Combine(root,segmentFile),segmentSha);
                        report[scalarMomentFma?"scalar_moment_fma_segment_sha256":"first_norm_fma_segment_sha256"]=segmentSha;
                    }
                    var asset=AssetDatabase.LoadAssetAtPath<ModelAsset>("Assets/"+folder+"/"+segmentFile);
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
                var modelLoadMs=timer.Elapsed.TotalMilliseconds;report["model_load_and_transform_ms"]=modelLoadMs;
                var inputs=new List<string>();var bindingsByOperation=new List<string[]>();var outputs=new List<string>();
                var operations=(JArray)plan[nativeSilu?"operations":"projections"];
                if(nativeSilu){Assert.That(plan.Value<int>("silu_count"),Is.EqualTo(36));Assert.That(operations.Count,Is.EqualTo(nativeGlu?267:nativeSoftmax?255:nativeDivision?243:nativeTanh?231:nativeQueryScale?219:nativeBmm?207:171));}
                if(nativeBmm)Assert.That(plan.Value<int>("dynamic_matmul_count"),Is.EqualTo(36));
                if(nativeQueryScale)Assert.That(plan.Value<int>("query_scale_count"),Is.EqualTo(12));
                if(nativeTanh)Assert.That(plan.Value<int>("tanh_count"),Is.EqualTo(12));
                if(nativeDivision)Assert.That(plan.Value<int>("scalar_divide_count"),Is.EqualTo(12));
                if(nativeSoftmax)Assert.That(plan.Value<int>("softmax_count"),Is.EqualTo(12));
                if(nativeGlu)Assert.That(plan.Value<int>("glu_count"),Is.EqualTo(12));
                foreach(var projection in operations)
                {
                    if(projection.Value<string>("kind")=="silu")kernels.Add(new GpuSilu());
                    else if(projection.Value<string>("kind")=="tanh")kernels.Add(new GpuTanh());
                    else if(projection.Value<string>("kind")=="scalar_divide")kernels.Add(new GpuScalarDivide(projection.Value<float>("divisor")));
                    else if(projection.Value<string>("kind")=="softmax")kernels.Add(new GpuSoftmax());
                    else if(projection.Value<string>("kind")=="glu")kernels.Add(new GpuGlu());
                    else if(projection.Value<string>("kind")=="batched_matmul")kernels.Add(new GpuBatchedMatmul());
                    else if(projection.Value<string>("kind")=="sequential_scale")
                    {
                        var weightPath=Path.Combine(root,projection.Value<string>("file"));Check(weightPath,projection.Value<string>("weight_sha256"));
                        kernels.Add(new GpuSequentialScale(projection.Value<float>("first_scale"),Floats(weightPath)));
                    }
                    else
                    {
                        var shape=projection["weight_shape"].ToObject<int[]>();var weightPath=Path.Combine(root,projection.Value<string>("file"));
                        Check(weightPath,projection.Value<string>("weight_sha256"));kernels.Add(new GpuBlockedMatmul(shape[0],shape[1],Floats(weightPath)));
                    }
                    inputs.Add(projection.Value<string>("input"));outputs.Add(projection.Value<string>("output"));
                    bindingsByOperation.Add(projection["inputs"]?.ToObject<string[]>()??new[]{projection.Value<string>("input")});
                }
                report["segment_layer_counts"]=layers;report["prepare_ms"]=timer.Elapsed.TotalMilliseconds;
                report["native_operation_prepare_ms"]=timer.Elapsed.TotalMilliseconds-modelLoadMs;
                report["projection_count"]=((JArray)plan["projections"]).Count;report["operation_count"]=kernels.Count;report["silu_count"]=nativeSilu?36:0;
                var workerTimer=System.Diagnostics.Stopwatch.StartNew();
                if(adapterMode)
                {
                    Assert.That(scalarMomentFma&&!observe&&!isolated&&!early&&!observeBmm&&layerObservation==0,Is.True);
                    Assert.That(plan.Value<string>("profile"),Is.EqualTo("frames-1119-valid-1119"));
                }
                using var adapter=adapterMode?new PartitionedAudioFeatureEncoder(models.ToArray(),bindingsByOperation.ToArray(),outputs.ToArray(),kernels.ToArray(),new AudioEncoderProfile(1119)):null;
                using var pipeline=adapterMode?null:new GpuProjectionPipeline(models.ToArray(),bindingsByOperation.ToArray(),outputs.ToArray(),kernels.ToArray());
                report["worker_creation_ms"]=workerTimer.Elapsed.TotalMilliseconds;
                report["dynamic_matmul_count"]=nativeBmm?36:0;
                report["query_scale_count"]=nativeQueryScale?12:0;
                report["tanh_count"]=nativeTanh?12:0;
                report["scalar_divide_count"]=nativeDivision?12:0;
                report["softmax_count"]=nativeSoftmax?12:0;
                report["glu_count"]=nativeGlu?12:0;
                report["cpu_computation_or_input_count"]=0;report["readback_flush_count"]=0;
                workerTimer.Restart();
                using var input=new Tensor<float>(new TensorShape(inputShape),inputValues);ComputeTensorData.Pin(input);
                using var frameMask=adapterMode?new Tensor<int>(new TensorShape(1,1119),Enumerable.Repeat(1,1119).ToArray()):null;
                if(frameMask!=null)ComputeTensorData.Pin(frameMask);
                using var preprocessing=adapterMode?(AudioPreprocessingResult)Activator.CreateInstance(typeof(AudioPreprocessingResult),BindingFlags.Instance|BindingFlags.NonPublic,null,new object[]{input,frameMask,1119},null):null;
                report["input_tensor_creation_and_pin_ms"]=workerTimer.Elapsed.TotalMilliseconds;
                using var process=System.Diagnostics.Process.GetCurrentProcess();
                GpuExecutionProfiler.ReadProcessMemory(process,out var workingSetBefore,out _);
                report["host_working_set_before_execute_bytes"]=workingSetBefore;
                report["phase"]="execute";Save();timer.Restart();
                using var adapterResult=adapterMode?adapter.Encode(preprocessing):null;
                using var encoded=adapterMode?adapterResult.SoftTokens:pipeline.Execute(input);Assert.That(encoded.backendType,Is.EqualTo(BackendType.GPUCompute));
                if(adapterMode)
                {
                    Assert.That(adapterResult.ValidFrameCount,Is.EqualTo(1119));
                    Assert.That(encoded.shape,Is.EqualTo(new TensorShape(280,512)));
                    report["encoder_api_adapter_validated"]=true;report["actual_pcm_preprocessing_validated"]=false;
                }
                var enqueueMs=timer.Elapsed.TotalMilliseconds;report["execute_enqueue_ms"]=enqueueMs;
                var actual=encoded.DownloadToArray();report["execute_and_final_download_ms"]=timer.Elapsed.TotalMilliseconds;
                report["final_download_and_gpu_wait_ms"]=timer.Elapsed.TotalMilliseconds-enqueueMs;
                GpuExecutionProfiler.ReadProcessMemory(process,out var workingSetAfter,out var peakWorkingSet);
                report["host_working_set_after_download_bytes"]=workingSetAfter;report["host_peak_working_set_bytes"]=peakWorkingSet;
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
                if(layerObservation>0)
                {
                    encoded.Dispose();pipeline.Dispose();
                    ObserveSelectedLayerManifest(models.ToArray(),bindingsByOperation.ToArray(),outputs.ToArray(),kernels.ToArray(),input,actual,output,report,Save,layerObservation);
                    return;
                }
                if(observeBmm)
                {
                    report["scope"]="connected_first_attention_operand_observation_only";report["full_encoder_validated"]=false;
                    var referenceRoot=Path.GetFullPath(Path.Combine(Application.dataPath,"../../numerical-acceptance-state/native-attention-operations-5a51cff-20261015"));
                    var referenceFile=Path.Combine(referenceRoot,"audio-attention-replay-report.json");
                    Check(referenceFile,"d16f82c1c343769577106a39203da32f00fb797468cc656ab1b12b685b8e691e");
                    var native=JObject.Parse(File.ReadAllText(referenceFile));
                    Assert.That(native.Value<bool>("success")&&native.Value<bool>("context_validated")&&native.Value<bool>("native_final_exact_match"),Is.True);
                    Assert.That(native.Value<string>("reference_source"),Is.EqualTo("4edbe17a8105233c5aaa1514faa2862b08a4f7e0"));
                    var products=operations.Where(p=>p.Value<string>("kind")=="batched_matmul").Take(3).ToArray();
                    Assert.That(products.Select(p=>p.Value<string>("node")),Is.EqualTo(new[]{"/attentions.0/MatMul","/attentions.0/MatMul_1","/attentions.0/MatMul_2"}));
                    var bindings=new List<(string part,string boundary,JToken shape,bool collapsed)>();var names=new[]{"matrix_ac","matrix_bd","weighted_values"};
                    for(var product=0;product<3;product++)
                    {
                        var operation=products[product];var boundaries=operation["inputs"].ToObject<string[]>().Append(operation.Value<string>("output")).ToArray();
                        var shapes=new[]{operation["input_shape"],operation["right_shape"],operation["output_shape"]};var kinds=new[]{"left","right","output"};
                        for(var operand=0;operand<3;operand++)bindings.Add(("self_attn."+names[product]+"."+kinds[operand],boundaries[operand],shapes[operand],true));
                    }
                    if(nativeTanh)
                    {
                        var operation=operations.First(p=>p.Value<string>("kind")=="tanh");
                        bindings.Add(("self_attn.tanh.input",operation.Value<string>("input"),operation["input_shape"],false));
                        bindings.Add(("self_attn.tanh.output",operation.Value<string>("output"),operation["output_shape"],false));
                    }
                    if(nativeSoftmax)
                    {
                        var operation=operations.First(p=>p.Value<string>("kind")=="softmax");
                        bindings.Add(("self_attn.softmax.input",operation.Value<string>("input"),operation["input_shape"],false));
                        bindings.Add(("self_attn.softmax.output",operation.Value<string>("output"),operation["output_shape"],false));
                    }
                    var wanted=bindings.Select(b=>b.boundary).ToHashSet();
                    // Reuse the already recorded normal result and release its
                    // workers before creating diagnostic workers.
                    encoded.Dispose();pipeline.Dispose();
                    var snapshots=CaptureOperationBoundaries(models.ToArray(),bindingsByOperation.ToArray(),outputs.ToArray(),kernels.ToArray(),input,wanted,actual);
                    Assert.That(snapshots["__final"].Select(BitConverter.SingleToInt32Bits),Is.EqualTo(actual.Select(BitConverter.SingleToInt32Bits)));
                    report["normal_final_matches_observed_bits"]=true;
                    var captureRoot=Path.ChangeExtension(output,null)+"-captures";Assert.That(Directory.Exists(captureRoot),Is.False);Directory.CreateDirectory(captureRoot);
                    var stages=new JArray();report["stages"]=stages;
                    foreach(var binding in bindings)
                    {
                            var partName=binding.part;var part=native["parts"][partName];
                            var expectedPath=Path.Combine(referenceRoot,"audio-attention-replay",part.Value<string>("file"));Check(expectedPath,part.Value<string>("sha256"));
                            // ONNX retains leading batch axes; collapsing them preserves native BMM row order.
                            var actualShape=binding.shape.ToObject<int[]>();var nativeShape=part["shape"].ToObject<int[]>();
                            if(binding.collapsed)
                            {
                                Assert.That(actualShape.Skip(actualShape.Length-2),Is.EqualTo(nativeShape.Skip(1)));
                                Assert.That(actualShape.Take(actualShape.Length-2).Aggregate(1,(a,b)=>a*b),Is.EqualTo(nativeShape[0]));
                            }
                            else Assert.That(actualShape,Is.EqualTo(nativeShape));
                            var observed=snapshots[binding.boundary];var golden=Floats(expectedPath);Assert.That(observed.Length,Is.EqualTo(golden.Length));
                            int bits=0,bad=0;double maxRatio=0;
                            for(var i=0;i<observed.Length;i++)
                            {
                                Assert.That(float.IsNaN(observed[i])||float.IsInfinity(observed[i]),Is.False);
                                if(BitConverter.SingleToInt32Bits(observed[i])!=BitConverter.SingleToInt32Bits(golden[i]))bits++;
                                var delta=Math.Abs((double)observed[i]-golden[i]);var allowed=1e-5+1e-4*Math.Abs((double)golden[i]);
                                if(delta>allowed)bad++;maxRatio=Math.Max(maxRatio,delta/allowed);
                            }
                            var capture=Path.Combine(captureRoot,partName+"-fp32.bin");var bytes=new byte[observed.Length*4];Buffer.BlockCopy(observed,0,bytes,0,bytes.Length);File.WriteAllBytes(capture,bytes);
                            using var stream=File.OpenRead(capture);var hash=PlayerFileHash.Compute(stream,out _);
                            stages.Add(new JObject{["part"]=partName,["boundary"]=binding.boundary,["actual_shape"]=binding.shape.DeepClone(),["native_shape"]=part["shape"].DeepClone(),["count"]=observed.Length,["bit_differences"]=bits,["mismatched_elements"]=bad,["maximum_tolerance_ratio"]=maxRatio,["capture"]=capture,["sha256"]=hash,["native_sha256"]=part["sha256"]});Save();
                    }
                    Assert.That(stages.Count,Is.EqualTo(nativeSoftmax?13:nativeTanh?11:9));report["success"]=true;report["observation_completed"]=true;return;
                }
                if(early)
                {
                    var nativeRoot=Path.GetFullPath(Path.Combine(Application.dataPath,"../../k/artifacts/audio-native-feed-forward-replay"));
                    var replayPath=Path.Combine(nativeRoot,"audio-layer-replay-report.json");
                    Check(replayPath,"99529dc0fa6c74825b23fca653d82091797ff1e5267d26c5cdae723fcf86f501");
                    var replay=JObject.Parse(File.ReadAllText(replayPath));
                    JObject projectionReplay=null;string projectionRoot=null;
                    JObject convolutionReplay=null;string convolutionRoot=null;
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
                    if(attention)bindings=bindings.Concat(new[]{
                        (name:"self_attn",boundary:"/attentions.0/post/linear/MatMul_output_0",part:"self_attn",minimum:-101.5f,maximum:100.5f),
                        (name:"attention_residual",boundary:"/lconv1d/pre_layer_norm/Cast_output_0",part:"norm_post_attn",minimum:float.NegativeInfinity,maximum:float.PositiveInfinity),
                        (name:"lconv1d",boundary:"/lconv1d/Add_output_0",part:"lconv1d",minimum:float.NegativeInfinity,maximum:float.PositiveInfinity)}).ToArray();
                    if(attention)
                    {
                        var referenceRoot=Path.GetFullPath(Path.Combine(Application.dataPath,"../../numerical-acceptance-state/native-attention-reference-c030222-20261015"));
                        var projectionPath=Path.Combine(referenceRoot,"audio-attention-replay-report.json");
                        Check(projectionPath,"d7c46da5391561c2879c56bc7d1796b95aba38e7bf61d8875a5c685b4623d5bb");
                        projectionReplay=JObject.Parse(File.ReadAllText(projectionPath));projectionRoot=Path.Combine(referenceRoot,"audio-attention-replay");
                        Assert.That(projectionReplay.Value<bool>("success")&&projectionReplay.Value<bool>("native_final_exact_match")&&projectionReplay.Value<bool>("context_validated"),Is.True);
                        Assert.That(projectionReplay.Value<string>("reference_source"),Is.EqualTo("4edbe17a8105233c5aaa1514faa2862b08a4f7e0"));
                        Assert.That(projectionReplay.Value<string>("diagnostic_source"),Is.EqualTo("c03022264c4bab5035f1e87721a6c71fe6a61d01"));
                        foreach(var original in (JObject)replay["parts"])Assert.That(projectionReplay["parts"][original.Key].Value<string>("sha256"),Is.EqualTo(original.Value.Value<string>("sha256")));
                        Assert.That(projectionReplay["parts"]["self_attn.post"].Value<string>("sha256"),Is.EqualTo(replay["parts"]["self_attn"].Value<string>("sha256")));
                        bindings=bindings.Concat(new[]{
                            (name:"q_proj",boundary:"/attentions.0/q_proj/linear/MatMul_output_0",part:"self_attn.q_proj",minimum:-34.5f,maximum:34.25f),
                            (name:"k_proj",boundary:"/attentions.0/k_proj/linear/MatMul_output_0",part:"self_attn.k_proj",minimum:-34.5f,maximum:34.25f),
                            (name:"v_proj",boundary:"/attentions.0/v_proj/linear/MatMul_output_0",part:"self_attn.v_proj",minimum:-34.5f,maximum:34.25f),
                            (name:"relative_k_proj",boundary:"/attentions.0/relative_k_proj/MatMul_output_0",part:"self_attn.relative_k_proj",minimum:float.NegativeInfinity,maximum:float.PositiveInfinity)}).ToArray();
                    }
                    if(convolutionDetail)
                    {
                        var referenceRoot=Path.GetFullPath(Path.Combine(Application.dataPath,"../../numerical-acceptance-state/native-convolution-reference-5d5c5eb-20261015/audio-layer-replay-5d5c5eb8a58b4d84f65caea9f7cddf5b5ae0a4ab"));
                        var referenceFile=Path.Combine(referenceRoot,"audio-attention-replay-report.json");Check(referenceFile,"906d186ec19f25e7397ddc3e826d0fa0f6ab07ecd988c1ec314cfeaab95d7c78");
                        convolutionReplay=JObject.Parse(File.ReadAllText(referenceFile));convolutionRoot=Path.Combine(referenceRoot,"audio-attention-replay");
                        Assert.That(convolutionReplay.Value<bool>("success")&&convolutionReplay.Value<bool>("context_validated")&&convolutionReplay.Value<bool>("native_final_exact_match"),Is.True);
                        Assert.That(convolutionReplay.Value<string>("reference_source"),Is.EqualTo("4edbe17a8105233c5aaa1514faa2862b08a4f7e0"));
                        Assert.That(convolutionReplay.Value<string>("diagnostic_source"),Is.EqualTo("5d5c5eb8a58b4d84f65caea9f7cddf5b5ae0a4ab"));
                        var oldFile=Path.GetFullPath(Path.Combine(Application.dataPath,"../../numerical-acceptance-state/native-attention-operations-5a51cff-20261015/audio-attention-replay-report.json"));
                        Check(oldFile,"d16f82c1c343769577106a39203da32f00fb797468cc656ab1b12b685b8e691e");
                        foreach(var part in (JObject)JObject.Parse(File.ReadAllText(oldFile))["parts"])Assert.That(convolutionReplay["parts"][part.Key].Value<string>("sha256"),Is.EqualTo(part.Value.Value<string>("sha256")));
                        bindings=bindings.Concat(new[]{
                            (name:"conv_pre_norm",boundary:"/lconv1d/linear_start/Clip_output_0",part:"lconv1d.pre_layer_norm.output",minimum:-32.5f,maximum:32.25f),
                            (name:"conv_linear_start",boundary:"/lconv1d/linear_start/linear/MatMul_output_0",part:"lconv1d.linear_start.output",minimum:-29.625f,maximum:29.375f),
                            (name:"conv_norm",boundary:"/lconv1d/conv_norm/Mul_1_output_0",part:"lconv1d.conv_norm.output",minimum:float.NegativeInfinity,maximum:float.PositiveInfinity),
                            (name:"conv_activation",boundary:"/lconv1d/act_fn/NativeSiLU_output_0",part:"lconv1d.act_fn.output",minimum:float.NegativeInfinity,maximum:float.PositiveInfinity),
                            (name:"conv_linear_end_input",boundary:"/lconv1d/linear_end/Clip_output_0",part:"lconv1d.act_fn.output",minimum:-5.8125f,maximum:5.78125f),
                            (name:"conv_linear_end",boundary:"/lconv1d/linear_end/linear/MatMul_output_0",part:"lconv1d.linear_end.output",minimum:-6.25f,maximum:6.1875f)}).ToArray();
                        if(nativeGlu)
                        {
                            var operation=operations.First(p=>p.Value<string>("kind")=="glu");
                            bindings=bindings.Append((name:"conv_glu_output",boundary:operation.Value<string>("output"),part:"lconv1d.glu.output",minimum:float.NegativeInfinity,maximum:float.PositiveInfinity)).ToArray();
                        }
                    }
                    report["phase"]="observe_early";Save();
                    encoded.Dispose();pipeline.Dispose();
                    var snapshots=CaptureOperationBoundaries(models.ToArray(),bindingsByOperation.ToArray(),outputs.ToArray(),kernels.ToArray(),input,bindings.Select(b=>b.boundary).ToHashSet(),actual);
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
                            var projectedPart=binding.part.StartsWith("self_attn.",StringComparison.Ordinal);
                            var convolutionPart=binding.part.StartsWith("lconv1d.",StringComparison.Ordinal);
                            var part=(convolutionPart?convolutionReplay:projectedPart?projectionReplay:replay)["parts"][binding.part];
                            path=convolutionPart?Path.Combine(convolutionRoot,part.Value<string>("file")):projectedPart?Path.Combine(projectionRoot,part.Value<string>("file")):Path.Combine(nativeRoot,"audio-layer-replay",part.Value<string>("file"));expectedSha=part.Value<string>("sha256");
                        }
                        Check(path,expectedSha);var native=Floats(path);var values=snapshots[binding.boundary];Assert.That(values.Length,Is.EqualTo(native.Length));
                        if(binding.name=="attention_residual")
                        {
                            var part=replay["parts"]["feed_forward1"];var residualPath=Path.Combine(nativeRoot,"audio-layer-replay",part.Value<string>("file"));
                            Check(residualPath,part.Value<string>("sha256"));var residual=Floats(residualPath);Assert.That(residual.Length,Is.EqualTo(native.Length));
                            for(var index=0;index<native.Length;index++)native[index]=native[index]+residual[index];
                        }
                        Assert.That(values.Any(v=>float.IsNaN(v)||float.IsInfinity(v)),Is.False);
                        int bits=0,bad=0;double maxRatio=0;
                        var clipObserved=binding.name=="self_attn"||binding.name=="q_proj"||binding.name=="k_proj"||binding.name=="v_proj"||binding.name=="conv_linear_start"||binding.name=="conv_linear_end";
                        for(var index=0;index<values.Length;index++)
                        {
                            // Transform only the immutable expected values to the original
                            // quantized-linear input Clip boundary. Product computation stays GPU.
                            var target=Math.Max(binding.minimum,Math.Min(binding.maximum,native[index]));
                            var observed=clipObserved?Math.Max(binding.minimum,Math.Min(binding.maximum,values[index])):values[index];
                            if(BitConverter.SingleToInt32Bits(observed)!=BitConverter.SingleToInt32Bits(target))bits++;
                            var difference=Math.Abs((double)observed-target);var allowed=1e-5+1e-4*Math.Abs((double)target);
                            if(difference>allowed)bad++;maxRatio=Math.Max(maxRatio,difference/allowed);
                        }
                        var bytes=new byte[values.Length*4];Buffer.BlockCopy(values,0,bytes,0,bytes.Length);var capture=Path.Combine(captureRoot,binding.name+"-fp32.bin");File.WriteAllBytes(capture,bytes);
                        using var stream=File.OpenRead(capture);var captureSha=PlayerFileHash.Compute(stream,out _);
                        stages.Add(new JObject{["stage"]=binding.name,["boundary"]=binding.boundary,["original_native_sha256"]=expectedSha,
                            ["expected_transform"]=binding.name=="attention_residual"?"original_fp32_residual_sum":!clipObserved&&(!float.IsInfinity(binding.minimum)||!float.IsInfinity(binding.maximum))?"original_fp32_input_clip":"identity",
                            ["actual_transform"]=clipObserved?"original_fp32_output_clip_for_diagnostic_comparison_only":"identity",
                            ["actual_file"]=capture,["actual_sha256"]=captureSha,["bit_differences"]=bits,["mismatched_elements"]=bad,["maximum_tolerance_ratio"]=maxRatio});Save();
                    }
                    Assert.That(stages.Count,Is.EqualTo(convolutionDetail?(nativeGlu?19:18):attention?12:5));report["observation_completed"]=true;report["success"]=true;report["phase"]="completed_observation";return;
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
                    if(signal=="impulse")anchors.Insert(0,new JObject{["stage"]="subsample",["onnx_node"]="/subsample_conv_projection/input_proj_linear/MatMul"});
                    var wanted=anchors.Select(a=>a.Value<string>("onnx_node")+"_output_0").ToHashSet();
                    report["phase"]="observe";Save();
                    encoded.Dispose();pipeline.Dispose();
                    var snapshots=CaptureOperationBoundaries(models.ToArray(),bindingsByOperation.ToArray(),outputs.ToArray(),kernels.ToArray(),input,wanted,actual);
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
                        Assert.That(values.Any(value=>float.IsNaN(value)||float.IsInfinity(value)),Is.False);
                        for(var i=0;i<values.Length;i++)
                        {
                            if(BitConverter.SingleToInt32Bits(values[i])!=BitConverter.SingleToInt32Bits(native[i]))bits++;
                            var difference=Math.Abs((double)values[i]-native[i]);var allowed=1e-5+1e-4*Math.Abs((double)native[i]);
                            if(difference>allowed)bad++;maxRatio=Math.Max(maxRatio,difference/allowed);
                        }
                        stages.Add(new JObject{["stage"]=name,["boundary"]=boundary,["bit_differences"]=bits,["mismatched_elements"]=bad,["maximum_tolerance_ratio"]=maxRatio});Save();
                    }
                    Assert.That(stages.Count,Is.EqualTo(signal=="impulse"?13:12));report["observation_completed"]=true;
                    report["full_encoder_validated"]=false;report["success"]=true;report["phase"]="completed_observation";return;
                }
                Assert.That(mismatch,Is.Zero,"Keep the original full FP32 audio encoder gate.");report["success"]=true;
            }
            catch(Exception error){report["error"]=error.ToString();throw;}
            finally { foreach(var kernel in kernels)kernel.Dispose();Save(); }
        }
        // Diagnostic-only downloads; the product GPU pipeline is unchanged.
        static Dictionary<string,float[]> ObserveBoundaries(Model[] models,string[] inputs,string[] outputs,IGpuTensorOperation[] kernels,Tensor<float> input,HashSet<string> wanted)
            =>ObserveOperationBoundaries(models,inputs.Select(name=>new[]{name}).ToArray(),outputs,kernels,input,wanted);

        static Dictionary<string,float[]> ObserveOperationBoundaries(Model[] models,string[][] inputs,string[] outputs,IGpuTensorOperation[] kernels,Tensor<float> input,HashSet<string> wanted)
        {
            float[] baseline;
            using(var normal=new GpuProjectionPipeline(models,inputs,outputs,kernels))
            using(var final=normal.Execute(input))baseline=final.DownloadToArray();
            return CaptureOperationBoundaries(models,inputs,outputs,kernels,input,wanted,baseline);
        }

        static Dictionary<string,float[]> CaptureOperationBoundaries(Model[] models,string[][] inputs,string[] outputs,IGpuTensorOperation[] kernels,Tensor<float> input,HashSet<string> wanted,float[] baseline)
        {
            if(!wanted.IsSubsetOf(models.SelectMany(m=>m.outputs.Select(o=>o.name)).Concat(outputs)))throw new ArgumentException("Observed boundary is missing.");
            var values=new Dictionary<string,Tensor<float>>{{models[0].inputs[0].name,input}};
            var owned=new HashSet<Tensor<float>>();var snapshots=new Dictionary<string,float[]>();
            try
            {
                for(var stage=0;stage<models.Length;stage++)
                {
                    var model=models[stage];using var worker=new Worker(model,BackendType.GPUCompute);
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
                    if(stage<kernels.Length)
                    {
                        var operation=kernels[stage];var value=operation is IGpuBinaryTensorOperation binary
                            ?binary.Execute(values[inputs[stage][0]],values[inputs[stage][1]])
                            :operation.Execute(values[inputs[stage][0]]);
                        owned.Add(value);Assert.That(value.backendType,Is.EqualTo(BackendType.GPUCompute));values[outputs[stage]]=value;
                        if(wanted.Contains(outputs[stage]))snapshots[outputs[stage]]=value.DownloadToArray();
                    }
                    // This observer schedules each stage once. Retain only
                    // values consumed by future stages, as in the product path.
                    var future=models.Skip(stage+1).SelectMany(m=>m.inputs.Select(i=>i.name))
                        .Concat(inputs.Skip(stage+1).SelectMany(names=>names))
                        .Append(models.Last().outputs.Single().name).ToHashSet();
                    foreach(var name in values.Keys.Where(name=>!future.Contains(name)).ToArray())values.Remove(name);
                    var retained=values.Values.ToHashSet();
                    foreach(var value in owned.Where(value=>!retained.Contains(value)).ToArray()){value.Dispose();owned.Remove(value);}
                }
                snapshots["__final"]=values[models.Last().outputs.Single().name].DownloadToArray();
                Assert.That(snapshots["__final"].Select(BitConverter.SingleToInt32Bits),Is.EqualTo(baseline.Select(BitConverter.SingleToInt32Bits)),"Diagnostic downloads changed normal final bits.");
                Assert.That(snapshots.Keys.Where(k=>k!="__final"),Is.EquivalentTo(wanted));return snapshots;
            }
            finally {foreach(var value in owned)value.Dispose();}
        }
        static void ObserveSelectedLayerManifest(Model[] models,string[][] inputs,string[] outputs,IGpuTensorOperation[] kernels,Tensor<float> input,float[] baseline,string output,JObject report,Action save,int layerIndex)
        {
            Assert.That(layerIndex,Is.EqualTo(1).Or.EqualTo(3));
            var manifestSha=layerIndex==1?"c72769f97ab1c82b684aed947579b8c8b83759be377b5037c268477d0e7a28f5":"e521184be954eacc264d0d12c2aaf5dc4ee7dfab4a0339fb79ed829c3c3f8018";
            var referenceSha=layerIndex==1?"08a1d5d60402d9cda0624c1ab14edc81db42b8e70d3e50f2554f1fdc0d54cafb":"13d560530a892a363f0ecdb1beb6d98f11697501e5f8249613f9e80fca63c866";
            var diagnosticSource=layerIndex==1?"9d719229e739a788cfc685946033a4414daa0e16":"56c9f5ec3919ca0ce0774e411bc6ad8b4cf95262";
            var entrySha=layerIndex==1?"3cc353963c7eafd2a209526c5c19daedd66c9c0da7cdf43dc6a4f4ccee158cf2":"5452f03c4aa0f6b518b3c2b8deb0d9c679508b6ec19a8e7849c7948f21ec30f2";
            void Check(string path,string sha){using var stream=File.OpenRead(path);Assert.That(PlayerFileHash.Compute(stream,out _),Is.EqualTo(sha));}
            var root=Path.GetFullPath(Path.Combine(Application.dataPath,"../../numerical-acceptance-state"));var manifestFile=Path.Combine(root,"layer"+layerIndex+"-native-attention-boundary-manifest-20261015.json");
            Check(manifestFile,manifestSha);var manifest=JObject.Parse(File.ReadAllText(manifestFile));
            Assert.That(manifest.Value<string>("source_graph_sha256"),Is.EqualTo("c050c2efa87d83aea45587b25f7b8d1d24dc471f33cc732dcf17d89588c70c13"));Assert.That(manifest.Value<int>("layer_index"),Is.EqualTo(layerIndex));
            var referenceFile=Path.Combine(root,manifest.Value<string>("reference_file"));Check(referenceFile,referenceSha);
            var reference=JObject.Parse(File.ReadAllText(referenceFile));Assert.That(reference.Value<bool>("success")&&reference.Value<bool>("context_validated")&&reference.Value<bool>("native_final_exact_match"),Is.True);
            Assert.That(reference.Value<int>("layer_index"),Is.EqualTo(layerIndex));Assert.That(reference.Value<string>("input_stage"),Is.EqualTo("layer_"+(layerIndex-1)));Assert.That(reference.Value<string>("output_stage"),Is.EqualTo("layer_"+layerIndex));
            Assert.That(reference.Value<string>("reference_source"),Is.EqualTo("4edbe17a8105233c5aaa1514faa2862b08a4f7e0"));Assert.That(reference.Value<string>("diagnostic_source"),Is.EqualTo(diagnosticSource));
            var bindings=(JArray)manifest["bindings"];Assert.That(bindings.Count,Is.EqualTo(35));var wanted=bindings.Select(b=>b.Value<string>("boundary")).ToHashSet();
            var entry="/norm_out"+(layerIndex==1?"":"_"+(layerIndex-1))+"/Mul_1_output_0";wanted.Add(entry);report["scope"]="one_original_layer"+layerIndex+"_module_observation_only";report["phase"]="observe_layer"+layerIndex;report["full_encoder_validated"]=false;save();
            var snapshots=CaptureOperationBoundaries(models,inputs,outputs,kernels,input,wanted,baseline);
            var entryFile=Path.GetFullPath(Path.Combine(root,"../k/artifacts/audio-current-stage-context/silence-stage-"+layerIndex.ToString("D2")+"-fp32.bin"));Check(entryFile,entrySha);
            Assert.That(snapshots[entry].Select(BitConverter.SingleToInt32Bits),Is.EqualTo(Floats(entryFile).Select(BitConverter.SingleToInt32Bits)),"Selected layer must receive its original native input.");
            report["original_layer_input_matches_bits"]=true;report["normal_final_matches_observed_bits"]=true;report["reference_report_sha256"]=referenceSha;
            report["not_observed"]=manifest["not_observed"].DeepClone();var stages=new JArray();report["stages"]=stages;
            var captureRoot=Path.ChangeExtension(output,null)+"-captures";Assert.That(Directory.Exists(captureRoot),Is.False);Directory.CreateDirectory(captureRoot);
            foreach(var binding in bindings)
            {
                var label=binding.Value<string>("stage");var boundary=binding.Value<string>("boundary");var expectedFile=Path.Combine(root,binding.Value<string>("expected_file"));Check(expectedFile,binding.Value<string>("expected_sha256"));
                var actual=snapshots[boundary];var expected=Floats(expectedFile);Assert.That(actual.Length,Is.EqualTo(expected.Length));
                var actualShape=binding["actual_shape"].ToObject<int[]>();var expectedShape=binding["expected_shape"].ToObject<int[]>();
                if(binding.Value<bool>("collapsed"))
                {
                    Assert.That(actualShape.Skip(actualShape.Length-2),Is.EqualTo(expectedShape.Skip(expectedShape.Length-2)));
                    Assert.That(actualShape.Take(actualShape.Length-2).Aggregate(1,(a,b)=>a*b),Is.EqualTo(expectedShape.Take(expectedShape.Length-2).Aggregate(1,(a,b)=>a*b)));
                }
                else Assert.That(actualShape,Is.EqualTo(expectedShape));
                var clip=binding["clip"] is JArray limits?limits.ToObject<float[]>():null;bool clipActual=binding.Value<bool>("clip_actual");int bits=0,bad=0;double maxRatio=0;
                Assert.That(actual.Any(value=>float.IsNaN(value)||float.IsInfinity(value)),Is.False);
                for(var i=0;i<actual.Length;i++)
                {
                    var target=clip==null?expected[i]:Math.Max(clip[0],Math.Min(clip[1],expected[i]));
                    var value=clipActual?Math.Max(clip[0],Math.Min(clip[1],actual[i])):actual[i];if(BitConverter.SingleToInt32Bits(value)!=BitConverter.SingleToInt32Bits(target))bits++;
                    var delta=Math.Abs((double)value-target);var allowed=1e-5+1e-4*Math.Abs((double)target);if(delta>allowed)bad++;maxRatio=Math.Max(maxRatio,delta/allowed);
                }
                var capture=Path.Combine(captureRoot,label+"-fp32.bin");var bytes=new byte[actual.Length*4];Buffer.BlockCopy(actual,0,bytes,0,bytes.Length);File.WriteAllBytes(capture,bytes);using var stream=File.OpenRead(capture);var hash=PlayerFileHash.Compute(stream,out _);
                stages.Add(new JObject{["stage"]=label,["part"]=binding["part"].DeepClone(),["boundary"]=boundary,["bit_differences"]=bits,["mismatched_elements"]=bad,["maximum_tolerance_ratio"]=maxRatio,["actual_file"]=capture,["actual_sha256"]=hash,["expected_sha256"]=binding["expected_sha256"].DeepClone(),["expected_transform"]=clip==null?"identity":"original_fp32_clip",["actual_transform"]=clipActual?"original_fp32_output_clip_for_diagnostic_comparison_only":"identity"});save();
            }
            Assert.That(stages.Count,Is.EqualTo(35));report["observation_completed"]=true;report["success"]=true;report["phase"]="completed_observation";save();
        }
        static float[] Floats(string path){var bytes=File.ReadAllBytes(path);var values=new float[bytes.Length/4];Buffer.BlockCopy(bytes,0,values,0,bytes.Length);return values;}
    }
}
