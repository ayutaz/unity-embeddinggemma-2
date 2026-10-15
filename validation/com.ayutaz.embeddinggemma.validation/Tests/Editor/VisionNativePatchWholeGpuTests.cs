using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Threading.Tasks;
using EmbeddingGemma.Tests;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.TestTools;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class VisionNativePatchWholeGpuTests
    {
        [TestCase("square-0")] [TestCase("square-1")]
        [TestCase("landscape-0")] [TestCase("landscape-1")]
        [TestCase("portrait-0")] [TestCase("portrait-1")]
        public void NativePatch384OriginalVisionEncoderMustMatchGoldenOnGpu(string caseId)
            => Run(caseId,false).GetAwaiter().GetResult();

        [TestCase("square-0")] [TestCase("square-1")]
        [TestCase("landscape-0")] [TestCase("landscape-1")]
        [TestCase("portrait-0")] [TestCase("portrait-1")]
        public void InlineAllVisionLayerRmsOriginalEncoderMustMatchGoldenOnGpu(string caseId)
            => Run(caseId,true).GetAwaiter().GetResult();

        [TestCase("square-0")] [TestCase("square-1")]
        [TestCase("landscape-0")] [TestCase("landscape-1")]
        [TestCase("portrait-0")] [TestCase("portrait-1")]
        public void NativeGeluAndInlineVisionRmsEncoderMustMatchGoldenOnGpu(string caseId)
            => Run(caseId,true,true).GetAwaiter().GetResult();

        [TestCase("square-0")] [TestCase("square-1")]
        [TestCase("landscape-0")] [TestCase("landscape-1")]
        [TestCase("portrait-0")] [TestCase("portrait-1")]
        public void CacheReleasedNativeGeluVisionEncoderMustMatchGoldenOnGpu(string caseId)
            => Run(caseId,true,true,true).GetAwaiter().GetResult();

        [Test]
        public void CacheReleasedIncludingGeluPreservesSquareZeroCandidateAndOriginalReferenceGate()
            => Run("square-0",true,true,true,false,true).GetAwaiter().GetResult();

        [Test]
        public void AuditSquareZeroPartitionBoundariesWithoutExecutingModel()
            => Run("square-0",true,true,true,false,true,true).GetAwaiter().GetResult();

        [UnityTest]
        public IEnumerator StageFencesPreserveSquareZeroCandidateAndOriginalReferenceGate()
        {
            var pending=Run("square-0",true,true,true,true);
            while(!pending.IsCompleted)yield return null;
            pending.GetAwaiter().GetResult();
        }

        [UnityTest]
        public IEnumerator StageFencesAndReleasedGeluCachesPreserveSquareZeroCandidateAndOriginalReferenceGate()
        {
            var pending=Run("square-0",true,true,true,true,true);
            while(!pending.IsCompleted)yield return null;
            pending.GetAwaiter().GetResult();
        }

        [UnityTest]
        public IEnumerator StageFencesWithEditorFramePumpPreserveSquareZeroCandidateAndOriginalReferenceGate()
        {
            var pending=Run("square-0",true,true,true,true,true,false,true);
            while(!pending.IsCompleted){UnityEditor.EditorApplication.QueuePlayerLoopUpdate();yield return null;}
            pending.GetAwaiter().GetResult();
        }

        [UnityTest]
        public IEnumerator QualifiedNativeOperatorsSquareZeroMustMatchOriginalOnGpu()
        {
            var pending=Run("square-0",true,true,true,true,true,false,true,true);
            while(!pending.IsCompleted){UnityEditor.EditorApplication.QueuePlayerLoopUpdate();yield return null;}
            pending.GetAwaiter().GetResult();
        }

        [UnityTest]
        public IEnumerator OriginalSavedContextBlock192SquareZeroMustMatchOriginalOnGpu()
        {
            var pending=Run("square-0",true,true,true,true,true,false,true,true,192);
            while(!pending.IsCompleted){UnityEditor.EditorApplication.QueuePlayerLoopUpdate();yield return null;}
            pending.GetAwaiter().GetResult();
        }

        [UnityTest]
        public IEnumerator OriginalSavedContextAllRmsBlock192SquareZeroMustMatchOriginalOnGpu()
        {
            var pending=Run("square-0",true,true,true,true,true,false,true,true,192,true);
            while(!pending.IsCompleted){UnityEditor.EditorApplication.QueuePlayerLoopUpdate();yield return null;}
            pending.GetAwaiter().GetResult();
        }

        [UnityTest]
        public IEnumerator OriginalContextSoftmaxSixteenSquareZeroMustMatchOriginalOnGpu()
        {
            var pending=Run("square-0",true,true,true,true,true,false,true,true,192,true,16);
            while(!pending.IsCompleted){UnityEditor.EditorApplication.QueuePlayerLoopUpdate();yield return null;}
            pending.GetAwaiter().GetResult();
        }

        [UnityTest]
        public IEnumerator OriginalContextFusedGeluSoftmaxSixteenSquareZeroMustMatchOriginalOnGpu()
        {
            var pending=Run("square-0",true,true,true,true,true,false,true,true,192,true,16,true);
            while(!pending.IsCompleted){UnityEditor.EditorApplication.QueuePlayerLoopUpdate();yield return null;}
            pending.GetAwaiter().GetResult();
        }

        [UnityTest]
        public IEnumerator OriginalContextPreciseSleefGeluSquareZeroMustMatchOriginalOnGpu()
            =>PreciseSleefCase("square-0");
        [UnityTest]
        public IEnumerator OriginalContextPreciseSleefGeluSquareOneMustMatchOriginalOnGpu()
            =>PreciseSleefCase("square-1");
        [UnityTest]
        public IEnumerator OriginalContextPreciseSleefGeluLandscapeZeroMustMatchOriginalOnGpu()
            =>PreciseSleefCase("landscape-0");
        [UnityTest]
        public IEnumerator OriginalContextPreciseSleefGeluLandscapeOneMustMatchOriginalOnGpu()
            =>PreciseSleefCase("landscape-1");
        [UnityTest]
        public IEnumerator OriginalContextPreciseSleefGeluPortraitZeroMustMatchOriginalOnGpu()
            =>PreciseSleefCase("portrait-0");
        [UnityTest]
        public IEnumerator OriginalContextPreciseSleefGeluPortraitOneMustMatchOriginalOnGpu()
            =>PreciseSleefCase("portrait-1");

        static IEnumerator PreciseSleefCase(string caseId)
        {
            var pending=Run(caseId,true,true,true,true,true,false,true,true,192,true,16,true,true);
            while(!pending.IsCompleted){UnityEditor.EditorApplication.QueuePlayerLoopUpdate();yield return null;}
            pending.GetAwaiter().GetResult();
        }

        [UnityTest]
        public IEnumerator OriginalContextNativeRotarySquareZeroMustMatchOriginalOnGpu()
        {
            var pending=Run("square-0",true,true,true,true,true,false,true,true,192,true,16,true,true,true);
            while(!pending.IsCompleted){UnityEditor.EditorApplication.QueuePlayerLoopUpdate();yield return null;}
            pending.GetAwaiter().GetResult();
        }

        static async Task Run(string caseId,bool inlineAllRms,bool nativeGelu=false,bool releaseUnusedGpuCache=false,bool staged=false,bool releaseGeluCaches=false,bool planOnly=false,bool editorFramePump=false,bool nativeOperators=false,int nativeStaticBlockSize=384,bool inlineTailRms=false,int softmaxVectorWidth=8,bool fuseGeluInner=false,bool sleefGelu=false,bool nativeRotary=false)
        {
            Assert.That(SystemInfo.supportsComputeShaders,Is.True,"Actual GPU required; never skip.");
            var source=Environment.GetEnvironmentVariable("EG_ACCEPTANCE_SOURCE");
            var root=Path.GetFullPath(Path.Combine(Application.dataPath,"../../k/artifacts/vision-encoder"));
            var label=releaseGeluCaches&&staged?"vision-staged-nested-gelu-cache-release-whole-gpu-":releaseGeluCaches?"vision-nested-gelu-cache-release-whole-gpu-":staged?"vision-staged-cache-release-native-gelu-rms-whole-gpu-":releaseUnusedGpuCache?"vision-cache-release-native-gelu-rms-whole-gpu-":nativeGelu?"vision-native-gelu-inline-rms-whole-gpu-":inlineAllRms?"vision-inline-all-layer-rms-whole-gpu-":"vision-native-patch-whole-gpu-";
            if(planOnly)label="vision-partition-boundary-audit-";
            if(nativeOperators)label="vision-qualified-native-operators-whole-gpu-";
            if(nativeOperators&&nativeStaticBlockSize!=384)label="vision-original-context-block"+nativeStaticBlockSize+"-whole-gpu-";
            if(inlineTailRms)label="vision-original-context-all-rms-block"+nativeStaticBlockSize+"-whole-gpu-";
            if(softmaxVectorWidth==16)label="vision-original-context-softmax16-all-rms-block"+nativeStaticBlockSize+"-whole-gpu-";
            if(fuseGeluInner)label="vision-original-context-fused-gelu-softmax"+softmaxVectorWidth+"-all-rms-block"+nativeStaticBlockSize+"-whole-gpu-";
            if(sleefGelu)label="vision-original-context-precise-sleef-gelu-softmax"+softmaxVectorWidth+"-all-rms-block"+nativeStaticBlockSize+"-whole-gpu-";
            if(nativeRotary)label="vision-original-native-rotary-whole-gpu-";
            var output=Path.GetFullPath(Path.Combine(Application.dataPath,"../artifacts/"+label+source+"-"+caseId+".json"));
            Assert.That(File.Exists(output),Is.False);
            var report=new JObject{["success"]=false,["scope"]=releaseUnusedGpuCache?"one_original_vision_encoder_unused_cache_release_native_gelu_rms_candidate":nativeGelu?"one_original_vision_encoder_gpu_native_gelu_all_layer_rms_and_patch384_candidate":inlineAllRms?"one_original_vision_encoder_gpu_all_sixteen_layer_rms_and_native_patch384_candidate":"one_original_vision_encoder_gpu_native_patch384_candidate",["runtime_source"]=source,
                ["case_id"]=caseId,["phase"]="audit",["all_profiles_validated"]=false,["end_to_end_validated"]=false,
                ["scheduling_mode"]=staged?"gpu_fence_per_stage":"synchronous_queue",["intermediate_tensor_readback_added"]=false,["gelu_internal_cache_release"]=releaseGeluCaches,["editor_frame_pump"]=editorFramePump,["test_forced_gc_during_run"]=false};
            report["qualified_operator_reconnection"]=nativeOperators;
            if(nativeOperators)report["native_static_block_size"]=nativeStaticBlockSize;
            report["inline_tail_rms"]=inlineTailRms;
            report["softmax_vector_width"]=softmaxVectorWidth;
            report["fused_gelu_inner"]=fuseGeluInner;
            report["precise_sleef_gelu"]=sleefGelu;
            report["native_derived_rotary"]=nativeRotary;
            void Save(){Directory.CreateDirectory(Path.GetDirectoryName(output));File.WriteAllText(output,report.ToString());}
            void Check(string path,string sha){using var stream=File.OpenRead(path);Assert.That(PlayerFileHash.Compute(stream,out _),Is.EqualTo(sha));}
            var operations=new List<IGpuTensorOperation>();Save();
            try
            {
                var locked=JObject.Parse(File.ReadAllText(Path.Combine(Application.dataPath,"../Packages/packages-lock.json")));
                foreach(var package in new[]{"com.ayutaz.embeddinggemma","com.ayutaz.embeddinggemma.validation"})Assert.That(locked["dependencies"][package].Value<string>("hash"),Is.EqualTo(source));
                Check(Path.Combine(root,"vision-reference.json"),"3dfccd1a487ce04df3f8d640483ca045f7a7a56231bc1096ad0f845c923a29c6");
                Check(Path.Combine(root,"vision-validation.json"),"ab1aa775b553f33a5e2892abed2cdcef7768e35fbad6fa5af9ec3286a7ba90ba");
                var reference=JObject.Parse(File.ReadAllText(Path.Combine(root,"vision-reference.json")));
                var audit=JObject.Parse(File.ReadAllText(Path.Combine(root,"vision-validation.json")));
                var row=((JArray)reference["cases"]).Single(value=>value.Value<string>("id")==caseId);var profile=row.Value<string>("profile");
                var modelSha=profile=="square"?"4ecd0106463e833588c7584a58ea4973fdbab1eb1a56389c2f1597d3e7df34e7":profile=="landscape"?"b9bc2336a34167eb0fdfdd99e49c6486d66b20afc23d72da8df1e10da4a1e632":"41c0ade742b09cf291c0b14343b1cf8bffa6f44c9bb2e8d782bc7ba508371e6b";
                report["phase"]="audit_saved_model_file";Save();var phaseTimer=System.Diagnostics.Stopwatch.StartNew();
                Check(Path.Combine(root,profile+"-fp32.sentis"),modelSha);report["model_file_audit_ms"]=phaseTimer.Elapsed.TotalMilliseconds;report["profile"]=profile;report["saved_model_sha256"]=modelSha;
                report["phase"]="load_saved_model";Save();var timer=System.Diagnostics.Stopwatch.StartNew();phaseTimer.Restart();
                var original=TextModelFile.Load(Path.Combine(root,profile+"-fp32.sentis"));var constants=original.constants.ToDictionary(value=>value.index);
                report["model_load_ms"]=phaseTimer.Elapsed.TotalMilliseconds;report["phase"]="audit_coefficients_before";Save();phaseTimer.Restart();
                var working=original;var originalCoefficients=CoefficientsHash(original);report["coefficient_audit_before_ms"]=phaseTimer.Elapsed.TotalMilliseconds;
                report["phase"]="rewrite_layer_rms";Save();phaseTimer.Restart();
                if(inlineAllRms)
                {
                    Check(Path.Combine(Application.dataPath,"GpuVisionInlineRmsCandidate/inline-rms.onnx"),"15c8ec537c8d40386e61b675c05aaaa65b296667c0d38c39dc7d12fe28a25ccb");
                    var preparationPath=Path.Combine(Application.dataPath,"GpuVisionHeadRmsCandidates/head-rms-candidates.json");
                    Check(preparationPath,"bb1f2d3df3905551b34723badefd7f1dfdc23da08a9ab4c72545d5d1368ce229");
                    var preparation=JObject.Parse(File.ReadAllText(preparationPath));
                    Model Head(int index)
                    {
                        var entry=((JArray)preparation["rows"]).Single(value=>value.Value<int>("index")==index);
                        var asset="GpuVisionHeadRmsCandidates/"+entry["onnx"].Value<string>("file");Check(Path.Combine(Application.dataPath,asset),entry["onnx"].Value<string>("sha256"));
                        return ModelLoader.Load(UnityEditor.AssetDatabase.LoadAssetAtPath<ModelAsset>("Assets/"+asset));
                    }
                    var template=ModelLoader.Load(UnityEditor.AssetDatabase.LoadAssetAtPath<ModelAsset>("Assets/GpuVisionInlineRmsCandidate/inline-rms.onnx"));
                    working=QualifiedVisionRmsOperation.InlineLayerRms(original,template,Head(1),Head(3),16);
                    Assert.That(original.constants.All(value=>working.constants.Any(other=>ReferenceEquals(value,other))),Is.True);
                    report["inline_rms_count"]=112;report["layer_count"]=working.layers.Count;
                    report["original_square_count"]=original.layers.Count(layer=>layer.GetType().Name=="Square");
                    report["remaining_original_rms_count"]=original.layers.Count(layer=>layer.GetType().Name=="Square")-112;
                    report["all_original_coefficient_descriptors_retained"]=true;
                    if(inlineTailRms)
                    {
                        var featureCount=reference["profiles"][profile].Value<int>("feature_count");Assert.That(featureCount==252||featureCount==256,Is.True);
                        var tailFile=featureCount==252?"vision-tail-252-gpu-a4df94698cb1f7c7ea78081830a14111c38c3ce0.sentis":"vision-saved-original-cpu-tail-gpu-c3460fc20d6fc7f8599abcb626d210b0d23aef20-captures/candidate-tail-rms.sentis";
                        var tailSha=featureCount==252?"36b688b118f091490b75af9deab3e2e4873172dbab64e9f021cd99c3c7ba5599":"215fd08ac859acc3501b111104c048fb4450f11092418be4ea45a9fc3d985835";
                        var tailPath=Path.GetFullPath(Path.Combine(Application.dataPath,"../artifacts/"+tailFile));Check(tailPath,tailSha);
                        report["tail_template_feature_count"]=featureCount;report["tail_template_sha256"]=tailSha;
                        working=QualifiedVisionRmsOperation.InlineTailRms(working,TextModelFile.Load(tailPath));
                        Assert.That(original.constants.All(value=>working.constants.Any(other=>ReferenceEquals(value,other))),Is.True);
                        Assert.That(working.layers.Count(layer=>layer.GetType().Name=="Square"),Is.Zero);
                        report["inline_rms_count"]=113;report["remaining_original_rms_count"]=0;report["layer_count"]=working.layers.Count;
                    }
                }
                if(nativeRotary)
                {
                    Assert.That(profile,Is.EqualTo("square"),"This frozen position fixture qualifies the square profile only.");
                    var before=working;working=QualifiedVisionRotaryTables.RebindOriginalSquareForValidation(before);
                    var changed=before.layers.Zip(working.layers,(a,b)=>a.inputs.SequenceEqual(b.inputs)?0:1).Sum();
                    Assert.That(changed,Is.EqualTo(128));Assert.That(original.constants.All(c=>working.constants.Contains(c)),Is.True);
                    report["derived_rotary_rebound_tables"]=4;report["derived_rotary_changed_multiplication_operands"]=changed;
                    report["derived_rotary_scope"]="audited_data_independent_original_square_position_tables";
                }
                report["graph_rewrite_ms"]=phaseTimer.Elapsed.TotalMilliseconds;report["phase"]="prepare_gpu_pipeline";Save();phaseTimer.Restart();
                var projection=original.layers.First(layer=>layer.GetType().Name=="MatMul"&&constants.TryGetValue(layer.inputs[1],out var w)&&w.shape==new TensorShape(768,768));
                var raw=(ArraySegment<byte>)typeof(Constant).GetField("array",BindingFlags.NonPublic|BindingFlags.Instance).GetValue(constants[projection.inputs[1]]);
                using(var stream=new MemoryStream(raw.Array,raw.Offset,raw.Count,false))Assert.That(PlayerFileHash.Compute(stream,out _),Is.EqualTo("696c2e091ed4ad117c10faa6bf9707eb808649fb09c84fee457d2327408acc90"));
                var weights=new float[raw.Count/4];Buffer.BlockCopy(raw.Array,raw.Offset,weights,0,raw.Count);
                Model[] parts;string[][] inputs;string[] outputs;
                if(nativeOperators)
                {
                    Assert.That(inlineAllRms&&nativeGelu,Is.True);
                    var matrices=original.layers.Where(layer=>layer.GetType().Name=="MatMul"||layer.GetType().Name=="MatMul2D").ToArray();
                    var staticProducts=new HashSet<int>(matrices.Where(layer=>constants.ContainsKey(layer.inputs[1])).Select(layer=>layer.outputs[0]));
                    var binaryProducts=new HashSet<int>(matrices.Where(layer=>!constants.ContainsKey(layer.inputs[0])&&!constants.ContainsKey(layer.inputs[1])).Select(layer=>layer.outputs[0]));
                    var preserved=matrices.Where(layer=>!staticProducts.Contains(layer.outputs[0])&&!binaryProducts.Contains(layer.outputs[0])).ToArray();
                    report["preserved_original_matrix_operations"]=new JArray(preserved.Select(layer=>new JObject{["output_index"]=layer.outputs[0],
                        ["left_is_constant"]=constants.ContainsKey(layer.inputs[0]),["right_is_constant"]=constants.ContainsKey(layer.inputs[1]),
                        ["left_constant_shape"]=constants.TryGetValue(layer.inputs[0],out var coefficient)?JArray.FromObject(Enumerable.Range(0,coefficient.shape.rank).Select(axis=>coefficient.shape[axis]).ToArray()):null}));Save();
                    Assert.That(staticProducts.Count,Is.EqualTo(114));Assert.That(binaryProducts.Count,Is.EqualTo(32));
                    Assert.That(preserved.Length,Is.EqualTo(1));Assert.That(constants.ContainsKey(preserved[0].inputs[0]),Is.True);
                    var marked=SavedProjectionPartition.MarkGelu(working);working=marked.model;var geluOutputs=new HashSet<int>(marked.outputs);Assert.That(geluOutputs.Count,Is.EqualTo(16));
                    var softmaxes=working.layers.Where(layer=>layer.GetType().Name=="Softmax").ToArray();Assert.That(softmaxes.Length,Is.EqualTo(16));
                    var identity=typeof(Worker).Assembly.GetTypes().Single(type=>type.Name=="Identity"&&typeof(Layer).IsAssignableFrom(type));
                    var replacements=new Dictionary<Layer,Layer>();var softmaxOutputs=new HashSet<int>();
                    foreach(var layer in softmaxes)
                    {
                        Assert.That(layer.inputs.Length,Is.EqualTo(1));Assert.That(layer.outputs.Length,Is.EqualTo(1));
                        Assert.That((int)layer.GetType().GetField("axis").GetValue(layer),Is.EqualTo(-1));
                        var marker=(Layer)Activator.CreateInstance(identity);marker.inputs=(int[])layer.inputs.Clone();marker.outputs=(int[])layer.outputs.Clone();
                        replacements.Add(layer,marker);softmaxOutputs.Add(layer.outputs[0]);
                    }
                    working=new Model{ProducerName=working.ProducerName+"/qualified-native-operators",inputs=new List<Model.Input>(working.inputs),outputs=new List<Model.Output>(working.outputs),
                        constants=new List<Constant>(working.constants),layers=working.layers.Select(layer=>replacements.TryGetValue(layer,out var marker)?marker:layer).ToList()};
                    var wanted=new HashSet<int>(staticProducts.Concat(binaryProducts).Concat(geluOutputs).Concat(softmaxOutputs));
                    var selected=working.layers.Where(layer=>wanted.Contains(layer.outputs[0])).Select(layer=>layer.outputs[0]).ToArray();Assert.That(selected.Length,Is.EqualTo(178));
                    var plan=SavedProjectionPartition.Create(working,selected);Assert.That(plan.Models.Length,Is.EqualTo(179));
                    var operationRows=new JArray();report["qualified_operations"]=operationRows;
                    for(var i=0;i<selected.Length;i++)
                    {
                        string kind;
                        if(binaryProducts.Contains(selected[i]))
                        {
                            Assert.That(plan.Inputs[i].Length,Is.EqualTo(2));Assert.That(plan.Rows[i],Is.EqualTo(30240));
                            Assert.That((plan.Widths[i]==64&&plan.Columns[i]==2520)||(plan.Widths[i]==2520&&plan.Columns[i]==64),Is.True);
                            operations.Add(new GpuBatchedMatmul());kind="native_batched_matmul";
                        }
                        else if(geluOutputs.Contains(selected[i]))
                        {
                            Assert.That(plan.Rows[i],Is.EqualTo(2520));Assert.That(plan.Widths[i],Is.EqualTo(3072));
                            operations.Add(new GpuGelu(new TensorShape(1,2520,3072),releaseGeluCaches,fuseGeluInner,sleefGelu));kind=sleefGelu?"native_precise_sleef_gelu":"native_tanh_gelu";
                        }
                        else if(softmaxOutputs.Contains(selected[i]))
                        {
                            Assert.That(plan.Rows[i],Is.EqualTo(30240));Assert.That(plan.Widths[i],Is.EqualTo(2520));
                            operations.Add(new GpuSoftmax(softmaxVectorWidth));kind="native_softmax";
                        }
                        else
                        {
                            Assert.That(staticProducts.Contains(selected[i]),Is.True);Assert.That(plan.Weights[i],Is.Not.Null);Assert.That(plan.Inputs[i].Length,Is.EqualTo(1));
                            Assert.That(plan.Rows[i]==2520||plan.Rows[i]==reference["profiles"][profile].Value<int>("feature_count"),Is.True);
                            operations.Add(new GpuBlockedMatmul(plan.Widths[i],plan.Columns[i],plan.Weights[i],nativeStaticBlockSize));kind="native_static_block"+nativeStaticBlockSize;
                        }
                        operationRows.Add(new JObject{["output_index"]=selected[i],["kind"]=kind,["rows"]=plan.Rows[i],["width"]=plan.Widths[i],["columns"]=plan.Columns[i]});
                    }
                    parts=plan.Models;inputs=plan.Inputs;outputs=plan.Outputs;report["native_static_projection_count"]=114;report["native_batched_matmul_count"]=32;report["native_softmax_count"]=16;report["native_gelu_count"]=16;
                }
                else if(nativeGelu)
                {
                    var marked=SavedProjectionPartition.MarkGelu(working);working=marked.model;Assert.That(marked.outputs.Length,Is.EqualTo(16));
                    var wanted=new HashSet<int>(marked.outputs){projection.outputs[0]};
                    var selected=working.layers.Where(layer=>wanted.Contains(layer.outputs[0])).Select(layer=>layer.outputs[0]).ToArray();
                    var plan=SavedProjectionPartition.Create(working,selected);Assert.That(plan.Models.Length,Is.EqualTo(18));
                    for(var i=0;i<selected.Length;i++)
                    {
                        Assert.That(plan.Rows[i],Is.EqualTo(2520));
                        if(plan.Weights[i]==null)
                        {
                            Assert.That(plan.Widths[i],Is.EqualTo(3072));operations.Add(new GpuGelu(new TensorShape(1,2520,3072),releaseGeluCaches));
                        }
                        else
                        {
                            Assert.That(i,Is.Zero);Assert.That(plan.Widths[i],Is.EqualTo(768));Assert.That(plan.Columns[i],Is.EqualTo(768));
                            Assert.That(plan.Weights[i],Is.EqualTo(weights));operations.Add(new GpuBlockedMatmul(768,768,plan.Weights[i],384));
                        }
                    }
                    parts=plan.Models;inputs=plan.Inputs;outputs=plan.Outputs;report["native_gelu_count"]=16;
                }
                else
                {
                    var before=ModelTensorSlice.Extract(working,projection.inputs[0],"projection_input");
                    var after=ModelTensorSlice.Extract(working,working.outputs[0].index,"soft_tokens",new[]{new Model.Input{
                        name="projection_output",index=projection.outputs[0],dataType=DataType.Float,shape=new DynamicTensorShape(2520,768)}});
                    Assert.That(before.inputs.Count,Is.EqualTo(1));Assert.That(after.inputs.Count,Is.EqualTo(1));
                    parts=new[]{before,after};inputs=new[]{new[]{"projection_input"}};outputs=new[]{"projection_output"};operations.Add(new GpuBlockedMatmul(768,768,weights,384));
                }
                if(planOnly)
                {
                    report["plan_audit"]=AuditPlan(parts,inputs,outputs);report["inference_executed"]=false;
                    Assert.That(CoefficientsHash(original),Is.EqualTo(originalCoefficients));report["original_coefficient_bytes_preserved"]=true;
                    report["original_coefficients_sha256"]=originalCoefficients;report["phase"]="audit_complete";report["success"]=true;return;
                }
                using var pipeline=new GpuProjectionPipeline(parts,inputs,outputs,operations.ToArray(),releaseUnusedGpuCache);
                const BindingFlags privateFields=BindingFlags.Instance|BindingFlags.NonPublic;
                var deviceType=typeof(Worker).Assembly.GetType("Unity.InferenceEngine.GfxDeviceBackend");
                var backendType=typeof(Worker).Assembly.GetType("Unity.InferenceEngine.GPUComputeBackend");
                var pipelineWorkers=(Worker[])typeof(GpuProjectionPipeline).GetField("workers",privateFields).GetValue(pipeline);
                var backends=pipelineWorkers.Select(worker=>typeof(Worker).GetField("m_Backend",privateFields).GetValue(worker)).ToArray();
                var initialContexts=backends.Select(backend=>deviceType!=null&&deviceType.IsInstanceOfType(backend)?deviceType.GetField("m_GfxContext",privateFields).GetValue(backend):null).ToArray();
                report["release_unused_gpu_cache"]=releaseUnusedGpuCache;
                report["pipeline_prepare_ms"]=phaseTimer.Elapsed.TotalMilliseconds;report["segment_count"]=parts.Length;report["gpu_operation_count"]=operations.Count;
                report["prepare_ms"]=timer.Elapsed.TotalMilliseconds;report["cpu_computation_or_input_count"]=0;report["readback_flush_count"]=0;
                var pixelsName=row.Value<string>("pixels");Check(Path.Combine(root,pixelsName),audit["files"][pixelsName].Value<string>("sha256"));
                using var input=new Tensor<float>(new TensorShape(1,2520,768),Floats(Path.Combine(root,pixelsName)));ComputeTensorData.Pin(input);
                using var process=System.Diagnostics.Process.GetCurrentProcess();GpuExecutionProfiler.ReadProcessMemory(process,out var memoryBefore,out _);
                report["host_working_set_before_execute_bytes"]=memoryBefore;report["phase"]=staged?"execute_async_stage_fences":"execute";Save();timer.Restart();
                report["frame_before_execute"]=Time.frameCount;report["rendered_frame_before_execute"]=Time.renderedFrameCount;
                using var encoded=staged?await pipeline.ExecuteAsync(input):pipeline.Execute(input);Assert.That(encoded.backendType,Is.EqualTo(BackendType.GPUCompute));
                report["frame_after_execute"]=Time.frameCount;report["rendered_frame_after_execute"]=Time.renderedFrameCount;
                var actual=encoded.DownloadToArray();report["execute_and_final_download_ms"]=timer.Elapsed.TotalMilliseconds;
                var backendCaches=new JArray();report["backend_caches"]=backendCaches;
                for(var i=0;i<backends.Length;i++)
                {
                    var backend=backends[i];var cache=backendType.GetField("m_MemoryPool",privateFields).GetValue(backend);
                    var free=(System.Collections.IDictionary)cache.GetType().GetField("freeBuffers",privateFields).GetValue(cache);
                    long bytes=0;foreach(var value in free.Values)bytes+=((ComputeTensorData)value).maxCapacity*4L;
                    var context=initialContexts[i]==null?null:deviceType.GetField("m_GfxContext",privateFields).GetValue(backend);
                    var changed=initialContexts[i]!=null&&!ReferenceEquals(initialContexts[i],context);
                    backendCaches.Add(new JObject{["stage"]=i,["type"]=backend.GetType().FullName,["scratch_free_count"]=free.Count,["scratch_free_bytes"]=bytes,["compiled_context_recreated"]=changed});
                    if(releaseUnusedGpuCache)Assert.That(free.Count,Is.Zero);
                    if(staged&&releaseUnusedGpuCache&&initialContexts[i]!=null)Assert.That(changed,Is.True);
                }
                var geluCaches=new JArray();report["gelu_internal_worker_caches"]=geluCaches;
                foreach(var operation in operations.OfType<GpuGelu>())
                {
                    const BindingFlags fields=BindingFlags.Instance|BindingFlags.NonPublic;
                    var nested=(GpuProjectionPipeline)typeof(GpuGelu).GetField("pipeline",fields).GetValue(operation);
                    if(sleefGelu)
                    {
                        Assert.That(nested,Is.Null,"Precise SLEEF GELU must use its single GPU kernel.");
                        Assert.That(typeof(GpuGelu).GetField("nativeShader",fields).GetValue(operation),Is.Not.Null);
                        geluCaches.Add(new JObject{["single_compute_kernel"]=true,["internal_worker_count"]=0});continue;
                    }
                    var workers=(Worker[])typeof(GpuProjectionPipeline).GetField("workers",fields).GetValue(nested);
                    foreach(var worker in workers)
                    {
                        var storage=typeof(Worker).GetField("m_Storage",fields).GetValue(worker);
                        var pool=storage.GetType().GetField("m_TensorPool",fields).GetValue(storage);
                        var cache=pool.GetType().GetField("m_computeMemoryPool",fields).GetValue(pool);
                        var free=(System.Collections.IDictionary)cache.GetType().GetField("freeBuffers",fields).GetValue(cache);
                        long bytes=0;foreach(var value in free.Values)bytes+=((ComputeTensorData)value).maxCapacity*4L;
                        geluCaches.Add(new JObject{["free_buffer_count"]=free.Count,["free_buffer_capacity_bytes"]=bytes});
                        if(releaseGeluCaches)Assert.That(free.Count,Is.Zero,"Both internal GELU Workers must release their unused caches.");
                    }
                }
                if(nativeGelu)Assert.That(geluCaches.Count,Is.EqualTo(sleefGelu?16:32),"Audit every GELU kernel, or both Workers per legacy GELU.");
                var capture=Path.ChangeExtension(output,"bin");Assert.That(File.Exists(capture),Is.False);var capturedBytes=new byte[actual.Length*4];Buffer.BlockCopy(actual,0,capturedBytes,0,capturedBytes.Length);File.WriteAllBytes(capture,capturedBytes);
                using(var stream=File.OpenRead(capture))report["capture_sha256"]=PlayerFileHash.Compute(stream,out _);
                if(releaseUnusedGpuCache&&caseId=="square-0"&&!nativeOperators)
                {
                    Assert.That(report.Value<string>("capture_sha256"),Is.EqualTo("43fa98f573bd55a0f634075bf17f42fcde2ff9e89894c2d5b1b238c8ab451aab"),"Memory retirement must preserve every bit of the fixed cache-only candidate.");
                    report["cache_only_candidate_bits_preserved"]=true;
                }
                GpuExecutionProfiler.ReadProcessMemory(process,out var memoryAfter,out var peak);report["host_working_set_after_download_bytes"]=memoryAfter;report["host_peak_working_set_bytes"]=peak;
                var expected=row["features"].ToObject<float[][]>().SelectMany(values=>values).ToArray();Assert.That(actual.Length,Is.EqualTo(expected.Length));
                Assert.That(encoded.shape,Is.EqualTo(new TensorShape(reference["profiles"][profile].Value<int>("feature_count"),512)));
                int bad=0;double maximum=0,ratio=0;
                for(var i=0;i<actual.Length;i++)
                {
                    Assert.That(float.IsNaN(actual[i])||float.IsInfinity(actual[i]),Is.False);
                    var d=Math.Abs((double)actual[i]-expected[i]);var allowed=1e-5+1e-4*Math.Abs((double)expected[i]);if(d>allowed)bad++;maximum=Math.Max(maximum,d);ratio=Math.Max(ratio,d/allowed);
                }
                report["actual_backend"]="GPUCompute";report["mismatched_elements"]=bad;report["maximum_absolute_error"]=maximum;report["maximum_tolerance_ratio"]=ratio;report["phase"]="audit_coefficients_after";Save();phaseTimer.Restart();
                Assert.That(CoefficientsHash(original),Is.EqualTo(originalCoefficients));report["original_coefficient_bytes_preserved"]=true;report["original_coefficients_sha256"]=originalCoefficients;
                report["coefficient_audit_after_ms"]=phaseTimer.Elapsed.TotalMilliseconds;report["phase"]="completed";Save();
                Assert.That(bad,Is.Zero,"Keep the unchanged original FP32 elementwise gate.");report["success"]=true;
            }
            catch(Exception error){report["error"]=error.ToString();throw;}
            finally{foreach(var operation in operations)operation.Dispose();Save();}
        }
        static string CoefficientsHash(Model model)
            =>ModelCoefficientHash.Compute(model,out _);
        static JObject AuditPlan(Model[] models,string[][] operationInputs,string[] operationOutputs)
        {
            var shapes=new Dictionary<string,TensorShape>();var rows=new JArray();
            var analysis=typeof(Worker).Assembly.GetType("Unity.InferenceEngine.Compiler.Analyser.PartialInferenceAnalysis");
            JObject Describe(string name,TensorShape shape)
            {
                if(shapes.TryGetValue(name,out var previous))Assert.That(shape,Is.EqualTo(previous));else shapes.Add(name,shape);
                return new JObject{["name"]=name,["shape"]=JArray.FromObject(Enumerable.Range(0,shape.rank).Select(i=>shape[i]).ToArray()),["bytes"]=shape.length*4L};
            }
            long constants=0;int layers=0;
            for(var stage=0;stage<models.Length;stage++)
            {
                var model=models[stage];var context=analysis.GetMethod("InferModelPartialTensors").Invoke(null,new object[]{model});
                var incoming=new JArray();foreach(var input in model.inputs){Assert.That(input.shape.IsStatic(),Is.True);incoming.Add(Describe(input.name,input.shape.ToTensorShape()));}
                var outgoing=new JArray();foreach(var output in model.outputs)
                {
                    var partial=context.GetType().GetMethod("GetPartialTensor").Invoke(context,new object[]{output.index});
                    var shape=(DynamicTensorShape)partial.GetType().GetProperty("shape").GetValue(partial);Assert.That(shape.IsStatic(),Is.True);outgoing.Add(Describe(output.name,shape.ToTensorShape()));
                }
                var bytes=model.constants.Sum(value=>(long)value.lengthBytes);constants+=bytes;layers+=model.layers.Count;
                var consumed=new HashSet<int>(model.layers.SelectMany(layer=>layer.inputs).Concat(model.outputs.Select(value=>value.index)));
                var unused=new JArray();
                foreach(var layer in model.layers)foreach(var index in layer.outputs.Where(index=>index>=0&&!consumed.Contains(index)))
                {
                    var partial=context.GetType().GetMethod("GetPartialTensor").Invoke(context,new object[]{index});
                    var shape=(DynamicTensorShape)partial.GetType().GetProperty("shape").GetValue(partial);
                    var type=(DataType)partial.GetType().GetProperty("dataType").GetValue(partial);
                    Assert.That(shape.IsStatic(),Is.True);
                    unused.Add(new JObject{["index"]=index,["producer"]=layer.GetType().Name,["shape"]=JArray.FromObject(Enumerable.Range(0,shape.rank).Select(i=>shape.ToTensorShape()[i]).ToArray()),
                        ["bytes"]=shape.ToTensorShape().length*4L,["data_type"]=type.ToString()});
                }
                rows.Add(new JObject{["stage"]=stage,["layers"]=model.layers.Count,["shape_layers"]=model.layers.Count(layer=>layer.GetType().Name=="Shape"),
                    ["coefficient_bytes"]=bytes,["inputs"]=incoming,["outputs"]=outgoing,["unconsumed_internal_outputs"]=unused,["unconsumed_internal_output_bytes"]=unused.Sum(value=>value.Value<long>("bytes"))});
            }
            var liveAfter=new HashSet<string>[models.Length];var live=new HashSet<string>{models[models.Length-1].outputs[0].name};
            for(var stage=models.Length-1;stage>=0;stage--)
            {
                liveAfter[stage]=new HashSet<string>(live);live.UnionWith(models[stage].inputs.Select(value=>value.name));
                if(stage<operationInputs.Length)live.UnionWith(operationInputs[stage]);
            }
            var values=new Dictionary<string,int>{{models[0].inputs[0].name,0}};var owned=new Dictionary<int,(string name,int stage,long bytes)>();var serial=0;long peak=0;
            for(var stage=0;stage<models.Length;stage++)
            {
                void Copy(string name){var id=++serial;values[name]=id;owned.Add(id,(name,stage,shapes[name].length*4L));}
                foreach(var output in models[stage].outputs)Copy(output.name);if(stage<operationOutputs.Length)Copy(operationOutputs[stage]);
                var before=owned.Values.Sum(value=>value.bytes);
                foreach(var name in values.Keys.Where(name=>!liveAfter[stage].Contains(name)).ToArray())values.Remove(name);
                var retained=new HashSet<int>(values.Values);foreach(var id in owned.Keys.Where(id=>!retained.Contains(id)).ToArray())owned.Remove(id);
                var after=owned.Values.Sum(value=>value.bytes);peak=Math.Max(peak,after);
                var row=(JObject)rows[stage];row["owned_before_release_bytes"]=before;row["retained_after_stage_bytes"]=after;row["retained_after_stage_count"]=owned.Count;
                row["largest_retained"]=JArray.FromObject(owned.Values.OrderByDescending(value=>value.bytes).Take(5).Select(value=>new{name=value.name,created_stage=value.stage,bytes=value.bytes}).ToArray());
            }
            return new JObject{["models"]=rows,["sum_partition_layers"]=layers,["sum_partition_coefficient_bytes"]=constants,["maximum_retained_boundary_bytes"]=peak,
                ["scope"]="static_metadata_and_runtime_boundary_liveness_simulation_only_no_model_inference"};
        }
        static float[] Floats(string path){var bytes=File.ReadAllBytes(path);var data=new float[bytes.Length/4];Buffer.BlockCopy(bytes,0,data,0,bytes.Length);return data;}
    }
}
