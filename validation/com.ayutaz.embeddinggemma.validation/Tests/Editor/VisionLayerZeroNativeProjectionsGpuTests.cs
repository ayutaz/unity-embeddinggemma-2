using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using EmbeddingGemma.Tests;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEngine;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class VisionLayerZeroNativeProjectionsGpuTests
    {
        [Test]
        public void NativeBlock384VisionLayerZeroFromExactPatchMatchesOriginalOnGpu()
            => Run(0);

        [Test]
        public void ExplicitDivisionFirstRmsAndNativeProjectionsMatchOriginalLayerZeroOnGpu()
            => Run(1);

        [Test]
        public void AllSevenRmsAndNativeProjectionsMatchOriginalLayerZeroOnGpu()
            => Run(7);

        [Test]
        public void InlineSevenRmsPreserveQualifiedLayerBitsAndOriginalNativeGateOnGpu()
            => Run(8);

        [Test]
        public void NativeGeluAndInlineRmsMatchOriginalLayerZeroOnGpu()
            => Run(9);

        [Test]
        public void ReleasedUnusedCachesPreserveNativeGeluLayerBitsOnGpu()
            => Run(10);

        [Test]
        public void QualifiedSoftmaxAndNativeGeluInlineRmsMatchOriginalLayerZeroOnGpu()
            => Run(11);

        static void Run(int normCount)
        {
            var qualifiedFirstRms=normCount>0;
            Assert.That(SystemInfo.supportsComputeShaders,Is.True,"Actual GPU required; never skip.");
            var source=Environment.GetEnvironmentVariable("EG_ACCEPTANCE_SOURCE");
            var root=Path.GetFullPath(Path.Combine(Application.dataPath,"../../k/artifacts/vision-diagnostics"));
            var modelPath=Path.GetFullPath(Path.Combine(Application.dataPath,"../artifacts/vision-layer0-original-patch-gpu-e39bdf1b5d0faf2ea424023d83076315736c6fde-captures/layer0.sentis"));
            var label=normCount==11?"vision-layer0-native-softmax-gelu-rms-gpu-":normCount==10?"vision-layer0-cache-release-gelu-rms-gpu-":normCount==9?"vision-layer0-native-gelu-inline-rms-gpu-":normCount==8?"vision-layer0-inline-rms-native-projections-gpu-":normCount==7?"vision-layer0-seven-rms-native-projections-gpu-":qualifiedFirstRms?"vision-layer0-first-rms-native-projections-gpu-":"vision-layer0-native-projections-gpu-";
            var output=Path.GetFullPath(Path.Combine(Application.dataPath,"../artifacts/"+label+source+".json"));
            Assert.That(File.Exists(output),Is.False);
            var report=new JObject{["success"]=false,["scope"]=normCount==11?"original_vision_layer0_native_softmax_gelu_inline_rms_and_seven_native_gpu_projections_candidate":normCount==10?"original_vision_layer0_cache_release_native_gelu_inline_rms_candidate":normCount==9?"original_vision_layer0_native_gelu_inline_rms_and_seven_native_gpu_projections_candidate":normCount==8?"original_vision_layer0_inline_rms_and_seven_native_gpu_projections_candidate":normCount==7?"original_vision_layer0_seven_rms_and_seven_native_gpu_projections_candidate":qualifiedFirstRms?"original_vision_layer0_first_rms_and_seven_native_gpu_projections_candidate":"original_vision_layer0_seven_native_gpu_projections_candidate",["runtime_source"]=source,
                ["full_encoder_validated"]=false,["end_to_end_validated"]=false,["phase"]="audit"};
            void Save(){Directory.CreateDirectory(Path.GetDirectoryName(output));File.WriteAllText(output,report.ToString());}
            void Check(string path,string sha){using var stream=File.OpenRead(path);Assert.That(PlayerFileHash.Compute(stream,out _),Is.EqualTo(sha));}
            var operations=new List<IGpuTensorOperation>();Save();
            try
            {
                var locked=JObject.Parse(File.ReadAllText(Path.Combine(Application.dataPath,"../Packages/packages-lock.json")));
                foreach(var package in new[]{"com.ayutaz.embeddinggemma","com.ayutaz.embeddinggemma.validation"})Assert.That(locked["dependencies"][package].Value<string>("hash"),Is.EqualTo(source));
                Check(modelPath,"d09b8884fae82ecd0245967774a0c8671efe09156c50367c45b9f133fc53983d");
                Check(Path.Combine(root,"vision-diagnostics.json"),"998b7d0d200b257b0bc19de32f45ef27be4328088d2d51706ea956cc3b4461ce");
                var reference=JObject.Parse(File.ReadAllText(Path.Combine(root,"vision-diagnostics.json")));
                var model=TextModelFile.Load(modelPath);var constants=model.constants.ToDictionary(value=>value.index);
                var selected=model.layers.Where(layer=>layer.GetType().Name=="MatMul"&&!constants.ContainsKey(layer.inputs[0])
                    &&constants.TryGetValue(layer.inputs[1],out var w)&&w.shape.rank==2).Select(layer=>layer.outputs[0]).ToArray();
                Assert.That(selected.Length,Is.EqualTo(7));
                var original=model;var sourceHash=SerializedHash(original);Model rmsTemplate=null;int[] normOutputs=null;Constant[] normScales=null;var softmaxOutput=-1;
                if(normCount>=8)
                {
                    var method=typeof(QualifiedVisionRmsOperation).GetMethod("InlineLayerRms");Assert.That(method,Is.Not.Null,"Inline RMS graph grafting is missing.");
                    Check(Path.Combine(Application.dataPath,"GpuVisionInlineRmsCandidate/inline-rms.onnx"),"15c8ec537c8d40386e61b675c05aaaa65b296667c0d38c39dc7d12fe28a25ccb");
                    Check(Path.Combine(Application.dataPath,"GpuVisionHeadRmsCandidates/head-rms-candidates.json"),"bb1f2d3df3905551b34723badefd7f1dfdc23da08a9ab4c72545d5d1368ce229");
                    var preparation=JObject.Parse(File.ReadAllText(Path.Combine(Application.dataPath,"GpuVisionHeadRmsCandidates/head-rms-candidates.json")));
                    Model LoadHead(int index)
                    {
                        var entry=((JArray)preparation["rows"]).Single(value=>value.Value<int>("index")==index);
                        var asset="GpuVisionHeadRmsCandidates/"+entry["onnx"].Value<string>("file");Check(Path.Combine(Application.dataPath,asset),entry["onnx"].Value<string>("sha256"));
                        return ModelLoader.Load(UnityEditor.AssetDatabase.LoadAssetAtPath<ModelAsset>("Assets/"+asset));
                    }
                    var template=ModelLoader.Load(UnityEditor.AssetDatabase.LoadAssetAtPath<ModelAsset>("Assets/GpuVisionInlineRmsCandidate/inline-rms.onnx"));
                    model=(Model)method.Invoke(null,new object[]{model,template,LoadHead(1),LoadHead(3),1});report["inline_rms_count"]=7;report["layer_count"]=model.layers.Count;
                    if(normCount>=9)
                    {
                        var marked=SavedProjectionPartition.MarkGelu(model);model=marked.model;Assert.That(marked.outputs.Length,Is.EqualTo(1));
                        var wanted=new HashSet<int>(selected.Concat(marked.outputs));selected=model.layers.Where(layer=>wanted.Contains(layer.outputs[0])).Select(layer=>layer.outputs[0]).ToArray();
                        report["gelu_count"]=1;
                    }
                    if(normCount==11)
                    {
                        var softmax=model.layers.Single(layer=>layer.GetType().Name=="Softmax");
                        Assert.That(softmax.inputs.Length,Is.EqualTo(1));Assert.That(softmax.outputs.Length,Is.EqualTo(1));
                        Assert.That((int)softmax.GetType().GetField("axis").GetValue(softmax),Is.EqualTo(-1));
                        var identity=typeof(Worker).Assembly.GetTypes().Single(type=>type.Name=="Identity"&&typeof(Layer).IsAssignableFrom(type));
                        var marker=(Layer)Activator.CreateInstance(identity);marker.inputs=(int[])softmax.inputs.Clone();marker.outputs=(int[])softmax.outputs.Clone();
                        softmaxOutput=marker.outputs[0];
                        model=new Model{ProducerName=model.ProducerName+"/qualified-softmax-marker",inputs=new List<Model.Input>(model.inputs),outputs=new List<Model.Output>(model.outputs),
                            constants=new List<Constant>(model.constants),layers=model.layers.Select(layer=>ReferenceEquals(layer,softmax)?marker:layer).ToList()};
                        var wanted=new HashSet<int>(selected){softmaxOutput};selected=model.layers.Where(layer=>wanted.Contains(layer.outputs[0])).Select(layer=>layer.outputs[0]).ToArray();
                        Assert.That(selected.Length,Is.EqualTo(9));report["softmax_count"]=1;
                    }
                }
                if(normCount==1)
                {
                    var templatePath=Path.Combine(Application.dataPath,"GpuVisionFirstRmsCandidate/first-rms.onnx");
                    Check(templatePath,"724605434efc0b70bb28b59b5dc639f8b0aeb765a4170ec782397506f6549976");
                    rmsTemplate=ModelLoader.Load(UnityEditor.AssetDatabase.LoadAssetAtPath<ModelAsset>("Assets/GpuVisionFirstRmsCandidate/first-rms.onnx"));
                    var producer=model.layers.SelectMany(layer=>layer.outputs.Select(index=>(index,layer))).ToDictionary(row=>row.index,row=>row.layer);
                    var firstProjection=producer[selected[0]];var reshape=producer[firstProjection.inputs[0]];
                    Assert.That(reshape.GetType().Name,Is.EqualTo("Reshape"));
                    var final=producer[reshape.inputs[0]];Assert.That(final.GetType().Name,Is.EqualTo("Mul"));
                    var scale=constants[final.inputs.Single(constants.ContainsKey)];
                    Assert.That(scale.shape,Is.EqualTo(new TensorShape(768)));Assert.That(scale.dataType,Is.EqualTo(DataType.Float));
                    var rawScale=(ArraySegment<byte>)typeof(Constant).GetField("array",BindingFlags.Instance|BindingFlags.NonPublic).GetValue(scale);
                    using(var bytes=new MemoryStream(rawScale.Array,rawScale.Offset,rawScale.Count,false))
                        Assert.That(PlayerFileHash.Compute(bytes,out _),Is.EqualTo("acc544061ccfe25009301bdfc15bab61468511ef6f27962b70c54efb65da9cb7"));
                    Check(Path.Combine(Application.dataPath,"GpuVisionFirstRmsCandidate/scale-fp32.bin"),"acc544061ccfe25009301bdfc15bab61468511ef6f27962b70c54efb65da9cb7");
                    var chain=new HashSet<Layer>();var pending=new Stack<int>();pending.Push(final.outputs[0]);
                    while(pending.Count>0)
                    {
                        var value=pending.Pop();if(value==model.inputs[0].index||constants.ContainsKey(value))continue;
                        var layer=producer[value];if(!chain.Add(layer))continue;
                        foreach(var operand in layer.inputs.Where(index=>index>=0))pending.Push(operand);
                    }
                    Assert.That(chain.Select(layer=>layer.GetType().Name).OrderBy(name=>name),
                        Is.EqualTo(new[]{"Mul","Mul","ReduceMean","Rsqrt","ScalarMad","Square"}));
                    foreach(var layer in chain.Where(layer=>!ReferenceEquals(layer,final)))
                        Assert.That(model.layers.Where(consumer=>consumer.inputs.Intersect(layer.outputs).Any()).All(chain.Contains),Is.True,"An internal RMS value escapes its cut.");
                    var scalar=chain.Single(layer=>layer.GetType().Name=="ScalarMad");
                    Assert.That((float)scalar.GetType().GetField("sFloat").GetValue(scalar),Is.EqualTo(1f));
                    Assert.That((float)scalar.GetType().GetField("bFloat").GetValue(scalar),Is.EqualTo(1e-6f));
                    var identity=typeof(Worker).Assembly.GetTypes().Single(type=>type.Name=="Identity"&&typeof(Layer).IsAssignableFrom(type));
                    var marker=(Layer)Activator.CreateInstance(identity);marker.inputs=new[]{model.inputs[0].index};marker.outputs=(int[])final.outputs.Clone();
                    model=new Model{ProducerName=model.ProducerName+"/qualified-first-rms-marker",inputs=new List<Model.Input>(model.inputs),outputs=new List<Model.Output>(model.outputs),
                        constants=new List<Constant>(model.constants),layers=model.layers.Select(layer=>ReferenceEquals(layer,final)?marker:layer).ToList()};
                    selected=new[]{marker.outputs[0]}.Concat(selected).ToArray();
                    report["first_rms_replaced"]=true;report["source_scale_sha256"]="acc544061ccfe25009301bdfc15bab61468511ef6f27962b70c54efb65da9cb7";
                }
                if(normCount==7)
                {
                    Check(Path.Combine(Application.dataPath,"GpuVisionFirstRmsCandidate/first-rms.onnx"),"724605434efc0b70bb28b59b5dc639f8b0aeb765a4170ec782397506f6549976");
                    rmsTemplate=ModelLoader.Load(UnityEditor.AssetDatabase.LoadAssetAtPath<ModelAsset>("Assets/GpuVisionFirstRmsCandidate/first-rms.onnx"));
                    var marked=QualifiedVisionRmsOperation.MarkAll(model);model=marked.model;normOutputs=marked.outputs;normScales=marked.scales;
                    var wanted=new HashSet<int>(selected.Concat(normOutputs));selected=model.layers.Where(layer=>wanted.Contains(layer.outputs[0])).Select(layer=>layer.outputs[0]).ToArray();
                    Assert.That(selected.Length,Is.EqualTo(14));report["rms_count"]=7;
                }
                report["phase"]="prepare";Save();
                var timer=System.Diagnostics.Stopwatch.StartNew();var plan=SavedProjectionPartition.Create(model,selected);
                var rows=new JArray();report["projections"]=rows;
                for(var i=0;i<selected.Length;i++)
                {
                    if(plan.Weights[i]==null)
                    {
                        if(selected[i]==softmaxOutput)
                        {
                            Assert.That(plan.Widths[i],Is.EqualTo(2520));Assert.That(plan.Rows[i],Is.EqualTo(30240));
                            operations.Add(new GpuSoftmax());rows.Add(new JObject{["kind"]="native_softmax",["rows"]=plan.Rows[i],["width"]=plan.Widths[i]});continue;
                        }
                        if(normCount>=9)
                        {
                            Assert.That(plan.Widths[i],Is.EqualTo(3072));Assert.That(plan.Rows[i],Is.EqualTo(2520));
                            operations.Add(new GpuGelu(new TensorShape(1,2520,3072)));rows.Add(new JObject{["kind"]="native_tanh_gelu",["rows"]=plan.Rows[i],["width"]=plan.Widths[i]});continue;
                        }
                        if(normCount==1)
                        {
                            Assert.That(i,Is.Zero);operations.Add(new QualifiedVisionRmsOperation(rmsTemplate));
                            rows.Add(new JObject{["kind"]="explicit_division_first_rms",["rows"]=plan.Rows[i],["width"]=plan.Widths[i]});continue;
                        }
                        Assert.That(normCount,Is.EqualTo(7));var ordinal=Array.IndexOf(normOutputs,selected[i]);Assert.That(ordinal,Is.GreaterThanOrEqualTo(0));
                        Model template;
                        if(plan.Widths[i]==64)
                        {
                            Assert.That(ordinal,Is.InRange(1,3));Assert.That(plan.Rows[i],Is.EqualTo(30240));
                            var headRoot=Path.Combine(Application.dataPath,"GpuVisionHeadRmsCandidates");
                            Check(Path.Combine(headRoot,"head-rms-candidates.json"),"bb1f2d3df3905551b34723badefd7f1dfdc23da08a9ab4c72545d5d1368ce229");
                            var preparation=JObject.Parse(File.ReadAllText(Path.Combine(headRoot,"head-rms-candidates.json")));
                            var entry=((JArray)preparation["rows"]).Single(value=>value.Value<int>("index")==ordinal);
                            Check(Path.Combine(headRoot,entry["onnx"].Value<string>("file")),entry["onnx"].Value<string>("sha256"));
                            template=ModelLoader.Load(UnityEditor.AssetDatabase.LoadAssetAtPath<ModelAsset>("Assets/GpuVisionHeadRmsCandidates/"+entry["onnx"].Value<string>("file")));
                            if(normScales[ordinal]!=null)
                                Assert.That(CoefficientHash(template.constants.Single(value=>value.dataType==DataType.Float&&value.shape==new TensorShape(64))),Is.EqualTo(CoefficientHash(normScales[ordinal])));
                            else Assert.That(template.constants.Any(value=>value.dataType==DataType.Float&&value.shape==new TensorShape(64)),Is.False);
                        }
                        else
                        {
                            Assert.That(plan.Widths[i],Is.EqualTo(768));Assert.That(plan.Rows[i],Is.EqualTo(2520));
                            template=QualifiedVisionRmsOperation.BorrowScale(rmsTemplate,normScales[ordinal]);
                        }
                        operations.Add(new QualifiedVisionRmsOperation(template));
                        rows.Add(new JObject{["kind"]="qualified_rms",["ordinal"]=ordinal,["rows"]=plan.Rows[i],["width"]=plan.Widths[i],
                            ["scale_sha256"]=normScales[ordinal]==null?null:CoefficientHash(normScales[ordinal])});continue;
                    }
                    Assert.That(plan.Rows[i],Is.EqualTo(2520));
                    operations.Add(new GpuBlockedMatmul(plan.Widths[i],plan.Columns[i],plan.Weights[i],384));
                    var bytes=new byte[plan.Weights[i].Length*4];Buffer.BlockCopy(plan.Weights[i],0,bytes,0,bytes.Length);
                    using var stream=new MemoryStream(bytes);rows.Add(new JObject{["width"]=plan.Widths[i],["columns"]=plan.Columns[i],["rows"]=plan.Rows[i],["block"]=384,["weight_sha256"]=PlayerFileHash.Compute(stream,out _)});
                }
                using var pipeline=new GpuProjectionPipeline(plan.Models,plan.Inputs,plan.Outputs,operations.ToArray(),normCount>=10);
                report["release_unused_gpu_cache"]=normCount>=10;
                report["prepare_ms"]=timer.Elapsed.TotalMilliseconds;report["segment_count"]=plan.Models.Length;report["cpu_computation_or_input_count"]=0;report["readback_flush_count"]=0;
                var inputPath=Path.Combine(root,"native-stage-0.bin");var expectedPath=Path.Combine(root,"native-stage-3.bin");
                foreach(var file in new[]{inputPath,expectedPath})Check(file,reference["files"][Path.GetFileName(file)].Value<string>("sha256"));
                using var input=new Tensor<float>(new TensorShape(1,2520,768),Floats(inputPath));ComputeTensorData.Pin(input);
                report["phase"]="execute";Save();timer.Restart();using var encoded=pipeline.Execute(input);
                Assert.That(encoded.backendType,Is.EqualTo(BackendType.GPUCompute));var actual=encoded.DownloadToArray();
                report["execute_and_final_download_ms"]=timer.Elapsed.TotalMilliseconds;var expected=Floats(expectedPath);Assert.That(actual.Length,Is.EqualTo(expected.Length));
                var capture=Path.ChangeExtension(output,"bin");var raw=new byte[actual.Length*4];Buffer.BlockCopy(actual,0,raw,0,raw.Length);File.WriteAllBytes(capture,raw);
                using(var stream=File.OpenRead(capture))report["capture_sha256"]=PlayerFileHash.Compute(stream,out _);
                if(normCount==10)
                {
                    Assert.That(report.Value<string>("capture_sha256"),Is.EqualTo("fe3b9fe2e554d83acf9bbf82e6f89974c78db06a93f99cad4f63a7ac251e59ba"),"Releasing unused cache must preserve every bit of the qualified GELU/RMS layer.");
                    report["qualified_gelu_rms_layer_bits_preserved"]=true;
                }
                if(normCount==8)
                {
                    Assert.That(report.Value<string>("capture_sha256"),Is.EqualTo("8312ee24467c3ee955dbd21f27e79fcd980c31e5b23ad442f8a4ee324c422448"),"Inlining must preserve every bit of the qualified seven-RMS layer.");
                    report["qualified_seven_rms_layer_bits_preserved"]=true;
                }
                int bits=0,bad=0;double maximum=0,ratio=0;
                for(var i=0;i<actual.Length;i++)
                {
                    Assert.That(float.IsNaN(actual[i])||float.IsInfinity(actual[i]),Is.False);
                    if(BitConverter.SingleToInt32Bits(actual[i])!=BitConverter.SingleToInt32Bits(expected[i]))bits++;
                    var d=Math.Abs((double)actual[i]-expected[i]);var allowed=1e-5+1e-4*Math.Abs((double)expected[i]);if(d>allowed)bad++;maximum=Math.Max(maximum,d);ratio=Math.Max(ratio,d/allowed);
                }
                report["actual_backend"]="GPUCompute";report["bit_differences"]=bits;report["mismatched_elements"]=bad;report["maximum_absolute_error"]=maximum;report["maximum_tolerance_ratio"]=ratio;report["phase"]="completed";Save();
                Assert.That(SerializedHash(original),Is.EqualTo(sourceHash));report["source_model_bytes_preserved"]=true;report["source_model_serialized_sha256"]=sourceHash;
                Assert.That(bad,Is.Zero,"Keep the unchanged original layer-0 FP32 gate.");report["success"]=true;
            }
            catch(Exception error){report["error"]=error.ToString();throw;}
            finally{foreach(var operation in operations)operation.Dispose();Save();}
        }
        static string CoefficientHash(Constant value){var raw=(ArraySegment<byte>)typeof(Constant).GetField("array",BindingFlags.Instance|BindingFlags.NonPublic).GetValue(value);using var stream=new MemoryStream(raw.Array,raw.Offset,raw.Count,false);return PlayerFileHash.Compute(stream,out _);}
        static string SerializedHash(Model model){using var stream=new MemoryStream();ModelWriter.Save(stream,model);stream.Position=0;return PlayerFileHash.Compute(stream,out _);}
        static float[] Floats(string path){var bytes=File.ReadAllBytes(path);var data=new float[bytes.Length/4];Buffer.BlockCopy(bytes,0,data,0,bytes.Length);return data;}
    }
}
