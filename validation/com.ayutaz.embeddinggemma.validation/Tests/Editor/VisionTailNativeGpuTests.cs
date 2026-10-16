using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEditor;
using UnityEngine;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class VisionTailNativeGpuTests
    {
        [Test]
        public void OriginalTailRmsAndProjectionQualifyAgainstPinnedNativeStagesOnGpu()
            =>Run(false);

        [Test]
        public void SavedOriginalCpuTailRmsAndBlock192ProjectionMustMatchObservedBitsOnGpu()
            =>Run(true);

        void Run(bool savedOriginalCpuContext)
        {
            Assert.That(SystemInfo.supportsComputeShaders,Is.True,"Actual GPU required; no skip.");
            var source=Environment.GetEnvironmentVariable("EG_ACCEPTANCE_SOURCE");var root=Path.GetFullPath(Path.Combine(Application.dataPath,"../../k/artifacts/vision-diagnostics"));
            var label=savedOriginalCpuContext?"vision-saved-original-cpu-tail-gpu-":"vision-tail-native-gpu-";
            var output=Path.GetFullPath(Path.Combine(Application.dataPath,"../artifacts/"+label+source+".json"));Assert.That(File.Exists(output),Is.False);
            var report=new JObject{["success"]=false,["runtime_source"]=source,["scope"]="original_final_rms_and_projection_exact_native_input_component_gates_only",
                ["full_encoder_validated"]=false,["end_to_end_validated"]=false,["phase"]="audit"};
            report["saved_original_cpu_context"]=savedOriginalCpuContext;
            report["original_generation_intermediate_reference"]=false;
            void Save(){Directory.CreateDirectory(Path.GetDirectoryName(output));File.WriteAllText(output,report.ToString());}
            string Hash(string path){using var stream=File.OpenRead(path);return PlayerFileHash.Compute(stream,out _);}
            void Check(string path,string hash)=>Assert.That(Hash(path),Is.EqualTo(hash));
            Save();
            try
            {
                var locked=JObject.Parse(File.ReadAllText(Path.Combine(Application.dataPath,"../Packages/packages-lock.json")));
                foreach(var package in new[]{"com.ayutaz.embeddinggemma","com.ayutaz.embeddinggemma.validation"})Assert.That(locked["dependencies"][package].Value<string>("hash"),Is.EqualTo(source));
                Check(Path.Combine(root,"vision-diagnostics.json"),"998b7d0d200b257b0bc19de32f45ef27be4328088d2d51706ea956cc3b4461ce");
                var reference=JObject.Parse(File.ReadAllText(Path.Combine(root,"vision-diagnostics.json")));
                var cpuRoot=Path.GetFullPath(Path.Combine(Application.dataPath,"../../numerical-acceptance-state/original-vision-node-capture-run-38713178242/original-vision-context-native-d2663b56068d6cb4c49c6eae79d2cef58782211a/original-context-results"));
                JObject cpuReference=null;
                if(savedOriginalCpuContext)
                {
                    var cpuReport=Path.Combine(cpuRoot,"original-image-native.json");Check(cpuReport,"cb3b46ae4cc6ec2d7aabb5e2e2abe756df9274ab864b824fc7363e955306885a");
                    cpuReference=JObject.Parse(File.ReadAllText(cpuReport));
                    Assert.That(cpuReference.Value<string>("graph_sha256"),Is.EqualTo("f9c5175a1d75289ce814a9651f496f9bdae5e873ac66199a5c3d03e0172c16cc"));
                    Assert.That(cpuReference["saved_node_capture"].Value<bool>("observed_final_matches_normal_bits"),Is.True);
                    report["reference_cpu_graph_sha256"]=cpuReference["graph_sha256"];
                }
                var path=Path.Combine(root,"vision-stages-fp32.sentis");Check(path,"63f25d442e6921529e009c4dbf9ca7cec7528499959aaf3a043f072723135508");
                var original=TextModelFile.Load(path);Assert.That(original.outputs.Count,Is.EqualTo(22));var before=ModelCoefficientHash.Compute(original,out _);
                var analysis=typeof(Worker).Assembly.GetType("Unity.InferenceEngine.Compiler.Analyser.PartialInferenceAnalysis");
                var context=analysis.GetMethod("InferModelPartialTensors").Invoke(null,new object[]{original});
                for(var stage=19;stage<=21;stage++)
                {
                    var partial=context.GetType().GetMethod("GetPartialTensor").Invoke(context,new object[]{original.outputs[stage].index});
                    var shape=(DynamicTensorShape)partial.GetType().GetProperty("shape").GetValue(partial);Assert.That(shape.IsStatic(),Is.True);
                    Assert.That(shape.ToTensorShape(),Is.EqualTo(new TensorShape(256,stage==21?512:768)));
                }
                Model Slice(int result,int input,string name)=>ModelTensorSlice.Extract(original,original.outputs[result].index,name,new[]{new Model.Input{
                    name="input",index=original.outputs[input].index,dataType=DataType.Float,shape=new DynamicTensorShape(256,768)}});
                var rms=Slice(20,19,"normalized");var projection=Slice(21,20,"projected");
                Assert.That(rms.inputs.Count,Is.EqualTo(1));Assert.That(projection.inputs.Count,Is.EqualTo(1));
                var square=rms.layers.Single(layer=>layer.GetType().Name=="Square");var scalar=rms.layers.Single(layer=>layer.GetType().Name=="ScalarMad");
                report["rms_scale_parameter"]=(float)scalar.GetType().GetField("sFloat").GetValue(scalar);
                report["rms_epsilon"]=(float)scalar.GetType().GetField("bFloat").GetValue(scalar);
                report["rms_layers"]=JArray.FromObject(rms.layers.Select(layer=>layer.GetType().Name).ToArray());
                var product=projection.layers.Single(layer=>layer.GetType().Name=="MatMul"||layer.GetType().Name=="MatMul2D");
                var plan=SavedProjectionPartition.Create(projection,product.outputs);Assert.That(plan.Widths.Single(),Is.EqualTo(768));Assert.That(plan.Columns.Single(),Is.EqualTo(512));Assert.That(plan.Rows.Single(),Is.EqualTo(256));
                var folder=Path.ChangeExtension(output,null)+"-captures";Assert.That(Directory.Exists(folder),Is.False);Directory.CreateDirectory(folder);
                TextModelFile.Save(rms,Path.Combine(folder,"tail-rms.sentis"));TextModelFile.Save(projection,Path.Combine(folder,"tail-projection.sentis"));
                JObject Write(string name,float[] values,TensorShape shape)
                {
                    var bytes=new byte[values.Length*4];Buffer.BlockCopy(values,0,bytes,0,bytes.Length);var target=Path.Combine(folder,name+".bin");File.WriteAllBytes(target,bytes);
                    return new JObject{["file"]=target,["sha256"]=Hash(target),["shape"]=JArray.FromObject(Enumerable.Range(0,shape.rank).Select(i=>shape[i]).ToArray())};
                }
                var scales=rms.constants.Where(value=>value.dataType==DataType.Float&&value.shape.length==768).ToArray();Assert.That(scales.Length,Is.LessThanOrEqualTo(1));
                if(scales.Length==1)
                {
                    var raw=(ArraySegment<byte>)typeof(Constant).GetField("array",BindingFlags.Instance|BindingFlags.NonPublic).GetValue(scales[0]);
                    var values=new float[768];Buffer.BlockCopy(raw.Array,raw.Offset,values,0,raw.Count);report["rms_scale"]=Write("rms-scale",values,scales[0].shape);
                }
                report["projection_weights"]=Write("projection-weights",plan.Weights.Single(),new TensorShape(768,512));
                var cases=new JArray();report["cases"]=cases;var exactBlocks=new HashSet<int>{192,384};var passed=true;
                report["phase"]="execute_components";Save();
                for(var caseIndex=0;caseIndex<(savedOriginalCpuContext?1:2);caseIndex++)
                {
                    string Node(int index)=>index==19?"mul_372":index==20?"mul_373":"mm_113";
                    string Stage(int index)=>savedOriginalCpuContext?Path.Combine(cpuRoot,"captures",Node(index)+"-fp32.bin"):
                        Path.Combine(root,caseIndex==0?"native-stage-"+index+".bin":"native-case-1-stage-"+index+".bin");
                    foreach(var stage in new[]{19,20,21})
                    {
                        var hash=savedOriginalCpuContext?((JArray)cpuReference["saved_node_capture"]["nodes"]).Single(value=>value.Value<string>("node")==Node(stage)).Value<string>("sha256"):
                            reference["files"][Path.GetFileName(Stage(stage))].Value<string>("sha256");
                        Check(Stage(stage),hash);
                    }
                    using var features=new Tensor<float>(new TensorShape(256,768),Floats(Stage(19)));ComputeTensorData.Pin(features);
                    using var normalized=new Tensor<float>(new TensorShape(256,768),Floats(Stage(20)));ComputeTensorData.Pin(normalized);
                    using var rmsPipeline=new GpuProjectionPipeline(new[]{rms},Array.Empty<string>(),Array.Empty<string>(),Array.Empty<IGpuTensorOperation>());
                    using var projectionPipeline=new GpuProjectionPipeline(new[]{projection},Array.Empty<string>(),Array.Empty<string>(),Array.Empty<IGpuTensorOperation>());
                    using var rmsOutput=rmsPipeline.Execute(features);using var projectionOutput=projectionPipeline.Execute(normalized);
                    var rmsActual=rmsOutput.DownloadToArray();var projectionActual=projectionOutput.DownloadToArray();var expectedRms=Floats(Stage(20));var expectedProjection=Floats(Stage(21));
                    var row=new JObject{["case_index"]=caseIndex,["rms"]=Metrics(rmsActual,expectedRms),["projection"]=Metrics(projectionActual,expectedProjection),
                        ["rms_output"]=Write("case-"+caseIndex+"-rms-output",rmsActual,rmsOutput.shape),["projection_output"]=Write("case-"+caseIndex+"-projection-output",projectionActual,projectionOutput.shape)};
                    var candidates=new JArray();row["projection_candidates"]=candidates;cases.Add(row);
                    if(savedOriginalCpuContext)
                    {
                        var asset="Assets/GpuVisionTailRmsCandidate/tail-rms.onnx";
                        Check(Path.Combine(Application.dataPath,"GpuVisionTailRmsCandidate/tail-rms.onnx"),"af13361f2eee7c768e53ba4bedabead4e21c4092ddbd50a869e6735b1b2d8f5c");
                        var model=ModelLoader.Load(AssetDatabase.LoadAssetAtPath<ModelAsset>(asset));
                        var candidateBefore=ModelCoefficientHash.Compute(model,out _);
                        TextModelFile.Save(model,Path.Combine(folder,"candidate-tail-rms.sentis"));
                        using var candidatePipeline=new GpuProjectionPipeline(new[]{model},Array.Empty<string>(),Array.Empty<string>(),Array.Empty<IGpuTensorOperation>());
                        using var candidateOutput=candidatePipeline.Execute(features);Assert.That(candidateOutput.backendType,Is.EqualTo(BackendType.GPUCompute));
                        var candidateValues=candidateOutput.DownloadToArray();var candidateMetrics=Metrics(candidateValues,expectedRms);
                        candidateMetrics["output"]=Write("case-"+caseIndex+"-candidate-rms-output",candidateValues,candidateOutput.shape);
                        candidateMetrics["layer_count"]=model.layers.Count;row["candidate_rms"]=candidateMetrics;
                        Assert.That(ModelCoefficientHash.Compute(model,out _),Is.EqualTo(candidateBefore));
                    }
                    foreach(var block in new[]{192,384})
                    {
                        using var operation=new GpuBlockedMatmul(768,512,plan.Weights.Single(),block);
                        using var pipeline=new GpuProjectionPipeline(plan.Models,plan.Inputs,plan.Outputs,new IGpuTensorOperation[]{operation},true);
                        using var result=pipeline.Execute(normalized);var actual=result.DownloadToArray();var metrics=Metrics(actual,expectedProjection);metrics["block"]=block;
                        metrics["output"]=Write("case-"+caseIndex+"-projection-block-"+block,actual,result.shape);candidates.Add(metrics);
                        if(metrics.Value<int>("bit_differences")!=0)exactBlocks.Remove(block);
                    }
                    passed&=row["rms"].Value<int>("mismatched_elements")==0&&row["projection"].Value<int>("mismatched_elements")==0;
                    passed&=candidates.Any(value=>value.Value<int>("mismatched_elements")==0);Save();
                    if(savedOriginalCpuContext)
                        passed&=row["candidate_rms"].Value<int>("bit_differences")==0&&candidates.Single(value=>value.Value<int>("block")==192).Value<int>("bit_differences")==0;
                }
                report["projection_blocks_exact_on_both_cases"]=JArray.FromObject(exactBlocks.OrderBy(value=>value).ToArray());
                Assert.That(ModelCoefficientHash.Compute(original,out _),Is.EqualTo(before));report["original_coefficients_sha256"]=before;report["original_coefficient_bytes_preserved"]=true;
                report["actual_backend"]="GPUCompute";report["cpu_computation_or_input_count"]=0;report["readback_flush_count"]=0;report["phase"]="completed";Save();
                Assert.That(passed,Is.True,"Keep the unchanged original FP32 tolerances for the isolated native tail components.");report["success"]=true;
            }
            catch(Exception error){report["error"]=error.ToString();throw;}
            finally{Save();}
        }
        static JObject Metrics(float[] actual,float[] expected)
        {
            Assert.That(actual.Length,Is.EqualTo(expected.Length));Assert.That(Array.TrueForAll(actual,value=>!float.IsNaN(value)&&!float.IsInfinity(value)),Is.True);
            int bits=0,bad=0;double maximum=0,ratio=0;
            for(var i=0;i<actual.Length;i++)
            {
                if(BitConverter.SingleToInt32Bits(actual[i])!=BitConverter.SingleToInt32Bits(expected[i]))bits++;
                var d=Math.Abs((double)actual[i]-expected[i]);var allowed=1e-5+1e-4*Math.Abs((double)expected[i]);if(d>allowed)bad++;maximum=Math.Max(maximum,d);ratio=Math.Max(ratio,d/allowed);
            }
            return new JObject{["bit_differences"]=bits,["mismatched_elements"]=bad,["maximum_absolute_error"]=maximum,["maximum_tolerance_ratio"]=ratio};
        }
        static float[] Floats(string path){var bytes=File.ReadAllBytes(path);var values=new float[bytes.Length/4];Buffer.BlockCopy(bytes,0,values,0,bytes.Length);return values;}
    }
}
