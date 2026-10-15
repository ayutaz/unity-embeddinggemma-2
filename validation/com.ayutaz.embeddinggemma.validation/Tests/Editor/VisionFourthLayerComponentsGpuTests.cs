using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEditor;
using UnityEngine;
using UnityEngine.TestTools;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class VisionFourthLayerComponentsGpuTests
    {
        [UnityTest]
        public IEnumerator FourthLayerRmsAndGeluMustMatchQualifiedRowBitsOnGpu()
        {
            Assert.That(SystemInfo.supportsComputeShaders,Is.True,"Actual GPU required; never skip.");
            var source=Environment.GetEnvironmentVariable("EG_ACCEPTANCE_SOURCE");
            var root=Path.GetFullPath(Path.Combine(Application.dataPath,"../../numerical-acceptance-state/lf38743853944"));
            var output=Path.GetFullPath(Path.Combine(Application.dataPath,"../artifacts/vision-fourth-layer-components-gpu-"+source+".json"));
            Assert.That(File.Exists(output),Is.False);
            var rows=new JArray();var report=new JObject{["success"]=false,["runtime_source"]=source,
                ["scope"]="fourth_vision_layer_original_scales_and_qualified_token_row_173_rms_gelu_components_only",
                ["selected_row"]=173,["rms"]=rows,["phase"]="audit",["full_encoder_validated"]=false,["end_to_end_validated"]=false,
                ["other_rms_input_rows_are_diagnostic_zeros"]=true,["native_avx512_intermediate_capture"]=false};
            void Save(){Directory.CreateDirectory(Path.GetDirectoryName(output));File.WriteAllText(output,report.ToString());}
            string Hash(string path){using var stream=File.OpenRead(path);return PlayerFileHash.Compute(stream,out _);}
            void Check(string path,string expected)=>Assert.That(Hash(path),Is.EqualTo(expected));
            Save();
            try
            {
                var packages=JObject.Parse(File.ReadAllText(Path.Combine(Application.dataPath,"../Packages/packages-lock.json")));
                foreach(var package in new[]{"com.ayutaz.embeddinggemma","com.ayutaz.embeddinggemma.validation"})
                    Assert.That(packages["dependencies"][package].Value<string>("hash"),Is.EqualTo(source));
                Check(Path.Combine(root,"layer-four-context.json"),"99bb3c88af3f4725ad359fe0d37f561f80a1414107e82e83ad2f2e79c44e4075");
                var context=JObject.Parse(File.ReadAllText(Path.Combine(root,"layer-four-context.json")));
                Assert.That(context.Value<bool>("success"),Is.True);Assert.That(context.Value<bool>("interval_matches_qualified_bits"),Is.True);
                Assert.That(context.Value<bool>("full_encoder_executed"),Is.False);Assert.That(context.Value<int>("row_index"),Is.EqualTo(173));
                Assert.That(context["capture"]["original_golden_metrics"].Value<double>("maximum_absolute_error"),Is.Zero);
                float[] Node(string name)
                {
                    var entry=context["files"][name];var path=Path.Combine(root,"row-captures",entry.Value<string>("file"));Check(path,entry.Value<string>("sha256"));
                    Assert.That(entry.Value<int>("selected_start"),Is.EqualTo(173));Assert.That(entry.Value<int>("selected_axis"),Is.EqualTo(1));
                    return Floats(path);
                }
                var modelPath=Path.GetFullPath(Path.Combine(Application.dataPath,"../../k/artifacts/vision-encoder/square-fp32.sentis"));
                Check(modelPath,"4ecd0106463e833588c7584a58ea4973fdbab1eb1a56389c2f1597d3e7df34e7");
                var original=TextModelFile.Load(modelPath);var coefficients=ModelCoefficientHash.Compute(original,out _);
                var constants=original.constants.ToDictionary(value=>value.index);
                Check(Path.Combine(Application.dataPath,"GpuVisionInlineRmsCandidate/inline-rms.onnx"),"15c8ec537c8d40386e61b675c05aaaa65b296667c0d38c39dc7d12fe28a25ccb");
                Check(Path.Combine(Application.dataPath,"GpuVisionHeadRmsCandidates/head-rms-candidates.json"),"bb1f2d3df3905551b34723badefd7f1dfdc23da08a9ab4c72545d5d1368ce229");
                var heads=JObject.Parse(File.ReadAllText(Path.Combine(Application.dataPath,"GpuVisionHeadRmsCandidates/head-rms-candidates.json")));
                Model Head(int index)
                {
                    var entry=heads["rows"].Single(value=>value.Value<int>("index")==index);
                    var asset="GpuVisionHeadRmsCandidates/"+entry["onnx"].Value<string>("file");Check(Path.Combine(Application.dataPath,asset),entry["onnx"].Value<string>("sha256"));
                    return ModelLoader.Load(AssetDatabase.LoadAssetAtPath<ModelAsset>("Assets/"+asset));
                }
                var hidden=ModelLoader.Load(AssetDatabase.LoadAssetAtPath<ModelAsset>("Assets/GpuVisionInlineRmsCandidate/inline-rms.onnx"));
                var working=QualifiedVisionRmsOperation.InlineLayerRms(original,hidden,Head(1),Head(3),16);
                var squares=original.layers.Where(layer=>layer.GetType().Name=="Square").Skip(21).Take(7).ToArray();Assert.That(squares.Length,Is.EqualTo(7));
                Layer Next(Layer value,string kind)=>original.layers.Single(layer=>layer.GetType().Name==kind&&layer.inputs.Contains(value.outputs[0]));
                var folder=Path.ChangeExtension(output,null)+"-captures";Directory.CreateDirectory(folder);
                JObject Capture(string name,float[] values)
                {
                    var path=Path.Combine(folder,name+".bin");var bytes=new byte[values.Length*4];Buffer.BlockCopy(values,0,bytes,0,bytes.Length);File.WriteAllBytes(path,bytes);
                    return new JObject{["file"]=path,["sha256"]=Hash(path)};
                }
                for(var i=0;i<7;i++)
                {
                    var spec=context["rms_operations"][i];var square=squares[i];var mean=Next(square,"ReduceMean");var scalar=Next(mean,"ScalarMad");var inverse=Next(scalar,"Rsqrt");var normalized=Next(inverse,"Mul");
                    var scaled=original.layers.SingleOrDefault(layer=>layer.GetType().Name=="Mul"&&layer.inputs.Contains(normalized.outputs[0])
                        &&layer.inputs.Any(index=>constants.TryGetValue(index,out var c)&&c.dataType==DataType.Float&&c.shape.rank==1));
                    Assert.That(scaled==null,Is.EqualTo(i==3));
                    var row=new JObject{["index"]=i};rows.Add(row);
                    if(scaled!=null)
                    {
                        var scale=constants[scaled.inputs.Single(constants.ContainsKey)];var segment=(ArraySegment<byte>)typeof(Constant).GetField("array",BindingFlags.Instance|BindingFlags.NonPublic).GetValue(scale);
                        using var stream=new MemoryStream(segment.Array,segment.Offset,segment.Count,false);var digest=PlayerFileHash.Compute(stream,out _);
                        Assert.That(digest,Is.EqualTo(spec["scale"].Value<string>("sha256")),"Original Sentis and qualified CPU learned scale bytes must agree.");row["learned_scale_sha256"]=digest;
                    }
                    var name=spec.Value<string>("input_node");var dimensions=context["files"][name]["source_shape"].Values<int>().ToArray();
                    var shape=new TensorShape(dimensions);Assert.That(shape.length,Is.EqualTo(2520*768));
                    var values=new float[shape.length];var selectedInput=Node(name);Assert.That(selectedInput.Length,Is.EqualTo(768));Array.Copy(selectedInput,0,values,173*768,768);
                    var final=(scaled??normalized).outputs[0];var slice=ModelTensorSlice.Extract(working,final,"normalized",new[]{new Model.Input{name="values",index=square.inputs[0],dataType=DataType.Float,shape=new DynamicTensorShape(shape)}});
                    var saved=Path.Combine(folder,"rms-"+i+".sentis");TextModelFile.Save(slice,saved);row["model_sha256"]=Hash(saved);
                    report["phase"]="execute_rms_"+i;Save();
                    using(var pipeline=new GpuProjectionPipeline(new[]{slice},Array.Empty<string>(),Array.Empty<string>(),Array.Empty<IGpuTensorOperation>()))
                    using(var input=new Tensor<float>(shape,values))
                    {
                        ComputeTensorData.Pin(input);using var result=pipeline.Execute(input);Assert.That(result.backendType,Is.EqualTo(BackendType.GPUCompute));
                        var actual=result.DownloadToArray().Skip(173*768).Take(768).ToArray();var expected=Node(spec.Value<string>("output_node"));
                        row["metrics"]=Metrics(actual,expected);row["output"]=Capture("rms-"+i+"-row173",actual);
                    }
                    Save();EditorApplication.QueuePlayerLoopUpdate();yield return null;
                }
                var gelu=context["gelu"];var geluValues=Node(gelu.Value<string>("input_node"));var geluExpected=Node(gelu.Value<string>("output_node"));
                var geluShape=new TensorShape(1,1,3072);Assert.That(geluValues.Length,Is.EqualTo(geluShape.length));report["phase"]="execute_gelu";Save();
                using(var operation=new GpuGelu(geluShape,true,true,true))
                using(var input=new Tensor<float>(geluShape,geluValues))
                {
                    ComputeTensorData.Pin(input);using var result=operation.Execute(input);Assert.That(result.backendType,Is.EqualTo(BackendType.GPUCompute));
                    var actual=result.DownloadToArray();report["gelu"]=Metrics(actual,geluExpected);report["gelu_output"]=Capture("gelu-row173",actual);
                }
                Assert.That(ModelCoefficientHash.Compute(original,out _),Is.EqualTo(coefficients));report["original_coefficient_bytes_preserved"]=true;
                report["actual_backend"]="GPUCompute";report["cpu_fallback_count"]=0;report["intermediate_readback_count"]=0;report["phase"]="completed";Save();
                Assert.That(rows.All(row=>row["metrics"].Value<int>("bit_differences")==0)&&report["gelu"].Value<int>("bit_differences")==0,Is.True,"Keep every qualified selected-row RMS/GELU bit.");
                report["success"]=true;
            }
            finally{Save();}
        }
        static JObject Metrics(float[] actual,float[] expected)
        {
            Assert.That(actual.Length,Is.EqualTo(expected.Length));int bits=0;double maximum=0;var witnesses=new JArray();
            for(var i=0;i<actual.Length;i++)
            {
                Assert.That(float.IsNaN(actual[i])||float.IsInfinity(actual[i]),Is.False);
                if(BitConverter.SingleToInt32Bits(actual[i])!=BitConverter.SingleToInt32Bits(expected[i]))
                {
                    bits++;if(witnesses.Count<16)witnesses.Add(new JObject{["index"]=i,["actual"]=actual[i],["expected"]=expected[i]});
                }
                maximum=Math.Max(maximum,Math.Abs((double)actual[i]-expected[i]));
            }
            return new JObject{["bit_differences"]=bits,["maximum_absolute_error"]=maximum,["witnesses"]=witnesses};
        }
        static float[] Floats(string path){var bytes=File.ReadAllBytes(path);var values=new float[bytes.Length/4];Buffer.BlockCopy(bytes,0,values,0,bytes.Length);return values;}
    }
}
