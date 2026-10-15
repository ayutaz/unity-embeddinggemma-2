using System;
using System.IO;
using System.Linq;
using EmbeddingGemma.Tests;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEngine;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class VisionGeluGpuTests
    {
        [Test]
        public void CapturedOriginalLinuxPrefixGeluFusedInnerMustImproveNativeBitsOnGpu()
            =>RunNativePrefix(false);

        [Test]
        public void CapturedOriginalLinuxPrefixSleefGeluMustMatchNativeBitsOnGpu()
            =>RunNativePrefix(true);

        void RunNativePrefix(bool sleef)
        {
            var source=Environment.GetEnvironmentVariable("EG_ACCEPTANCE_SOURCE");
            var root=Path.GetFullPath(Path.Combine(Application.dataPath,"../../numerical-acceptance-state/np38719644488"));
            var label=sleef?"vision-original-prefix-sleef-gelu-gpu-":"vision-original-prefix-fused-gelu-gpu-";
            var output=Path.GetFullPath(Path.Combine(Application.dataPath,"../artifacts/"+label+source+".json"));Assert.That(File.Exists(output),Is.False);
            var report=new JObject{["success"]=false,["runtime_source"]=source,["scope"]="captured_linux_avx2_original_first_gelu_prefix_only",
                ["full_encoder_validated"]=false,["end_to_end_validated"]=false,["reference_reproduces_original_golden"]=false};
            void Save(){File.WriteAllText(output,report.ToString());}
            void Check(string path,string expected){using var stream=File.OpenRead(path);Assert.That(PlayerFileHash.Compute(stream,out _),Is.EqualTo(expected));}
            Save();
            try
            {
                var locked=JObject.Parse(File.ReadAllText(Path.Combine(Application.dataPath,"../Packages/packages-lock.json")));
                foreach(var package in new[]{"com.ayutaz.embeddinggemma","com.ayutaz.embeddinggemma.validation"})Assert.That(locked["dependencies"][package].Value<string>("hash"),Is.EqualTo(source));
                var path=Path.Combine(root,"original-image-native.json");Check(path,"b36c8af5e9e248d587eb2cb99f20407375f5656a0a173ad1dc0063e740863bd6");
                var reference=JObject.Parse(File.ReadAllText(path));Assert.That(reference["saved_node_capture"].Value<bool>("observed_final_matches_normal_bits"),Is.True);
                Assert.That(reference.Value<string>("graph_sha256"),Is.EqualTo("f9c5175a1d75289ce814a9651f496f9bdae5e873ac66199a5c3d03e0172c16cc"));
                var nodes=(JArray)reference["saved_node_capture"]["nodes"];
                string Node(string name)
                {
                    var row=nodes.Single(value=>value.Value<string>("node")==name);Assert.That(row.Value<int>("prefix_rows"),Is.EqualTo(16));
                    Assert.That(row["shape"].Values<int>().ToArray(),Is.EqualTo(new[]{1,16,3072}));
                    var result=Path.Combine(root,"captures",row.Value<string>("file"));Check(result,row.Value<string>("sha256"));return result;
                }
                var values=Floats(Node("view_21"));var expected=Floats(Node("gelu"));var shape=new TensorShape(1,16,3072);
                using var input=new Tensor<float>(shape,values);ComputeTensorData.Pin(input);
                using var separate=new GpuGelu(shape,true);using var fused=new GpuGelu(shape,true,true,sleef);
                if(sleef&&typeof(GpuGelu).GetField("pipeline",System.Reflection.BindingFlags.Instance|System.Reflection.BindingFlags.NonPublic).GetValue(fused)!=null)
                {
                    // Diagnostic readbacks only: locate the existing failed bit gate without
                    // adding intermediate transfers to the runtime operation.
                    const System.Reflection.BindingFlags fields=System.Reflection.BindingFlags.Instance|System.Reflection.BindingFlags.NonPublic;
                    var pipeline=(GpuProjectionPipeline)typeof(GpuGelu).GetField("pipeline",fields).GetValue(fused);
                    var models=(Model[])typeof(GpuProjectionPipeline).GetField("models",fields).GetValue(pipeline);
                    using var before=new GpuProjectionPipeline(new[]{models[0]},Array.Empty<string>(),Array.Empty<string>(),Array.Empty<IGpuTensorOperation>());
                    using var inner=before.Execute(input);using var activation=new GpuTanh(true);using var activated=activation.Execute(inner);
                    foreach(var stage in new[]{("inner",inner),("tanh",activated)})
                    {
                        var data=stage.Item2.DownloadToArray();var bytes=new byte[data.Length*4];Buffer.BlockCopy(data,0,bytes,0,bytes.Length);
                        var capture=Path.ChangeExtension(output,stage.Item1+".bin");File.WriteAllBytes(capture,bytes);
                        using var stream=new MemoryStream(bytes,false);report[stage.Item1+"_capture_sha256"]=PlayerFileHash.Compute(stream,out _);
                    }
                    report["diagnostic_stage_readbacks"]=2;
                }
                using var baseline=separate.Execute(input);using var candidate=fused.Execute(input);
                Assert.That(candidate.backendType,Is.EqualTo(BackendType.GPUCompute));var actual=candidate.DownloadToArray();
                report["separate"]=Metrics(baseline.DownloadToArray(),expected);report["fused"]=Metrics(actual,expected);
                var raw=new byte[actual.Length*4];Buffer.BlockCopy(actual,0,raw,0,raw.Length);File.WriteAllBytes(Path.ChangeExtension(output,"bin"),raw);
                using(var stream=new MemoryStream(raw,false))report["capture_sha256"]=PlayerFileHash.Compute(stream,out _);
                Assert.That(input.DownloadToArray(),Is.EqualTo(values));report["actual_backend"]="GPUCompute";report["cpu_computation_or_input_count"]=0;report["readback_flush_count"]=0;Save();
                Assert.That(report["fused"].Value<int>("mismatched_elements"),Is.Zero);
                if(sleef)Assert.That(report["fused"].Value<int>("bit_differences"),Is.Zero,"Keep captured native GELU bits.");
                Assert.That(report["fused"].Value<int>("bit_differences"),Is.LessThan(report["separate"].Value<int>("bit_differences")));
                report["success"]=true;
            }
            catch(Exception error){report["error"]=error.ToString();throw;}
            finally{Save();}
        }

        [Test]
        public void OriginalGateProjectionGeluMatchesCurrentCpuEquationOnGpu()
        {
            Assert.That(SystemInfo.supportsComputeShaders,Is.True,"Actual GPU required; no skip.");
            var type=typeof(IGpuTensorOperation).Assembly.GetType("EmbeddingGemma.GpuGelu");Assert.That(type,Is.Not.Null,"GPU GELU candidate is missing.");
            var source=Environment.GetEnvironmentVariable("EG_ACCEPTANCE_SOURCE");var root=Path.Combine(Application.dataPath,"GpuVisionGeluReference");
            var output=Path.GetFullPath(Path.Combine(Application.dataPath,"../artifacts/vision-gelu-gpu-"+source+".json"));Assert.That(File.Exists(output),Is.False);
            var report=new JObject{["success"]=false,["runtime_source"]=source,["scope"]="captured_gate_gelu_against_independent_current_cpu_equation_only",
                ["original_internal_native_reference_available"]=false,["full_encoder_validated"]=false,["end_to_end_validated"]=false};
            void Save(){Directory.CreateDirectory(Path.GetDirectoryName(output));File.WriteAllText(output,report.ToString());}
            void Check(string path,string expected){using var stream=File.OpenRead(path);Assert.That(PlayerFileHash.Compute(stream,out _),Is.EqualTo(expected));}
            Save();
            try
            {
                var locked=JObject.Parse(File.ReadAllText(Path.Combine(Application.dataPath,"../Packages/packages-lock.json")));
                foreach(var package in new[]{"com.ayutaz.embeddinggemma","com.ayutaz.embeddinggemma.validation"})Assert.That(locked["dependencies"][package].Value<string>("hash"),Is.EqualTo(source));
                var referencePath=Path.Combine(root,"gelu-reference.json");Check(referencePath,"e8dc7c1a82b3ac06e902554a67c9235db547141408c0067ce9136abfcaa9ceaa");
                var reference=JObject.Parse(File.ReadAllText(referencePath));
                foreach(var key in new[]{"input","original_layer_model"})Check(reference[key].Value<string>("file"),reference[key].Value<string>("sha256"));
                var expectedPath=Path.Combine(root,reference["expected"].Value<string>("file"));Check(expectedPath,reference["expected"].Value<string>("sha256"));
                var model=TextModelFile.Load(reference["original_layer_model"].Value<string>("file"));var constants=model.constants.ToDictionary(value=>value.index);
                var projections=model.layers.Where(layer=>layer.GetType().Name=="MatMul"&&constants.TryGetValue(layer.inputs[1],out var weight)&&weight.shape.rank==2).ToArray();
                var gelu=model.layers.Single(layer=>layer.GetType().Name=="GeluFast");var reshape=model.layers.Single(layer=>layer.outputs.Contains(gelu.inputs[0]));
                Assert.That(reshape.GetType().Name,Is.EqualTo("Reshape"));Assert.That(reshape.inputs[0],Is.EqualTo(projections[4].outputs[0]));
                var analysis=typeof(Worker).Assembly.GetType("Unity.InferenceEngine.Compiler.Analyser.PartialInferenceAnalysis");
                var context=analysis.GetMethod("InferModelPartialTensors").Invoke(null,new object[]{model});var partial=context.GetType().GetMethod("GetPartialTensor").Invoke(context,new object[]{gelu.inputs[0]});
                var inferred=(DynamicTensorShape)partial.GetType().GetProperty("shape").GetValue(partial);Assert.That(inferred.IsStatic(),Is.True);
                var shape=inferred.ToTensorShape();Assert.That(shape,Is.EqualTo(new TensorShape(1,2520,3072)));report["original_projection_reshape_preserved"]=true;
                var slice=ModelTensorSlice.Extract(model,gelu.outputs[0],"gelu",new[]{new Model.Input{name="input",index=gelu.inputs[0],dataType=DataType.Float,shape=new DynamicTensorShape(shape)}});
                using var baseline=new GpuProjectionPipeline(new[]{slice},Array.Empty<string>(),Array.Empty<string>(),Array.Empty<IGpuTensorOperation>());
                using var candidate=(IGpuTensorOperation)Activator.CreateInstance(type,new object[]{shape});
                using var input=new Tensor<float>(shape,Floats(reference["input"].Value<string>("file")));ComputeTensorData.Pin(input);
                using var previous=baseline.Execute(input);using var tensor=candidate.Execute(input);
                Assert.That(tensor.backendType,Is.EqualTo(BackendType.GPUCompute));Assert.That(tensor.shape,Is.EqualTo(shape));
                var expected=Floats(expectedPath);report["baseline"]=Metrics(previous.DownloadToArray(),expected);report["candidate"]=Metrics(tensor.DownloadToArray(),expected);
                report["actual_backend"]="GPUCompute";report["cpu_computation_or_input_count"]=0;report["readback_flush_count"]=0;Save();
                Assert.That(report["candidate"].Value<int>("mismatched_elements"),Is.Zero,"Keep original FP32 tolerances for the independent equation gate.");report["success"]=true;
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
        static float[] Floats(string path){var bytes=File.ReadAllBytes(path);var data=new float[bytes.Length/4];Buffer.BlockCopy(bytes,0,data,0,bytes.Length);return data;}
    }
}
