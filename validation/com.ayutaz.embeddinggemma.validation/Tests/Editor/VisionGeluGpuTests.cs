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
