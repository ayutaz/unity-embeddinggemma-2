using System;
using System.IO;
using System.Linq;
using EmbeddingGemma.Tests;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using UnityEditor;
using Unity.InferenceEngine;
using UnityEngine;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class VisionRemainingRmsGpuTests
    {
        [TestCase(0)] [TestCase(4)] [TestCase(5)] [TestCase(6)]
        public void BorrowedOriginalScaleRmsMatchesIndependentCurrentCpuEquationOnGpu(int index)
        {
            Assert.That(SystemInfo.supportsComputeShaders,Is.True,"Actual GPU required; never skip.");
            var source=Environment.GetEnvironmentVariable("EG_ACCEPTANCE_SOURCE");var root=Path.Combine(Application.dataPath,"GpuVisionRemainingRmsReference");
            var output=Path.GetFullPath(Path.Combine(Application.dataPath,"../artifacts/vision-borrowed-rms-gpu-"+source+"-"+index+".json"));Assert.That(File.Exists(output),Is.False);
            var report=new JObject{["success"]=false,["scope"]="one_borrowed_width768_rms_against_captured_input_current_cpu_equation_only",["runtime_source"]=source,["index"]=index,
                ["original_internal_native_reference_available"]=false,["full_encoder_validated"]=false,["end_to_end_validated"]=false};
            void Save(){Directory.CreateDirectory(Path.GetDirectoryName(output));File.WriteAllText(output,report.ToString());}
            void Check(string path,string expected){using var stream=File.OpenRead(path);Assert.That(PlayerFileHash.Compute(stream,out _),Is.EqualTo(expected));}
            Save();
            try
            {
                var locked=JObject.Parse(File.ReadAllText(Path.Combine(Application.dataPath,"../Packages/packages-lock.json")));
                foreach(var package in new[]{"com.ayutaz.embeddinggemma","com.ayutaz.embeddinggemma.validation"})Assert.That(locked["dependencies"][package].Value<string>("hash"),Is.EqualTo(source));
                var referencePath=Path.Combine(root,"remaining-rms-reference.json");Check(referencePath,"6ba555bef760f18cfd1bd01c5ddcbf3dc552544fc49951050f504aef62b9a797");
                var reference=JObject.Parse(File.ReadAllText(referencePath));var row=((JArray)reference["rows"]).Single(value=>value.Value<int>("index")==index);
                var inputPath=row["input"].Value<string>("file");Check(inputPath,row["input"].Value<string>("sha256"));
                var modelPath=row["model"].Value<string>("file");Check(modelPath,row["model"].Value<string>("sha256"));
                var expectedPath=Path.Combine(root,row["expected"].Value<string>("file"));Check(expectedPath,row["expected"].Value<string>("sha256"));
                Check(row["scale"].Value<string>("file"),row["scale"].Value<string>("sha256"));
                Check(Path.Combine(Application.dataPath,"GpuVisionFirstRmsCandidate/first-rms.onnx"),"724605434efc0b70bb28b59b5dc639f8b0aeb765a4170ec782397506f6549976");
                var template=ModelLoader.Load(AssetDatabase.LoadAssetAtPath<ModelAsset>("Assets/GpuVisionFirstRmsCandidate/first-rms.onnx")));
                var original=TextModelFile.Load(modelPath);var before=SerializedHash(template);var originalBefore=SerializedHash(original);
                var scale=original.constants.Single(value=>value.dataType==DataType.Float&&value.shape==new TensorShape(768));
                var bound=QualifiedVisionRmsOperation.BorrowScale(template,scale);using var operation=new QualifiedVisionRmsOperation(bound);
                using var input=new Tensor<float>(new TensorShape(1,2520,768),Floats(inputPath));ComputeTensorData.Pin(input);
                using var tensor=operation.Execute(input);Assert.That(tensor.backendType,Is.EqualTo(BackendType.GPUCompute));Assert.That(tensor.shape,Is.EqualTo(input.shape));
                var actual=tensor.DownloadToArray();var expected=Floats(expectedPath);Assert.That(actual.Length,Is.EqualTo(expected.Length));Assert.That(Array.TrueForAll(actual,value=>!float.IsNaN(value)&&!float.IsInfinity(value)),Is.True);
                int bits=0,bad=0;double maximum=0,ratio=0;
                for(var i=0;i<actual.Length;i++)
                {
                    if(BitConverter.SingleToInt32Bits(actual[i])!=BitConverter.SingleToInt32Bits(expected[i]))bits++;
                    var d=Math.Abs((double)actual[i]-expected[i]);var allowed=1e-5+1e-4*Math.Abs((double)expected[i]);if(d>allowed)bad++;maximum=Math.Max(maximum,d);ratio=Math.Max(ratio,d/allowed);
                }
                Assert.That(SerializedHash(template),Is.EqualTo(before));Assert.That(SerializedHash(original),Is.EqualTo(originalBefore));
                report["template_and_coefficient_source_bytes_preserved"]=true;report["actual_backend"]="GPUCompute";report["cpu_computation_or_input_count"]=0;report["readback_flush_count"]=0;
                report["bit_differences"]=bits;report["mismatched_elements"]=bad;report["maximum_absolute_error"]=maximum;report["maximum_tolerance_ratio"]=ratio;Save();
                Assert.That(bits,Is.Zero,"Keep the independent current CPU equation strict gate.");report["success"]=true;
            }
            catch(Exception error){report["error"]=error.ToString();throw;}
            finally{Save();}
        }
        static string SerializedHash(Model model){using var stream=new MemoryStream();ModelWriter.Save(stream,model);stream.Position=0;return PlayerFileHash.Compute(stream,out _);}
        static float[] Floats(string path){var bytes=File.ReadAllBytes(path);var data=new float[bytes.Length/4];Buffer.BlockCopy(bytes,0,data,0,bytes.Length);return data;}
    }
}
