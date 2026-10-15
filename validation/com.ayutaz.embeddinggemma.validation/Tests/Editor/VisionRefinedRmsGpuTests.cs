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
    public sealed class VisionRefinedRmsGpuTests
    {
        [Test]
        public void RefinedFirstVisionRmsMatchesIndependentCurrentCpuEquationOnGpu()
        {
            Assert.That(SystemInfo.supportsComputeShaders,Is.True,"Actual GPU required; never skip.");
            var source=Environment.GetEnvironmentVariable("EG_ACCEPTANCE_SOURCE");var root=Path.Combine(Application.dataPath,"GpuVisionFirstRmsCandidate");
            var output=Path.GetFullPath(Path.Combine(Application.dataPath,"../artifacts/vision-first-rms-refined-gpu-"+source+".json"));
            Assert.That(File.Exists(output),Is.False);
            var report=new JObject{["success"]=false,["scope"]="first_vision_rms_gpu_against_current_cpu_equation_only",["runtime_source"]=source,
                ["original_internal_native_reference_available"]=false,["full_encoder_validated"]=false,["end_to_end_validated"]=false,["phase"]="audit"};
            void Save(){Directory.CreateDirectory(Path.GetDirectoryName(output));File.WriteAllText(output,report.ToString());}
            void Check(string path,string sha){using var stream=File.OpenRead(path);Assert.That(PlayerFileHash.Compute(stream,out _),Is.EqualTo(sha));}
            Save();
            try
            {
                var locked=JObject.Parse(File.ReadAllText(Path.Combine(Application.dataPath,"../Packages/packages-lock.json")));
                foreach(var package in new[]{"com.ayutaz.embeddinggemma","com.ayutaz.embeddinggemma.validation"})Assert.That(locked["dependencies"][package].Value<string>("hash"),Is.EqualTo(source));
                Check(Path.Combine(root,"first-rms.onnx"),"724605434efc0b70bb28b59b5dc639f8b0aeb765a4170ec782397506f6549976");
                Check(Path.Combine(root,"scale-fp32.bin"),"acc544061ccfe25009301bdfc15bab61468511ef6f27962b70c54efb65da9cb7");
                Check(Path.Combine(root,"current-cpu-equation-fp32.bin"),"881b26f2dba59d783185646334a9d50fe9fe1e3c78315d1a3644b6a09313db4d");
                var inputPath=Path.GetFullPath(Path.Combine(Application.dataPath,"../../k/artifacts/vision-diagnostics/native-stage-0.bin"));
                Check(inputPath,"bc4c19c9aaf9688801c99ed073fdde22f0daf6b822dc2c9c4b7a3f7c9aded636");
                var model=ModelLoader.Load(AssetDatabase.LoadAssetAtPath<ModelAsset>("Assets/GpuVisionFirstRmsCandidate/first-rms.onnx"));report["layer_count"]=model.layers.Count;
                using var beforeBytes=new MemoryStream();ModelWriter.Save(beforeBytes,model);
                using var operation=new QualifiedVisionRmsOperation(model);
                report["mean_division_gpu_boundary_preserved"]=true;report["cpu_computation_or_input_count"]=0;report["readback_flush_count"]=0;
                using var input=new Tensor<float>(new TensorShape(1,2520,768),Floats(inputPath));ComputeTensorData.Pin(input);report["phase"]="execute";Save();
                using var tensor=operation.Execute(input);Assert.That(tensor.backendType,Is.EqualTo(BackendType.GPUCompute));
                var actual=tensor.DownloadToArray();var expected=Floats(Path.Combine(root,"current-cpu-equation-fp32.bin"));Assert.That(actual.Length,Is.EqualTo(expected.Length));
                var path=Path.ChangeExtension(output,"bin");var bytes=new byte[actual.Length*4];Buffer.BlockCopy(actual,0,bytes,0,bytes.Length);File.WriteAllBytes(path,bytes);
                using(var stream=File.OpenRead(path))report["capture_sha256"]=PlayerFileHash.Compute(stream,out _);
                int bits=0,bad=0;double maximum=0,ratio=0;
                for(var i=0;i<actual.Length;i++)
                {
                    Assert.That(float.IsNaN(actual[i])||float.IsInfinity(actual[i]),Is.False);
                    if(BitConverter.SingleToInt32Bits(actual[i])!=BitConverter.SingleToInt32Bits(expected[i]))bits++;
                    var d=Math.Abs((double)actual[i]-expected[i]);var allowed=1e-5+1e-4*Math.Abs((double)expected[i]);if(d>allowed)bad++;maximum=Math.Max(maximum,d);ratio=Math.Max(ratio,d/allowed);
                }
                report["actual_backend"]="GPUCompute";report["bit_differences"]=bits;report["mismatched_elements"]=bad;report["maximum_absolute_error"]=maximum;report["maximum_tolerance_ratio"]=ratio;report["phase"]="completed";Save();
                using var afterBytes=new MemoryStream();ModelWriter.Save(afterBytes,model);Assert.That(afterBytes.ToArray(),Is.EqualTo(beforeBytes.ToArray()));
                report["source_model_bytes_preserved"]=true;
                Assert.That(bits,Is.Zero,"Qualify the exported RMS against its independent current CPU equation before encoder integration.");report["success"]=true;
            }
            catch(Exception error){report["error"]=error.ToString();throw;}
            finally{Save();}
        }
        static float[] Floats(string path){var bytes=File.ReadAllBytes(path);var data=new float[bytes.Length/4];Buffer.BlockCopy(bytes,0,data,0,bytes.Length);return data;}
    }
}
