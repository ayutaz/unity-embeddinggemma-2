using System;
using System.IO;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEngine;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class VisionTail252GpuTests
    {
        [Test]
        public void Prepared252TailRmsMustMatchCapturedNativeRowsOnGpu()
        {
            var source=Environment.GetEnvironmentVariable("EG_ACCEPTANCE_SOURCE");
            var root=Path.GetFullPath(Path.Combine(Application.dataPath,"../../numerical-acceptance-state/t252"));
            var output=Path.GetFullPath(Path.Combine(Application.dataPath,"../artifacts/vision-tail-252-gpu-"+source+".json"));
            Assert.That(File.Exists(output),Is.False);
            var report=new JObject{["success"]=false,["runtime_source"]=source,
                ["scope"]="252_row_rms_on_first_252_rows_of_captured_square_native_tail_only",
                ["landscape_or_portrait_encoder_validated"]=false,["full_encoder_validated"]=false,["end_to_end_validated"]=false};
            void Save(){Directory.CreateDirectory(Path.GetDirectoryName(output));File.WriteAllText(output,report.ToString());}
            string Hash(string path){using var stream=File.OpenRead(path);return PlayerFileHash.Compute(stream,out _);}
            void Check(string path,string sha)=>Assert.That(Hash(path),Is.EqualTo(sha));
            Save();
            try
            {
                var locked=JObject.Parse(File.ReadAllText(Path.Combine(Application.dataPath,"../Packages/packages-lock.json")));
                foreach(var package in new[]{"com.ayutaz.embeddinggemma","com.ayutaz.embeddinggemma.validation"})Assert.That(locked["dependencies"][package].Value<string>("hash"),Is.EqualTo(source));
                var referencePath=Path.Combine(root,"reference.json");Check(referencePath,"48de96448ff5f4e460bd7f412b442256d6f77f4701eb01ddc93aca20b865b993");
                var reference=JObject.Parse(File.ReadAllText(referencePath));
                Check(Path.Combine(root,"input.bin"),reference.Value<string>("input_sha256"));Check(Path.Combine(root,"expected.bin"),reference.Value<string>("expected_sha256"));
                var asset="Assets/GpuVisionTail252Candidate/tail-rms-252.onnx";
                Check(Path.GetFullPath(Path.Combine(Application.dataPath,"../"+asset)),reference.Value<string>("onnx_sha256"));
                var model=ModelLoader.Load(UnityEditor.AssetDatabase.LoadAssetAtPath<ModelAsset>(asset));
                Assert.That(model.inputs[0].shape.ToTensorShape(),Is.EqualTo(new TensorShape(252,768)));
                var saved=Path.ChangeExtension(output,"sentis");TextModelFile.Save(model,saved);report["saved_template_sha256"]=Hash(saved);report["saved_template_file"]=Path.GetFileName(saved);
                var values=Floats(Path.Combine(root,"input.bin"));var expected=Floats(Path.Combine(root,"expected.bin"));
                using var pipeline=new GpuProjectionPipeline(new[]{model},Array.Empty<string>(),Array.Empty<string>(),Array.Empty<IGpuTensorOperation>());
                using var input=new Tensor<float>(new TensorShape(252,768),values);ComputeTensorData.Pin(input);
                using var result=pipeline.Execute(input);var actual=result.DownloadToArray();var bits=0;
                Assert.That(result.backendType,Is.EqualTo(BackendType.GPUCompute));Assert.That(result.shape,Is.EqualTo(input.shape));
                for(var i=0;i<actual.Length;i++)if(BitConverter.SingleToInt32Bits(actual[i])!=BitConverter.SingleToInt32Bits(expected[i]))bits++;
                report["bit_differences"]=bits;report["value_count"]=actual.Length;report["actual_backend"]="GPUCompute";
                report["cpu_computation_or_input_count"]=0;report["readback_flush_count"]=0;
                Assert.That(input.DownloadToArray(),Is.EqualTo(values));Save();
                Assert.That(bits,Is.Zero,"Preserve every bit of the independently captured native RMS rows.");report["success"]=true;
            }
            catch(Exception error){report["error"]=error.ToString();throw;}
            finally{Save();}
        }
        static float[] Floats(string path){var raw=File.ReadAllBytes(path);var data=new float[raw.Length/4];Buffer.BlockCopy(raw,0,data,0,raw.Length);return data;}
    }
}
