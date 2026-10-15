using System;
using System.IO;
using System.Linq;
using System.Reflection;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEngine;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class VisionOriginalPoolingGpuTests
    {
        [TestCase(false)] [TestCase(true)]
        public void OriginalGoldenContextPoolingMustMatchNativeBitsOnGpu(bool nativeOrder)
        {
            Assert.That(SystemInfo.supportsComputeShaders,Is.True,"Actual GPU required; no skip.");
            var source=Environment.GetEnvironmentVariable("EG_ACCEPTANCE_SOURCE");
            var output=Path.GetFullPath(Path.Combine(Application.dataPath,"../artifacts/vision-original-pooling-gpu-"+source+"-"+(nativeOrder?"block192":"baseline")+".json"));
            Assert.That(File.Exists(output),Is.False);
            var report=new JObject{["success"]=false,["runtime_source"]=source,["scope"]="original_golden_reproducing_last_encoder_output_to_scaled_pool_features_only",
                ["native_block192"]=nativeOrder,["full_encoder_validated"]=false,["end_to_end_validated"]=false,["phase"]="audit"};
            void Save(){Directory.CreateDirectory(Path.GetDirectoryName(output));File.WriteAllText(output,report.ToString());}
            void Check(string path,string expected){using var stream=File.OpenRead(path);Assert.That(PlayerFileHash.Compute(stream,out _),Is.EqualTo(expected));}
            Save();
            try
            {
                var packages=JObject.Parse(File.ReadAllText(Path.Combine(Application.dataPath,"../Packages/packages-lock.json")));
                foreach(var package in new[]{"com.ayutaz.embeddinggemma","com.ayutaz.embeddinggemma.validation"})
                    Assert.That(packages["dependencies"][package].Value<string>("hash"),Is.EqualTo(source));
                var modelPath=Path.GetFullPath(Path.Combine(Application.dataPath,"../../k/artifacts/vision-encoder/square-fp32.sentis"));
                Check(modelPath,"4ecd0106463e833588c7584a58ea4973fdbab1eb1a56389c2f1597d3e7df34e7");
                var original=TextModelFile.Load(modelPath);var constants=original.constants.ToDictionary(value=>value.index);
                var product=original.layers.Single(layer=>(layer.GetType().Name=="MatMul"||layer.GetType().Name=="MatMul2D")&&constants.ContainsKey(layer.inputs[0]));
                Assert.That(constants[product.inputs[0]].shape,Is.EqualTo(new TensorShape(1,256,2520)));
                var producer=original.layers.SelectMany(layer=>layer.outputs.Select(index=>(index,layer))).ToDictionary(row=>row.index,row=>row.layer);
                var preceding=producer[product.inputs[1]];
                while(new[]{"Reshape","Identity","Unsqueeze","Squeeze"}.Contains(preceding.GetType().Name))preceding=producer[preceding.inputs[0]];
                Assert.That(preceding.GetType().Name,Is.EqualTo("Where"));
                Assert.That(constants.ContainsKey(preceding.inputs[1]),Is.True,"Original padding must choose its saved zero constant.");
                var hidden=preceding.inputs[2];var final=original.layers.Last(layer=>layer.GetType().Name=="Square").inputs[0];
                var model=ModelTensorSlice.Extract(original,final,"pooled",new[]{new Model.Input{name="hidden",index=hidden,dataType=DataType.Float,shape=new DynamicTensorShape(1,2520,768)}});
                Assert.That(model.inputs.Count,Is.EqualTo(1));
                var originalRoot=Path.GetFullPath(Path.Combine(Application.dataPath,"../../numerical-acceptance-state/gold38716679261"));
                var inputPath=Path.Combine(originalRoot,"add_225-fp32.bin");var expectedPath=Path.Combine(originalRoot,"mul_372-fp32.bin");
                Check(inputPath,"1cd9bd4d278a8931ae75f068d4b44c876e8bfa1d56c366d0159db3341b3a4d32");
                Check(expectedPath,"d2cdd48b2e945fd0d0c6ec368d29e996be1fa7d03fcbbdc071714e830815b950");
                var values=Floats(inputPath);var expected=Floats(expectedPath);
                using var input=new Tensor<float>(new TensorShape(1,2520,768),values);ComputeTensorData.Pin(input);
                float[] actual;
                if(nativeOrder)
                {
                    var map=model.constants.ToDictionary(value=>value.index);
                    var pool=model.layers.Single(layer=>(layer.GetType().Name=="MatMul"||layer.GetType().Name=="MatMul2D")&&map.ContainsKey(layer.inputs[0]));
                    var before=ModelTensorSlice.Extract(model,pool.inputs[1],"pool_right");
                    var after=ModelTensorSlice.Extract(model,model.outputs[0].index,"pooled",new[]{new Model.Input{name="pool",index=pool.outputs[0],dataType=DataType.Float,shape=new DynamicTensorShape(1,256,768)}});
                    var weight=map[pool.inputs[0]];var raw=(ArraySegment<byte>)typeof(Constant).GetField("array",BindingFlags.Instance|BindingFlags.NonPublic).GetValue(weight);
                    var coefficients=new float[raw.Count/4];Buffer.BlockCopy(raw.Array,raw.Offset,coefficients,0,raw.Count);
                    using var left=new Tensor<float>(weight.shape,coefficients);ComputeTensorData.Pin(left);
                    using var beforePipeline=new GpuProjectionPipeline(new[]{before},Array.Empty<string>(),Array.Empty<string>(),Array.Empty<IGpuTensorOperation>());
                    using var afterPipeline=new GpuProjectionPipeline(new[]{after},Array.Empty<string>(),Array.Empty<string>(),Array.Empty<IGpuTensorOperation>());
                    using var operation=new GpuBatchedMatmul();using var right=beforePipeline.Execute(input);
                    using var pooled=operation.Execute(left,right);using var result=afterPipeline.Execute(pooled);
                    Assert.That(result.backendType,Is.EqualTo(BackendType.GPUCompute));actual=result.DownloadToArray();
                }
                else
                {
                    using var pipeline=new GpuProjectionPipeline(new[]{model},Array.Empty<string>(),Array.Empty<string>(),Array.Empty<IGpuTensorOperation>());
                    using var result=pipeline.Execute(input);Assert.That(result.backendType,Is.EqualTo(BackendType.GPUCompute));actual=result.DownloadToArray();
                }
                Assert.That(actual.Length,Is.EqualTo(expected.Length));
                var bytes=new byte[actual.Length*4];Buffer.BlockCopy(actual,0,bytes,0,bytes.Length);File.WriteAllBytes(Path.ChangeExtension(output,"bin"),bytes);
                using(var stream=new MemoryStream(bytes,false))report["capture_sha256"]=PlayerFileHash.Compute(stream,out _);
                int bits=0,bad=0;double maximum=0,ratio=0;
                for(var i=0;i<actual.Length;i++)
                {
                    Assert.That(float.IsNaN(actual[i])||float.IsInfinity(actual[i]),Is.False);
                    if(BitConverter.SingleToInt32Bits(actual[i])!=BitConverter.SingleToInt32Bits(expected[i]))bits++;
                    var delta=Math.Abs((double)actual[i]-expected[i]);var bound=1e-5+1e-4*Math.Abs((double)expected[i]);
                    if(delta>bound)bad++;maximum=Math.Max(maximum,delta);ratio=Math.Max(ratio,delta/bound);
                }
                Assert.That(input.DownloadToArray(),Is.EqualTo(values));report["borrowed_input_preserved"]=true;
                report["actual_backend"]="GPUCompute";report["cpu_fallback_count"]=0;report["intermediate_readback_count"]=0;
                report["bit_differences"]=bits;report["mismatched_elements"]=bad;report["maximum_absolute_error"]=maximum;report["maximum_tolerance_ratio"]=ratio;report["phase"]="completed";Save();
                Assert.That(bits,Is.Zero,"Keep the original native pooling feature bits.");report["success"]=true;
            }
            catch(Exception error){report["error"]=error.ToString();throw;}
            finally{Save();}
        }
        static float[] Floats(string path){var bytes=File.ReadAllBytes(path);var values=new float[bytes.Length/4];Buffer.BlockCopy(bytes,0,values,0,bytes.Length);return values;}
    }
}
