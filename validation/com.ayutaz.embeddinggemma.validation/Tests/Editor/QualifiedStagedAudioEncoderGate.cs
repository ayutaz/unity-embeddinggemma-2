using System;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Threading.Tasks;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using Unity.InferenceEngine;

namespace EmbeddingGemma.Validation.Tests
{
    // Separate encoder-only strict numerical gate. Its final encoder output
    // readback is not part of the PCM/final768 product API path.
    public static class QualifiedStagedAudioEncoderGate
    {
        public static async Task Run(PartitionedAudioFeatureEncoder encoder,string root,JObject reference,JObject audit,JObject report,string output,Action save)
        {
            Assert.That(QualifiedImageBackboneSquare.Hash(Path.Combine(root,"audio-encoder-reference.json")),Is.EqualTo("ac23f39904b8149dd5ccb6485afcc93d96047f0977b5faf1d640c8bd018c13c9"));
            Assert.That(QualifiedImageBackboneSquare.Hash(Path.Combine(root,"audio-encoder-validation.json")),Is.EqualTo("68d29c96012ef0a44114f2d3edbbff8d14e116ac697abee1a36d0b212155cfa1"));
            var frames=encoder.Profile.ValidFrameCount;var profile="frames-1119-valid-"+frames;
            var originalCases=((JArray)reference["cases"]).Where(row=>row.Value<string>("profile")==profile).ToArray();
            Assert.That(originalCases.Length,Is.EqualTo(frames==839?1:5));
            report["scope"]="accepted_staged_gpu_encoder_original_strict_elementwise_gate";report["profile"]=profile;report["cases"]=new JArray();
            report["actual_pcm_preprocessing_validated"]=false;report["cpu_computation_or_input_count"]=0;report["readback_flush_count"]=0;report["intermediate_readback_count"]=0;
            byte[] Read(string name)
            {
                var path=Path.Combine(root,name);var spec=audit["files"][name];Assert.That(spec,Is.Not.Null);
                Assert.That(new FileInfo(path).Length,Is.EqualTo(spec.Value<long>("bytes")));
                Assert.That(QualifiedImageBackboneSquare.Hash(path),Is.EqualTo(spec.Value<string>("sha256")));return File.ReadAllBytes(path);
            }
            var maskBytes=Read(reference["profiles"][profile].Value<string>("frame_mask"));var mask=new int[maskBytes.Length/4];Buffer.BlockCopy(maskBytes,0,mask,0,maskBytes.Length);
            Assert.That(mask,Is.EqualTo(Enumerable.Repeat(1,frames).Concat(Enumerable.Repeat(0,1119-frames)).ToArray()));
            foreach(var row in originalCases)
            {
                var id=row.Value<string>("id");report["phase"]="execute_staged_encoder_"+id;save();
                var inputBytes=Read(row.Value<string>("input_features"));var inputValues=new float[inputBytes.Length/4];Buffer.BlockCopy(inputBytes,0,inputValues,0,inputBytes.Length);
                var expected=row["features"].ToObject<float[][]>().SelectMany(values=>values).ToArray();
                using var features=new Tensor<float>(new TensorShape(1,1119,128),inputValues);using var frameMask=new Tensor<int>(new TensorShape(1,1119),mask);
                ComputeTensorData.Pin(features);ComputeTensorData.Pin(frameMask);
                using var processed=(AudioPreprocessingResult)Activator.CreateInstance(typeof(AudioPreprocessingResult),BindingFlags.Instance|BindingFlags.NonPublic,null,new object[]{features,frameMask,frames},null);
                var timer=System.Diagnostics.Stopwatch.StartNew();using var result=await encoder.EncodeAsync(processed);
                Assert.That(result.ValidFrameCount,Is.EqualTo(frames));Assert.That(result.SoftTokens.backendType,Is.EqualTo(BackendType.GPUCompute));
                Assert.That(result.SoftTokens.shape,Is.EqualTo(new TensorShape(encoder.Profile.FeatureCount,512)));
                var actual=result.SoftTokens.DownloadToArray();var duration=timer.Elapsed.TotalMilliseconds;
                Assert.That(actual.Length,Is.EqualTo(expected.Length));int mismatches=0,bits=0;double maximum=0,ratio=0;
                for(int index=0;index<actual.Length;index++)
                {
                    Assert.That(float.IsNaN(actual[index])||float.IsInfinity(actual[index]),Is.False);
                    var difference=Math.Abs((double)actual[index]-expected[index]);var allowed=1e-5+1e-4*Math.Abs((double)expected[index]);
                    maximum=Math.Max(maximum,difference);ratio=Math.Max(ratio,difference/allowed);if(difference>allowed)mismatches++;
                    if(BitConverter.SingleToInt32Bits(actual[index])!=BitConverter.SingleToInt32Bits(expected[index]))bits++;
                }
                var capture=Path.Combine(Path.GetDirectoryName(output),Path.GetFileNameWithoutExtension(output)+"-"+id+".bin");Assert.That(File.Exists(capture),Is.False);
                var bytes=new byte[actual.Length*4];Buffer.BlockCopy(actual,0,bytes,0,bytes.Length);File.WriteAllBytes(capture,bytes);
                ((JArray)report["cases"]).Add(new JObject{["id"]=id,["profile"]=profile,["values"]=actual.Length,["mismatched_elements"]=mismatches,["bit_differences"]=bits,
                    ["maximum_absolute_error"]=maximum,["maximum_tolerance_ratio"]=ratio,["capture"]=Path.GetFileName(capture),["capture_sha256"]=QualifiedImageBackboneSquare.Hash(capture),
                    ["execute_and_final_encoder_download_ms"]=duration,["success"]=mismatches==0});save();
                Assert.That(mismatches,Is.Zero,"Original strict encoder tolerances remain atol1e-5 / rtol1e-4.");
            }
            report["success"]=true;report["phase"]="completed";report["original_strict_encoder_cases_passed"]=originalCases.Length;save();
        }
    }
}
