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
    public sealed class OriginalImpulseSubsampleGpuTests
    {
        [Test]
        public void OriginalImpulseSubsamplePrefixMustMatchAllBitsOnGpu()
            =>Execute(false);

        [Test]
        public void FirstNormFmaImpulseSubsamplePrefixMustMatchAllBitsOnGpu()
            =>Execute(true);

        [Test]
        public void ScalarMomentFmaImpulseSubsamplePrefixMustMatchAllBitsOnGpu()
            =>Execute(true,true);

        void Execute(bool firstNormFma,bool scalarMomentFma=false)
        {
            Assert.That(SystemInfo.supportsComputeShaders,Is.True,"Actual GPU required; no skip.");
            var source=Environment.GetEnvironmentVariable("EG_ACCEPTANCE_SOURCE");
            var root=Path.Combine(Application.dataPath,"GpuAudioPartitionedNativeGlu");
            var output=Path.GetFullPath(Path.Combine(Application.dataPath,"../artifacts/original-impulse-subsample-prefix-"+(scalarMomentFma?"scalar-moment-fma-":firstNormFma?"first-norm-fma-":"")+source+".json"));
            Assert.That(File.Exists(output),Is.False);
            var report=new JObject{["success"]=false,["scope"]="one_original_impulse_subsample_gpu_bit_gate_with_diagnostic_captures",
                ["runtime_source"]=source,["full_encoder_validated"]=false,["end_to_end_validated"]=false,
                ["signal"]="impulse",["profile"]="frames-1119-valid-1119",["phase"]="audit",
                ["intermediate_readbacks_for_diagnostics_only"]=true};
            void Save(){Directory.CreateDirectory(Path.GetDirectoryName(output));File.WriteAllText(output,report.ToString());}
            void Check(string path,string sha){using var stream=File.OpenRead(path);Assert.That(PlayerFileHash.Compute(stream,out _),Is.EqualTo(sha));}
            string Hash(Constant value)
            {
                var bytes=(ArraySegment<byte>)typeof(Constant).GetField("array",BindingFlags.NonPublic|BindingFlags.Instance).GetValue(value);
                using var stream=new MemoryStream(bytes.Array,bytes.Offset,bytes.Count,false);return PlayerFileHash.Compute(stream,out _);
            }
            Save();
            try
            {
                var locked=JObject.Parse(File.ReadAllText(Path.Combine(Application.dataPath,"../Packages/packages-lock.json")));
                foreach(var package in new[]{"com.ayutaz.embeddinggemma","com.ayutaz.embeddinggemma.validation"})
                    Assert.That(locked["dependencies"][package].Value<string>("hash"),Is.EqualTo(source));
                var segmentFile=scalarMomentFma?"segment-0-scalar-moment-fma.onnx":firstNormFma?"segment-0-first-norm-fma.onnx":"segment-0.onnx";
                var segmentSha=scalarMomentFma?"650929620cdb4974eec0ee41d1d21c6dfe376c6703cb9cf5091d261171dbd949":firstNormFma?"9bc00c121b589af7444e026532b4c01452a21b32b1b2b290e2377b55aa6fe19b":"05f0bb360fd691872ef632c9513c4a07e93cac6a92f30538564867be9b226f51";
                Check(Path.Combine(root,segmentFile),segmentSha);report["segment_sha256"]=segmentSha;
                var model=ModelLoader.Load(AssetDatabase.LoadAssetAtPath<ModelAsset>("Assets/GpuAudioPartitionedNativeGlu/"+segmentFile));
                model=CompensatedModel.CreateWithDynamicMatMuls(model,16);model=CompensatedModel.Create(model,64);
                var constants=model.constants.ToDictionary(c=>c.index);
                var map=AudioCoefficientLayerMap.Resolve(model,new JArray(
                    new JObject{["stage"]="layer0.norm",["coefficient_bytes"]=512,["coefficient_sha256"]="dbb0fc3431144ac4a6265e965a3bac5bb1c0ee9e87b266bc4cb13d2b7d4d8c24"},
                    new JObject{["stage"]="layer1.norm",["coefficient_bytes"]=128,["coefficient_sha256"]="7b744a9e296e2331f5553570fdb993412b8f760f1a000179f366554cc84b1434"}));
                var convs=model.layers.Where(l=>l.GetType().Name=="Conv"&&constants.TryGetValue(l.inputs[1],out var w)&&Hash(w)=="e4d4ee18dc86a9945d38c159a861784a2c3d9467d6e0be9c48a414e60e2b92fd").ToArray();
                Assert.That(convs.Length,Is.EqualTo(1));map["layer0.conv"]=convs[0].outputs.Single();
                Layer Producer(int index)=>model.layers.Single(l=>l.outputs.Contains(index));
                var scale=Producer(map["layer1.norm"]);var normalized=Producer(scale.inputs.Single(i=>!constants.ContainsKey(i)));
                Assert.That(normalized.GetType().Name,Is.EqualTo("Mul"));
                var centered=normalized.inputs.Select(Producer).Single(l=>l.GetType().Name=="Sub");
                var normInput=centered.inputs[0];var layout=Producer(normInput);
                for(var hop=0;hop<4&&(layout.GetType().Name=="Reshape"||layout.GetType().Name=="Identity");hop++){normInput=layout.inputs[0];layout=Producer(normInput);}
                Assert.That(layout.GetType().Name,Is.EqualTo("Transpose"));map["layer1.conv"]=layout.inputs[0];
                map["layer0.act"]=AudioCoefficientLayerMap.ResolveReluOutput(model,map["layer0.norm"]);
                map["layer1.act"]=AudioCoefficientLayerMap.ResolveReluOutput(model,map["layer1.norm"]);
                var inputFile=Path.GetFullPath(Path.Combine(Application.dataPath,"../../k/artifacts/audio-real-chain/audio-runtime-reference/frames-1119-valid-1119/impulse-features-fp32.bin"));
                Check(inputFile,"2202541551e06a429b222d31f3d495a2b051fca70191287c82c7f6a6d8803c82");
                var expectedFile=Path.GetFullPath(Path.Combine(Application.dataPath,"../../k/artifacts/audio-current-stage-context/impulse-stage-00-fp32.bin"));
                Check(expectedFile,"fbbc24afa5f1d8dc1bfd45319339e087538ce0b2d8d43239c6ae6db3b41a0176");var expected=Floats(expectedFile);
                var weightFile=Path.Combine(root,"weight-0.bin");Check(weightFile,"a59650e614c2746ee18bbb23c475f65677beeb9ffe49350854af2aad873052a1");
                using var projection=new GpuBlockedMatmul(1024,1024,Floats(weightFile));
                using var input=new Tensor<float>(new TensorShape(1,1119,128),Floats(inputFile));ComputeTensorData.Pin(input);
                float[] actual,projectionInput;
                report["phase"]="normal_prefix";Save();
                using(var worker=new Worker(model,BackendType.GPUCompute))
                {
                    var cpu=(HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallback",BindingFlags.NonPublic|BindingFlags.Instance).GetValue(worker);
                    var flush=(HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallbackShouldFlushGPU",BindingFlags.NonPublic|BindingFlags.Instance).GetValue(worker);
                    var count=cpu.Intersect(model.layers.SelectMany(l=>l.outputs).Concat(model.inputs.Select(i=>i.index))).Count();
                    Assert.That(count,Is.Zero);Assert.That(flush.Count,Is.Zero);
                    report["cpu_computation_or_input_count"]=count;report["readback_flush_count"]=flush.Count;
                    worker.Schedule(input);var hidden=(Tensor<float>)worker.PeekOutput();
                    using var encoded=projection.Execute(hidden);actual=encoded.DownloadToArray();projectionInput=hidden.DownloadToArray();
                    report["actual_backend"]=encoded.backendType.ToString();
                }
                report["phase"]="observe_prefix";Save();
                var observed=WorkerObservation.ObserveIndexes(model,BackendType.GPUCompute,input,new HashSet<int>(map.Values));
                Assert.That(observed.FinalMatchesBaselineBits,Is.True);Assert.That(observed.Snapshots.Count,Is.EqualTo(6));
                Assert.That(observed.Baseline.Select(BitConverter.SingleToInt32Bits),Is.EqualTo(projectionInput.Select(BitConverter.SingleToInt32Bits)));
                report["normal_prefix_final_matches_observed_bits"]=true;
                var captures=Path.ChangeExtension(output,null)+"-captures";Assert.That(Directory.Exists(captures),Is.False);Directory.CreateDirectory(captures);
                var files=new JObject();report["captures"]=files;
                void Capture(string name,float[] values,int[] shape)
                {
                    Assert.That(values.Any(value=>float.IsNaN(value)||float.IsInfinity(value)),Is.False);
                    var bytes=new byte[values.Length*4];Buffer.BlockCopy(values,0,bytes,0,bytes.Length);var path=Path.Combine(captures,name+"-fp32.bin");File.WriteAllBytes(path,bytes);
                    using var stream=File.OpenRead(path);files[name]=new JObject{["file"]=path,["sha256"]=PlayerFileHash.Compute(stream,out _),["shape"]=JArray.FromObject(shape)};
                }
                foreach(var entry in map)
                {
                    var snapshot=observed.Snapshots.Single(s=>s.TensorIndex==entry.Value);Assert.That(snapshot.Backend,Is.EqualTo(BackendType.GPUCompute));
                    Capture(entry.Key,snapshot.Values,Enumerable.Range(0,snapshot.Shape.rank).Select(i=>snapshot.Shape[i]).ToArray());
                }
                Capture("projection_input",projectionInput,new[]{1,280,1024});Capture("projection_output",actual,new[]{1,280,1024});
                var firstWeight=constants[convs[0].inputs[1]];var raw=(ArraySegment<byte>)typeof(Constant).GetField("array",BindingFlags.NonPublic|BindingFlags.Instance).GetValue(firstWeight);
                var coefficients=new float[raw.Count/4];Buffer.BlockCopy(raw.Array,raw.Offset,coefficients,0,raw.Count);Capture("layer0.conv.weight",coefficients,new[]{128,1,3,3});
                int bits=0,bad=0;double ratio=0;Assert.That(actual.Length,Is.EqualTo(expected.Length));
                for(var i=0;i<actual.Length;i++){if(BitConverter.SingleToInt32Bits(actual[i])!=BitConverter.SingleToInt32Bits(expected[i]))bits++;var d=Math.Abs((double)actual[i]-expected[i]);var allowed=1e-5+1e-4*Math.Abs((double)expected[i]);if(d>allowed)bad++;ratio=Math.Max(ratio,d/allowed);}
                report["bit_differences"]=bits;report["mismatched_elements"]=bad;report["maximum_tolerance_ratio"]=ratio;report["phase"]="captured";Save();
                Assert.That(bits,Is.Zero,"Keep the original subsample bits while locating the first upstream difference.");report["success"]=true;
            }
            catch(Exception error){report["error"]=error.ToString();throw;}
            finally{Save();}
        }
        static float[] Floats(string path){var bytes=File.ReadAllBytes(path);var values=new float[bytes.Length/4];Buffer.BlockCopy(bytes,0,values,0,bytes.Length);return values;}
    }
}
