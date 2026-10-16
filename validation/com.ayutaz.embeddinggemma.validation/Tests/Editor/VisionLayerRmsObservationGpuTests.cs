using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using EmbeddingGemma.Tests;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEngine;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class VisionLayerRmsObservationGpuTests
    {
        [Test]
        public void AllSevenLayerZeroRmsBoundariesPreserveNormalGpuOutput()
        {
            Assert.That(SystemInfo.supportsComputeShaders,Is.True,"Actual GPU required; never skip.");
            var source=Environment.GetEnvironmentVariable("EG_ACCEPTANCE_SOURCE");
            var output=Path.GetFullPath(Path.Combine(Application.dataPath,"../artifacts/vision-layer0-rms-observation-gpu-"+source+".json"));
            Assert.That(File.Exists(output),Is.False);
            var report=new JObject{["success"]=false,["runtime_source"]=source,["scope"]="original_layer0_all_rms_gpu_boundary_observation_only",
                ["native_reference_validated"]=false,["full_encoder_validated"]=false,["end_to_end_validated"]=false,["diagnostic_readbacks_only"]=true};
            void Save(){Directory.CreateDirectory(Path.GetDirectoryName(output));File.WriteAllText(output,report.ToString());}
            void Check(string path,string expected){using var stream=File.OpenRead(path);Assert.That(PlayerFileHash.Compute(stream,out _),Is.EqualTo(expected));}
            Save();
            try
            {
                var locked=JObject.Parse(File.ReadAllText(Path.Combine(Application.dataPath,"../Packages/packages-lock.json")));
                foreach(var package in new[]{"com.ayutaz.embeddinggemma","com.ayutaz.embeddinggemma.validation"})Assert.That(locked["dependencies"][package].Value<string>("hash"),Is.EqualTo(source));
                var path=Path.GetFullPath(Path.Combine(Application.dataPath,"../artifacts/vision-layer0-original-patch-gpu-e39bdf1b5d0faf2ea424023d83076315736c6fde-captures/layer0.sentis"));
                Check(path,"d09b8884fae82ecd0245967774a0c8671efe09156c50367c45b9f133fc53983d");
                var model=TextModelFile.Load(path);var constants=model.constants.ToDictionary(value=>value.index);
                Layer Next(Layer current,string kind)=>model.layers.Single(layer=>layer.GetType().Name==kind&&layer.inputs.Contains(current.outputs[0]));
                var norms=model.layers.Where(layer=>layer.GetType().Name=="Square").Select(square=>
                {
                    var mean=Next(square,"ReduceMean");var scalar=Next(mean,"ScalarMad");var inverse=Next(scalar,"Rsqrt");
                    Assert.That((float)scalar.GetType().GetField("sFloat").GetValue(scalar),Is.EqualTo(1f));
                    Assert.That((float)scalar.GetType().GetField("bFloat").GetValue(scalar),Is.EqualTo(1e-6f));
                    var multiply=Next(inverse,"Mul");Assert.That(multiply.inputs.Contains(square.inputs[0]),Is.True);
                    var scaled=model.layers.SingleOrDefault(layer=>layer.GetType().Name=="Mul"&&layer.inputs.Contains(multiply.outputs[0])
                        &&layer.inputs.Any(index=>constants.TryGetValue(index,out var constant)&&constant.dataType==DataType.Float&&constant.shape.rank==1));
                    return(input:square.inputs[0],output:(scaled??multiply).outputs[0],scale:scaled==null?null:constants[scaled.inputs.Single(constants.ContainsKey)]);
                }).ToArray();Assert.That(norms.Length,Is.EqualTo(7));
                using(var audit=new Worker(model,BackendType.GPUCompute))
                {
                    var cpu=(HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallback",BindingFlags.Instance|BindingFlags.NonPublic).GetValue(audit);
                    var flush=(HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallbackShouldFlushGPU",BindingFlags.Instance|BindingFlags.NonPublic).GetValue(audit);
                    Assert.That(cpu.Intersect(model.layers.SelectMany(layer=>layer.outputs).Concat(model.inputs.Select(value=>value.index))).Count(),Is.Zero);
                    Assert.That(flush.Count,Is.Zero);
                }
                var inputPath=Path.GetFullPath(Path.Combine(Application.dataPath,"../../k/artifacts/vision-diagnostics/native-stage-0.bin"));
                Check(inputPath,"bc4c19c9aaf9688801c99ed073fdde22f0daf6b822dc2c9c4b7a3f7c9aded636");
                var values=Floats(inputPath);using var input=new Tensor<float>(new TensorShape(1,2520,768),values);ComputeTensorData.Pin(input);
                var selected=new HashSet<int>(norms.SelectMany(norm=>new[]{norm.input,norm.output}).Where(index=>index!=model.inputs[0].index));
                var observed=WorkerObservation.ObserveIndexes(model,BackendType.GPUCompute,input,selected);
                Assert.That(observed.FinalMatchesBaselineBits,Is.True);Assert.That(observed.Snapshots.Select(value=>value.TensorIndex),Is.EquivalentTo(selected));
                var folder=Path.ChangeExtension(output,null)+"-captures";Assert.That(Directory.Exists(folder),Is.False);Directory.CreateDirectory(folder);
                JObject Write(string name,float[] data,TensorShape shape)
                {
                    var raw=new byte[data.Length*4];Buffer.BlockCopy(data,0,raw,0,raw.Length);var target=Path.Combine(folder,name+".bin");File.WriteAllBytes(target,raw);
                    using var stream=File.OpenRead(target);return new JObject{["file"]=target,["sha256"]=PlayerFileHash.Compute(stream,out _),
                        ["shape"]=JArray.FromObject(Enumerable.Range(0,shape.rank).Select(i=>shape[i]).ToArray())};
                }
                var final=Write("layer0-output",observed.Final,new TensorShape(1,2520,768));
                Assert.That(final.Value<string>("sha256"),Is.EqualTo("bc417e7f0e3486a5ac0025878d58c2c0b612ea5dea99a53e6b8db0e6f93b1513"));
                var rows=new JArray();report["norms"]=rows;
                for(var i=0;i<norms.Length;i++)
                {
                    var norm=norms[i];var incoming=norm.input==model.inputs[0].index?null:observed.Snapshots.Single(value=>value.TensorIndex==norm.input);
                    var outgoing=observed.Snapshots.Single(value=>value.TensorIndex==norm.output);
                    Assert.That(outgoing.Backend,Is.EqualTo(BackendType.GPUCompute));if(incoming!=null)Assert.That(incoming.Backend,Is.EqualTo(BackendType.GPUCompute));
                    var shape=incoming?.Shape??input.shape;var normModel=ModelTensorSlice.Extract(model,norm.output,"normalized",incoming==null?Array.Empty<Model.Input>():new[]{
                        new Model.Input{name="values",index=norm.input,dataType=DataType.Float,shape=new DynamicTensorShape(shape)}});
                    TextModelFile.Save(normModel,Path.Combine(folder,"rms-"+i+".sentis"));
                    var row=new JObject{["index"]=i,["input_index"]=norm.input,["output_index"]=norm.output,["epsilon"]=1e-6,
                        ["input"]=Write("rms-"+i+"-input",incoming?.Values??values,shape),["output"]=Write("rms-"+i+"-output",outgoing.Values,outgoing.Shape)};
                    if(norm.scale!=null)
                    {
                        var raw=(ArraySegment<byte>)typeof(Constant).GetField("array",BindingFlags.Instance|BindingFlags.NonPublic).GetValue(norm.scale);
                        var scale=new float[raw.Count/4];Buffer.BlockCopy(raw.Array,raw.Offset,scale,0,raw.Count);row["scale"]=Write("rms-"+i+"-scale",scale,norm.scale.shape);
                    }
                    rows.Add(row);
                }
                report["normal_final_matches_observed_bits"]=true;report["final"]=final;report["actual_backend"]="GPUCompute";
                report["cpu_computation_or_input_count"]=0;report["readback_flush_count"]=0;report["success"]=true;
            }
            catch(Exception error){report["error"]=error.ToString();throw;}
            finally{Save();}
        }
        static float[] Floats(string path){var bytes=File.ReadAllBytes(path);var data=new float[bytes.Length/4];Buffer.BlockCopy(bytes,0,data,0,bytes.Length);return data;}
    }
}
