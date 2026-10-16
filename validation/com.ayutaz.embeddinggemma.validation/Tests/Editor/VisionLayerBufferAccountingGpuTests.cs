using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.Profiling;
using UnityEngine.TestTools;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class VisionLayerBufferAccountingGpuTests
    {
        const BindingFlags Fields=BindingFlags.Instance|BindingFlags.NonPublic;
        [UnityTest]
        public IEnumerator AccountSingleLayerDisposalAcrossEditorUpdatesAndDiagnosticGc()
        {
            Assert.That(SystemInfo.supportsComputeShaders,Is.True,"Actual GPU required; no skip.");
            var source=Environment.GetEnvironmentVariable("EG_ACCEPTANCE_SOURCE");
            var output=Path.GetFullPath(Path.Combine(Application.dataPath,"../artifacts/vision-layer-disposal-timing-"+source+".json"));Assert.That(File.Exists(output),Is.False);
            var report=new JObject{["success"]=false,["runtime_source"]=source,["scope"]="single_layer_disposal_timing_diagnostic_only",
                ["full_encoder_validated"]=false,["end_to_end_validated"]=false,["forced_gc_and_unload_are_diagnostic_only"]=true};
            var snapshots=new JArray();report["snapshots"]=snapshots;
            void Save(){Directory.CreateDirectory(Path.GetDirectoryName(output));File.WriteAllText(output,report.ToString());}
            string Hash(string path){using var stream=File.OpenRead(path);return PlayerFileHash.Compute(stream,out _);}
            using var process=Process.GetCurrentProcess();
            void Sample(string phase)
            {
                GpuExecutionProfiler.ReadProcessMemory(process,out var host,out var peak);
                snapshots.Add(new JObject{["phase"]=phase,["frame"]=Time.frameCount,["rendered_frame"]=Time.renderedFrameCount,["is_playing"]=Application.isPlaying,
                    ["graphics_driver_bytes"]=Profiler.GetAllocatedMemoryForGraphicsDriver(),["engine_allocated_bytes"]=Profiler.GetTotalAllocatedMemoryLong(),
                    ["managed_gc_bytes"]=GC.GetTotalMemory(false),["host_working_set_bytes"]=host,["host_peak_working_set_bytes"]=peak});Save();
            }
            Save();
            var locked=JObject.Parse(File.ReadAllText(Path.Combine(Application.dataPath,"../Packages/packages-lock.json")));
            foreach(var package in new[]{"com.ayutaz.embeddinggemma","com.ayutaz.embeddinggemma.validation"})Assert.That(locked["dependencies"][package].Value<string>("hash"),Is.EqualTo(source));
            var path=Path.GetFullPath(Path.Combine(Application.dataPath,"../artifacts/vision-layer0-original-patch-gpu-e39bdf1b5d0faf2ea424023d83076315736c6fde-captures/layer0.sentis"));
            var inputPath=Path.GetFullPath(Path.Combine(Application.dataPath,"../../k/artifacts/vision-diagnostics/native-stage-0.bin"));
            Assert.That(Hash(path),Is.EqualTo("d09b8884fae82ecd0245967774a0c8671efe09156c50367c45b9f133fc53983d"));
            Assert.That(Hash(inputPath),Is.EqualTo("bc4c19c9aaf9688801c99ed073fdde22f0daf6b822dc2c9c4b7a3f7c9aded636"));
            Sample("before_owned_scope");
            void RunOwnedLayer()
            {
                var model=TextModelFile.Load(path);using var input=new Tensor<float>(new TensorShape(1,2520,768),Floats(inputPath));ComputeTensorData.Pin(input);
                using var worker=new Worker(model,BackendType.GPUCompute);var cpu=(HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallback",Fields).GetValue(worker);
                var flush=(HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallbackShouldFlushGPU",Fields).GetValue(worker);
                Assert.That(cpu.Intersect(model.layers.SelectMany(layer=>layer.outputs).Concat(model.inputs.Select(value=>value.index))).Count(),Is.Zero);Assert.That(flush.Count,Is.Zero);
                worker.Schedule(input);var actual=((Tensor<float>)worker.PeekOutput()).DownloadToArray();var bytes=new byte[actual.Length*4];Buffer.BlockCopy(actual,0,bytes,0,bytes.Length);
                using(var stream=new MemoryStream(bytes,false))Assert.That(PlayerFileHash.Compute(stream,out _),Is.EqualTo("bc417e7f0e3486a5ac0025878d58c2c0b612ea5dea99a53e6b8db0e6f93b1513"));
                report["before_disposal_account"]=Account(worker,process);Sample("gpu_complete_before_disposal");
            }
            RunOwnedLayer();Sample("immediately_after_disposal");
            for(var i=0;i<8;i++){yield return null;Sample("editor_update_"+i);}
            for(var i=0;i<8;i++){UnityEditor.EditorApplication.QueuePlayerLoopUpdate();yield return null;Sample("queued_player_loop_"+i);}
            GC.Collect();GC.WaitForPendingFinalizers();yield return null;Sample("after_diagnostic_gc");
            yield return Resources.UnloadUnusedAssets();Sample("after_diagnostic_unload_unused_assets");
            report["actual_backend"]="GPUCompute";report["cpu_computation_or_input_count"]=0;report["readback_flush_count"]=0;
            report["original_gpu_reference_hash_preserved"]=true;report["success"]=true;Save();
        }
        [TestCase(false)] [TestCase(true)]
        public void AccountOriginalLayerGpuBufferCapacityWithoutIntermediateReadbacks(bool inlineRms)
        {
            Assert.That(SystemInfo.supportsComputeShaders,Is.True,"Actual GPU required; no skip.");
            var source=Environment.GetEnvironmentVariable("EG_ACCEPTANCE_SOURCE");var label=inlineRms?"inline-rms":"original";
            var output=Path.GetFullPath(Path.Combine(Application.dataPath,"../artifacts/vision-layer-buffer-accounting-"+source+"-"+label+".json"));
            Assert.That(File.Exists(output),Is.False);var samples=Path.ChangeExtension(output,"jsonl");Assert.That(File.Exists(samples),Is.False);
            var report=new JObject{["success"]=false,["runtime_source"]=source,["scope"]="single_layer_buffer_accounting_diagnostic_only",
                ["inline_rms"]=inlineRms,["full_encoder_validated"]=false,["end_to_end_validated"]=false,["intermediate_tensor_readbacks"]=0,
                ["timing_and_peak_not_representative_of_normal_schedule"]=true};
            void Save(){Directory.CreateDirectory(Path.GetDirectoryName(output));File.WriteAllText(output,report.ToString());}
            string Hash(string path){using var stream=File.OpenRead(path);return PlayerFileHash.Compute(stream,out _);}
            void Check(string path,string expected)=>Assert.That(Hash(path),Is.EqualTo(expected));
            Save();
            try
            {
                var locked=JObject.Parse(File.ReadAllText(Path.Combine(Application.dataPath,"../Packages/packages-lock.json")));
                foreach(var package in new[]{"com.ayutaz.embeddinggemma","com.ayutaz.embeddinggemma.validation"})Assert.That(locked["dependencies"][package].Value<string>("hash"),Is.EqualTo(source));
                var path=Path.GetFullPath(Path.Combine(Application.dataPath,"../artifacts/vision-layer0-original-patch-gpu-e39bdf1b5d0faf2ea424023d83076315736c6fde-captures/layer0.sentis"));
                Check(path,"d09b8884fae82ecd0245967774a0c8671efe09156c50367c45b9f133fc53983d");
                var original=TextModelFile.Load(path);var originalHash=SerializedHash(original);var model=original;
                if(inlineRms)
                {
                    Check(Path.Combine(Application.dataPath,"GpuVisionInlineRmsCandidate/inline-rms.onnx"),"15c8ec537c8d40386e61b675c05aaaa65b296667c0d38c39dc7d12fe28a25ccb");
                    var prep=Path.Combine(Application.dataPath,"GpuVisionHeadRmsCandidates/head-rms-candidates.json");Check(prep,"bb1f2d3df3905551b34723badefd7f1dfdc23da08a9ab4c72545d5d1368ce229");
                    var entries=JObject.Parse(File.ReadAllText(prep));
                    Model Head(int index)
                    {
                        var row=entries["rows"].Single(value=>value.Value<int>("index")==index);var asset="GpuVisionHeadRmsCandidates/"+row["onnx"].Value<string>("file");
                        Check(Path.Combine(Application.dataPath,asset),row["onnx"].Value<string>("sha256"));return ModelLoader.Load(UnityEditor.AssetDatabase.LoadAssetAtPath<ModelAsset>("Assets/"+asset));
                    }
                    var template=ModelLoader.Load(UnityEditor.AssetDatabase.LoadAssetAtPath<ModelAsset>("Assets/GpuVisionInlineRmsCandidate/inline-rms.onnx"));
                    model=QualifiedVisionRmsOperation.InlineLayerRms(original,template,Head(1),Head(3),1);
                }
                var inputPath=Path.GetFullPath(Path.Combine(Application.dataPath,"../../k/artifacts/vision-diagnostics/native-stage-0.bin"));
                Check(inputPath,"bc4c19c9aaf9688801c99ed073fdde22f0daf6b822dc2c9c4b7a3f7c9aded636");
                using var input=new Tensor<float>(new TensorShape(1,2520,768),Floats(inputPath));ComputeTensorData.Pin(input);
                using var worker=new Worker(model,BackendType.GPUCompute);
                var cpu=(HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallback",Fields).GetValue(worker);
                var flush=(HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallbackShouldFlushGPU",Fields).GetValue(worker);
                Assert.That(cpu.Intersect(model.layers.SelectMany(layer=>layer.outputs).Concat(model.inputs.Select(value=>value.index))).Count(),Is.Zero);Assert.That(flush.Count,Is.Zero);
                var yielded=model.layers.Where(layer=>!cpu.Contains(layer.outputs[0])).ToArray();var captured=new List<JObject>();
                using var process=Process.GetCurrentProcess();using var writer=new StreamWriter(samples,false);
                void Sample(int index,Layer layer)
                {
                    var row=Account(worker,process);row["step"]=index;row["layer"]=layer?.GetType().Name??"before_schedule";
                    if(layer!=null)row["output_indexes"]=JArray.FromObject(layer.outputs);
                    captured.Add(row);writer.WriteLine(row.ToString(Formatting.None));writer.Flush();
                }
                Sample(-1,null);var iterator=worker.ScheduleIterable(input);var step=0;
                try{while(iterator.MoveNext()){Sample(step,yielded[step]);step++;}}
                finally{(iterator as IDisposable)?.Dispose();}
                Assert.That(step,Is.EqualTo(yielded.Length));var observed=((Tensor<float>)worker.PeekOutput()).DownloadToArray();
                worker.Schedule(input);var baseline=((Tensor<float>)worker.PeekOutput()).DownloadToArray();
                Assert.That(observed.Select(BitConverter.SingleToInt32Bits),Is.EqualTo(baseline.Select(BitConverter.SingleToInt32Bits)));
                var bytes=new byte[observed.Length*4];Buffer.BlockCopy(observed,0,bytes,0,bytes.Length);
                using(var stream=new MemoryStream(bytes,false))report["capture_sha256"]=PlayerFileHash.Compute(stream,out _);
                if(!inlineRms)Assert.That(report.Value<string>("capture_sha256"),Is.EqualTo("bc417e7f0e3486a5ac0025878d58c2c0b612ea5dea99a53e6b8db0e6f93b1513"));
                Assert.That(SerializedHash(original),Is.EqualTo(originalHash));report["source_model_bytes_preserved"]=true;report["normal_schedule_matches_observed_bits"]=true;
                report["actual_backend"]="GPUCompute";report["cpu_computation_or_input_count"]=0;report["readback_flush_count"]=0;report["layer_count"]=model.layers.Count;
                report["maximum_accounted_gpu_capacity_bytes"]=captured.Max(row=>row.Value<long>("accounted_gpu_capacity_bytes"));
                report["maximum_graphics_driver_bytes"]=captured.Max(row=>row.Value<long>("graphics_driver_bytes"));
                report["maximum_host_working_set_bytes"]=captured.Max(row=>row.Value<long>("host_working_set_bytes"));
                report["largest_buffer_steps"]=new JArray(captured.OrderByDescending(row=>row.Value<long>("accounted_gpu_capacity_bytes")).Take(8));
                report["largest_driver_steps"]=new JArray(captured.OrderByDescending(row=>row.Value<long>("graphics_driver_bytes")).Take(8));
                report["initial"]=captured[0];report["last_step"]=captured[captured.Count-1];report["success"]=true;
            }
            catch(Exception error){report["error"]=error.ToString();throw;}
            finally{Save();}
        }
        static JObject Account(Worker worker,Process process)
        {
            var storage=typeof(Worker).GetField("m_Storage",Fields).GetValue(worker);var pool=storage.GetType().GetField("m_TensorPool",Fields).GetValue(storage);
            var backend=typeof(Worker).GetField("m_Backend",Fields).GetValue(worker);var type=typeof(Worker).Assembly.GetType("Unity.InferenceEngine.GPUComputeBackend");
            var seen=new HashSet<ComputeBuffer>();var cpu=new HashSet<ITensorData>();var categories=new JObject();long gpu=0,logical=0,cpuBytes=0;
            void Add(string category,IEnumerable data)
            {
                long count=0,bytes=0;
                foreach(var value in data)if(value is ITensorData item)
                {
                    if(item is ComputeTensorData compute&&compute.buffer!=null&&seen.Add(compute.buffer)){count++;bytes+=(long)compute.buffer.count*compute.buffer.stride;}
                    else if(item.backendType==BackendType.CPU&&cpu.Add(item))cpuBytes+=item.maxCapacity*4L;
                }
                gpu+=bytes;categories[category]=new JObject{["buffers"]=count,["capacity_bytes"]=bytes};
            }
            var tensors=((IDictionary)storage.GetType().GetField("m_InUseTensorsPool",Fields).GetValue(storage)).Values.Cast<object>().OfType<Tensor>().Where(value=>value.dataOnBackend!=null).ToArray();
            logical=tensors.Where(value=>value.backendType==BackendType.GPUCompute).Sum(value=>value.shape.length*4L);Add("storage_live",tensors.Select(value=>value.dataOnBackend).ToArray());
            void Free(string name,object owner,string field)
            {
                var ownerType=field=="m_MemoryPool"?type:owner.GetType();var cache=ownerType.GetField(field,Fields).GetValue(owner);
                Add(name,((IDictionary)cache.GetType().GetField("freeBuffers",Fields).GetValue(cache)).Values);
            }
            Free("storage_free",pool,"m_computeMemoryPool");Free("storage_cpu_free",pool,"m_cpuMemoryPool");Free("backend_free",backend,"m_MemoryPool");
            GpuExecutionProfiler.ReadProcessMemory(process,out var host,out var peak);
            return new JObject{["backend_type"]=backend.GetType().FullName,["accounted_gpu_capacity_bytes"]=gpu,["storage_gpu_logical_bytes"]=logical,["accounted_cpu_tensor_capacity_bytes"]=cpuBytes,
                ["categories"]=categories,["graphics_driver_bytes"]=Profiler.GetAllocatedMemoryForGraphicsDriver(),["engine_allocated_bytes"]=Profiler.GetTotalAllocatedMemoryLong(),
                ["engine_reserved_bytes"]=Profiler.GetTotalReservedMemoryLong(),["managed_gc_bytes"]=GC.GetTotalMemory(false),["host_working_set_bytes"]=host,["host_peak_working_set_bytes"]=peak};
        }
        static string SerializedHash(Model model){using var stream=new MemoryStream();ModelWriter.Save(stream,model);stream.Position=0;return PlayerFileHash.Compute(stream,out _);}
        static float[] Floats(string path){var bytes=File.ReadAllBytes(path);var values=new float[bytes.Length/4];Buffer.BlockCopy(bytes,0,values,0,bytes.Length);return values;}
    }
}
