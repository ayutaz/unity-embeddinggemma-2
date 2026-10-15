using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEngine;

namespace EmbeddingGemma.Validation.Tests
{
    public static class QualifiedRealImageApiCase
    {
        sealed class ObservedPreprocessor : IImagePreprocessor
        {
            readonly IImagePreprocessor inner;public ObservedPreprocessor(IImagePreprocessor inner){this.inner=inner;}
            public BackendType Backend=>inner.Backend;public ImagePreprocessingResult Last;public int Calls;
            public ImagePreprocessingResult Process(Texture texture)
            {
                Calls++;Last=inner.Process(texture);Assert.That(Last.Pixels.backendType,Is.EqualTo(BackendType.GPUCompute));return Last;
            }
        }
        sealed class ObservedEncoder : IAsyncImageFeatureEncoder
        {
            readonly IAsyncImageFeatureEncoder inner;public ObservedEncoder(IAsyncImageFeatureEncoder inner){this.inner=inner;}
            public BackendType Backend=>inner.Backend;public ImageEncoderProfile Profile=>inner.Profile;
            public ImageEncodingResult Last;public int Calls;
            public ImageEncodingResult Encode(ImagePreprocessingResult input)=>throw new InvalidOperationException("Real API must use staged GPU encoding.");
            public async Task<ImageEncodingResult> EncodeAsync(ImagePreprocessingResult input,CancellationToken token=default)
            {
                Calls++;Last=await inner.EncodeAsync(input,token);Assert.That(Last.SoftTokens.backendType,Is.EqualTo(BackendType.GPUCompute));
                Assert.That(Last.SoftTokens.shape,Is.EqualTo(new TensorShape(Profile.FeatureCount,512)));return Last;
            }
        }
        public static async Task Run(PartitionedImageFeatureEncoder encoder,IImagePreprocessor preprocessing,Texture texture,
            string caseId,TextRole[] roles,JObject report,string reportPath,Action save,bool validateInputRecovery=false)
        {
            var featureCount=encoder.Profile.FeatureCount;Assert.That(featureCount==252||featureCount==256,Is.True);
            var root=Path.GetFullPath(Path.Combine(Application.dataPath,"../../numerical-acceptance-state/image-role-reference-real/image-role-reference"));
            var tokenizer=Path.GetFullPath(Path.Combine(Application.dataPath,"../../k/artifacts/multimodal-prompts"));
            void Check(string path,string expected)=>Assert.That(QualifiedImageBackboneSquare.Hash(path),Is.EqualTo(expected));
            Check(Path.Combine(tokenizer,"tokenizer.json"),"4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4");
            Check(Path.Combine(tokenizer,"tokenizer_config.json"),"17bd5d6e9364ca49a534e1502076593317c298d4a663623091ed45388f004874");
            var profile=new MultimodalBackboneProfile(1024,258880,258881,featureCount,0,262144,0);
            var builder=new MultimodalPromptBuilder(File.ReadAllText(Path.Combine(tokenizer,"tokenizer.json")),File.ReadAllText(Path.Combine(tokenizer,"tokenizer_config.json")),profile);
            report["phase"]="prepare_real_backbone";save();var model=featureCount==256?QualifiedImageBackboneSquare.Load():QualifiedImageBackboneSquare.LoadRectangular();
            var coefficients=ModelCoefficientHash.Compute(model,out _);
            Assert.That(coefficients,Is.EqualTo("c234c173d9a1f15fbe4bf7a3adbb7e2e5f17fdb8c6f4b85c0befe4d5fb2b097b"));
            using var backbone=new MultimodalBackbone(model,profile,BackendType.GPUCompute);
            const BindingFlags fields=BindingFlags.Instance|BindingFlags.NonPublic;
            var worker=(Worker)typeof(MultimodalBackbone).GetField("worker",fields).GetValue(backbone);
            var workerOwner=(GpuConstantWorker)typeof(MultimodalBackbone).GetField("workerOwner",fields).GetValue(backbone);
            report["backbone_constant_upload"]=new JObject{["uploaded_constant_count"]=workerOwner.UploadedConstantCount,
                ["upload_calls"]=workerOwner.UploadCallCount,["maximum_upload_bytes"]=workerOwner.MaximumUploadBytes,["skipped_cpu_shape_constants"]=workerOwner.SkippedCpuConstantCount};
            var cpu=(HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallback",fields).GetValue(worker);
            var flush=(HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallbackShouldFlushGPU",fields).GetValue(worker);
            var data=model.layers.SelectMany(layer=>layer.outputs).Concat(model.inputs.Where(input=>input.name=="image_features").Select(input=>input.index));
            Assert.That(cpu.Intersect(data),Is.Empty,"No backbone computation or nonempty feature input may fall back to CPU.");Assert.That(flush,Is.Empty);
            var pre=new ObservedPreprocessor(preprocessing);var enc=new ObservedEncoder(encoder);
            using var api=new ImageEmbedder(pre,enc,backbone,builder);var results=new JArray();report["final_api_cases"]=results;
            report["input_texture_kind"]=texture is RenderTexture?"RenderTexture":"Texture2D";
            if(validateInputRecovery)
            {
                var failures=new JArray();report["real_input_recovery_checks"]=failures;report["phase"]="reject_real_invalid_inputs";save();
                async Task Reject(Func<Task<float[]>> operation,Type expected,string name)
                {
                    Exception actual=null;try{await operation();}catch(Exception error){actual=error;}
                    Assert.That(actual,Is.Not.Null,name);Assert.That(expected.IsInstanceOfType(actual),Is.True,name+": "+actual);
                    failures.Add(new JObject{["name"]=name,["exception"]=actual.GetType().FullName,["success"]=true});save();
                }
                await Reject(()=>api.EmbedAsync(null),typeof(ArgumentNullException),"null_texture");
                await Reject(()=>api.EmbedAsync(texture,(TextRole)99),typeof(ArgumentOutOfRangeException),"invalid_role");
                var imageMarker=JObject.Parse(File.ReadAllText(Path.Combine(tokenizer,"tokenizer_config.json"))).Value<string>("image_token");
                Assert.That(string.IsNullOrEmpty(imageMarker),Is.False);
                await Reject(()=>api.EmbedAsync(texture,TextRole.Document,imageMarker),typeof(ArgumentException),"marker_in_title");
                var destroyed=new Texture2D(2,2);UnityEngine.Object.DestroyImmediate(destroyed);
                await Reject(()=>api.EmbedAsync(destroyed),typeof(ArgumentException),"destroyed_texture");
                Assert.That(pre.Calls,Is.Zero);Assert.That(enc.Calls,Is.Zero);
                var unsupported=new Texture2D(16,16,TextureFormat.RGBAFloat,false,true);
                try{await Reject(()=>api.EmbedAsync(unsupported),typeof(ArgumentException),"unsupported_float_storage");}
                finally{UnityEngine.Object.DestroyImmediate(unsupported);}
                var wrongGeometry=new Texture2D(160,96,TextureFormat.RGBA32,false,true);wrongGeometry.Apply(false,true);
                try{await Reject(()=>api.EmbedAsync(wrongGeometry),typeof(ArgumentException),"different_fixed_profile");}
                finally{UnityEngine.Object.DestroyImmediate(wrongGeometry);}
                Assert.That(pre.Calls,Is.EqualTo(2));Assert.That(enc.Calls,Is.EqualTo(1));
                Assert.Throws<ObjectDisposedException>(()=>{var ignored=pre.Last.Pixels;});
                pre.Calls=0;enc.Calls=0;report["real_invalid_input_owned_tensors_disposed"]=true;save();
                using(var cancellation=new CancellationTokenSource())
                {
                    report["phase"]="cancel_after_real_gpu_stage_yield";save();
                    var pending=api.EmbedAsync(texture,cancellationToken:cancellation.Token);
                    Assert.That(enc.Calls,Is.EqualTo(1));Assert.That(pending.IsCompleted,Is.False);
                    cancellation.Cancel();await Reject(()=>pending,typeof(OperationCanceledException),"cancel_after_real_gpu_stage_yield");
                    Assert.That(pending.IsCanceled,Is.True);Assert.Throws<ObjectDisposedException>(()=>{var ignored=pre.Last.Pixels;});
                }
                pre.Calls=0;enc.Calls=0;report["real_gpu_stage_cancellation_validated"]=true;save();
            }
            foreach(var role in roles)
            {
                var name=role.ToString().ToLowerInvariant();var referencePath=Path.Combine(root,name,"image-embedding-reference.json");
                var digest=role==TextRole.Raw?"120faf6aea190466b625f7c606a4c14e38315101a265c7dd457af13694512e9c":role==TextRole.Query?
                    "dc4ad0a8e43e391a251e7ff3fb4f569b6bddb7442852f8721b1b7657ea91f82d":"90692ad38c6d7d844dc6b6e79c7d5cfa719ea59866946dbf94a40bca10ac9eac";
                Check(referencePath,digest);var reference=JObject.Parse(File.ReadAllText(referencePath));
                Assert.That(reference["metadata"].Value<string>("model_revision"),Is.EqualTo("914f7f89142e33e77833254d9c9b90c3cef7303b"));
                Assert.That(reference["metadata"].Value<string>("source_commit"),Is.EqualTo("891ce9004deecc262437e56b2fd9a2eaf1054632"));
                var expectedRow=reference["cases"].Single(row=>row.Value<string>("id")==caseId);Assert.That(expectedRow.Value<string>("role"),Is.EqualTo(name));
                Assert.That(reference["profiles"][expectedRow.Value<string>("profile")].Value<int>("image_feature_count"),Is.EqualTo(featureCount));
                var pixels=Path.GetFullPath(Path.Combine(Application.dataPath,"../../k/artifacts/vision-encoder",expectedRow.Value<string>("pixels")));
                Check(pixels,expectedRow.Value<string>("pixels_sha256"));
                var prompt=builder.BuildImage(featureCount,role);Assert.That(prompt.InputIds,Is.EqualTo(expectedRow["input_ids"].Values<int>().ToArray()));
                Assert.That(prompt.AttentionMask,Is.EqualTo(expectedRow["attention_mask"].Values<int>().ToArray()));
                var metric=new JObject{["case_id"]=caseId,["role"]=name,["success"]=false,["reference_sha256"]=digest};results.Add(metric);
                report["phase"]="execute_final_api_"+name;save();var timer=Stopwatch.StartNew();
                var actual=await api.EmbedAsync(texture,role);metric["execute_and_final_download_ms"]=timer.Elapsed.TotalMilliseconds;
                Assert.That(worker.PeekOutput().backendType,Is.EqualTo(BackendType.GPUCompute));
                Assert.That(actual.Length,Is.EqualTo(768));var expected=expectedRow["embedding"].Values<float>().ToArray();Assert.That(expected.Length,Is.EqualTo(768));
                double dot=0,left=0,right=0,maximum=0,ratio=0;int bad=0;
                for(int i=0;i<actual.Length;i++)
                {
                    Assert.That(float.IsNaN(actual[i])||float.IsInfinity(actual[i]),Is.False);
                    dot+=(double)actual[i]*expected[i];left+=(double)actual[i]*actual[i];right+=(double)expected[i]*expected[i];
                    var error=Math.Abs((double)actual[i]-expected[i]);var bound=1e-5+1e-4*Math.Abs((double)expected[i]);
                    maximum=Math.Max(maximum,error);ratio=Math.Max(ratio,error/bound);if(error>bound)bad++;
                }
                var cosine=dot/Math.Sqrt(left*right);metric["cosine"]=cosine;metric["norm_squared"]=left;metric["maximum_absolute_error"]=maximum;
                metric["elementwise_mismatches"]=bad;metric["maximum_tolerance_ratio"]=ratio;metric["elementwise_metrics_are_diagnostic"]=true;
                metric["acceptance_cosine_minimum"]=0.999;metric["actual_backend"]="GPUCompute";
                var capture=Path.ChangeExtension(reportPath,null)+"-"+name+".bin";Assert.That(File.Exists(capture),Is.False);
                var bytes=new byte[actual.Length*4];Buffer.BlockCopy(actual,0,bytes,0,bytes.Length);File.WriteAllBytes(capture,bytes);
                metric["capture_file"]=Path.GetFileName(capture);metric["capture_sha256"]=QualifiedImageBackboneSquare.Hash(capture);save();
                Assert.That(cosine,Is.GreaterThanOrEqualTo(.999));Assert.That(Math.Abs(left-1),Is.LessThanOrEqualTo(.002));
                Assert.Throws<ObjectDisposedException>(()=>{var ignored=pre.Last.Pixels;});Assert.Throws<ObjectDisposedException>(()=>{var ignored=enc.Last.SoftTokens;});
                Assert.That(texture==null,Is.False,"Input texture is borrowed.");metric["owned_intermediates_disposed"]=true;metric["success"]=true;save();
            }
            Assert.That(pre.Calls,Is.EqualTo(roles.Length));Assert.That(enc.Calls,Is.EqualTo(roles.Length));
            Assert.That(ModelCoefficientHash.Compute(model,out _),Is.EqualTo(coefficients));report["backbone_coefficient_bytes_preserved"]=true;
            report["backbone_original_coefficients_sha256"]=coefficients;
            using var process=Process.GetCurrentProcess();GpuExecutionProfiler.ReadProcessMemory(process,out var current,out var peak);
            report["host_working_set_after_final_bytes"]=current;report["host_peak_working_set_bytes"]=peak;
            report["actual_backend"]="GPUCompute";report["cpu_computation_or_input_count"]=0;report["readback_flush_count"]=0;report["intermediate_readback_count"]=0;
            report["actual_texture_to_final_768_for_recorded_cases"]=true;report["all_image_cases_validated"]=false;report["end_to_end_validated"]=false;
        }
    }
}
