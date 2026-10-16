using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEngine;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class TextGpuRegressionReferenceTests
    {
        sealed class Recording : ITextEmbedder
        {
            readonly TextEmbedder inner;public readonly List<float[]> Documents=new List<float[]>();public float[] LastQuery;
            public Recording(TextEmbedder inner){this.inner=inner;}public BackendType Backend=>inner.Backend;
            public float[] EmbedDocument(string text,string title=null){var value=inner.EmbedDocument(text,title);Documents.Add(value);return value;}
            public float[] EmbedQuery(string text){LastQuery=inner.EmbedQuery(text);return LastQuery;}public void Dispose(){}
        }
        [TestCase("fp32")] [TestCase("float16")]
        public void OriginalText128DocumentsQueriesAndEveryRankingStillMatchOnGpu(string precision)
        {
            Assert.That(SystemInfo.supportsComputeShaders,Is.True,"Actual GPU required; never skip.");var source=Environment.GetEnvironmentVariable("EG_ACCEPTANCE_SOURCE");
            var directory=Path.GetFullPath(Path.Combine(Application.dataPath,"../../k/Assets/StreamingAssets/EmbeddingGemmaTextSearch"));
            var referencePath=Path.GetFullPath(Path.Combine(Application.dataPath,"../../k/artifacts/m1/search-reference.json"));
            var output=Path.GetFullPath(Path.Combine(Application.dataPath,"../artifacts/text128-regression-gpu-"+source+"-"+precision+".json"));Assert.That(File.Exists(output),Is.False);
            var report=new JObject{["success"]=false,["scope"]="original_text128_six_documents_four_queries_full_rankings_gpu_regression",
                ["runtime_source"]=source,["precision"]=precision,["phase"]="audit",["image_audio_accepted_by_this_test"]=false};
            void Save(){Directory.CreateDirectory(Path.GetDirectoryName(output));File.WriteAllText(output,report.ToString());}
            void Check(string path,string expected)=>Assert.That(QualifiedImageBackboneSquare.Hash(path),Is.EqualTo(expected));Save();
            try
            {
                var locked=JObject.Parse(File.ReadAllText(Path.Combine(Application.dataPath,"../Packages/packages-lock.json")));
                foreach(var package in new[]{"com.ayutaz.embeddinggemma","com.ayutaz.embeddinggemma.validation"})Assert.That(locked["dependencies"][package].Value<string>("hash"),Is.EqualTo(source));
                Check(referencePath,"3530756b36316a812ac853f0722bed5479e92e6bb02ca6aa41c06660d32a71ec");
                Check(Path.Combine(directory,"preparation.json"),"706e1e11cd817271e7c9391a7652614fbc2cb87a90443e4870acb13bb787712a");
                var prepared=JObject.Parse(File.ReadAllText(Path.Combine(directory,"preparation.json")));Assert.That(prepared.Value<bool>("success"),Is.True);
                var reference=JObject.Parse(File.ReadAllText(referencePath));Assert.That(reference["metadata"].Value<string>("model_revision"),Is.EqualTo("914f7f89142e33e77833254d9c9b90c3cef7303b"));
                Assert.That(prepared.Value<string>("searchSourceCommit"),Is.EqualTo(reference["metadata"].Value<string>("source_commit")));
                var modelName="model-"+precision+".sentis";
                foreach(var file in new[]{modelName,"tokenizer.json"})Check(Path.Combine(directory,file),prepared["files"].Single(row=>row.Value<string>("name")==file).Value<string>("sha256"));
                report["phase"]="prepare_gpu_worker";Save();var model=TextModelFile.Load(Path.Combine(directory,modelName));
                using var api=new TextEmbedder(model,File.ReadAllText(Path.Combine(directory,"tokenizer.json")),BackendType.GPUCompute);using var provider=new Recording(api);
                var worker=(Worker)typeof(TextEmbedder).GetField("worker",BindingFlags.Instance|BindingFlags.NonPublic).GetValue(api);
                var cpu=(HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallback",BindingFlags.Instance|BindingFlags.NonPublic).GetValue(worker);
                var flush=(HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallbackShouldFlushGPU",BindingFlags.Instance|BindingFlags.NonPublic).GetValue(worker);
                Assert.That(cpu.Intersect(model.layers.SelectMany(layer=>layer.outputs)),Is.Empty);Assert.That(flush,Is.Empty);
                var documents=reference["documents"].ToArray();var queries=reference["queries"].ToArray();Assert.That(documents.Length,Is.EqualTo(6));Assert.That(queries.Length,Is.EqualTo(4));
                var rows=new JArray();report["vectors"]=rows;double minimum=1;var captures=new List<float[]>();var threshold=precision=="fp32"?.999:.99;
                void Compare(float[] actual,JToken row,string role)
                {
                    Assert.That(worker.PeekOutput().backendType,Is.EqualTo(BackendType.GPUCompute));Assert.That(actual.Length,Is.EqualTo(768));
                    var cosine=TextSearchIndex.Cosine(actual,row["embedding"].Values<float>().ToArray());minimum=Math.Min(minimum,cosine);
                    Assert.That(cosine,Is.GreaterThanOrEqualTo(threshold));Assert.That(Math.Abs(actual.Sum(value=>(double)value*value)-1),Is.LessThanOrEqualTo(.002));
                    rows.Add(new JObject{["id"]=row.Value<string>("id"),["role"]=role,["cosine"]=cosine,["success"]=true});captures.Add(actual);
                }
                report["phase"]="embed_documents_and_search";Save();
                var index=new TextSearchIndex(documents.Select(row=>new SearchDocument(row.Value<string>("id"),row.Value<string>("text"),row.Value<string>("title"))),provider);
                for(int i=0;i<documents.Length;i++)Compare(provider.Documents[i],documents[i],"document");
                foreach(var query in queries)
                {
                    var hits=index.Search(query.Value<string>("text"));Compare(provider.LastQuery,query,"query");
                    Assert.That(hits.Select(hit=>hit.Document.Id),Is.EqualTo(query["ranking"].Select(row=>row.Value<string>("document_id"))));
                    for(int i=0;i<hits.Length;i++)Assert.That(hits[i].Score,Is.EqualTo(query["ranking"][i].Value<double>("score")).Within(precision=="fp32"?.002:.02));
                }
                Assert.That(provider.Documents.Count,Is.EqualTo(6));var values=captures.SelectMany(value=>value).ToArray();var bytes=new byte[values.Length*4];Buffer.BlockCopy(values,0,bytes,0,bytes.Length);
                File.WriteAllBytes(Path.ChangeExtension(output,"bin"),bytes);report["capture_sha256"]=QualifiedImageBackboneSquare.Hash(Path.ChangeExtension(output,"bin"));
                report["minimum_cosine"]=minimum;report["all_four_full_rankings_matched"]=true;report["actual_backend"]="GPUCompute";
                report["cpu_computation_count"]=0;report["readback_flush_count"]=0;report["phase"]="completed";report["success"]=true;
            }
            catch(Exception error){report["error"]=error.ToString();throw;}
            finally{Save();}
        }
    }
}
