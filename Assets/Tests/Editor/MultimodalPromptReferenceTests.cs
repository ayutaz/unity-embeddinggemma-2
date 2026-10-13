using System;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Security.Cryptography;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using UnityEngine;

namespace EmbeddingGemma.Tests
{
    public sealed class MultimodalPromptReferenceTests
    {
        [Test]
        [Timeout(300000)]
        public void OfficialProcessorTokensAndMasksMatchTheDistributedRuntime()
        {
            var root = Path.GetFullPath(Path.Combine(Application.dataPath,"../artifacts/multimodal-prompts"));
            var resultPath = Path.Combine(root,"../multimodal-prompt-results/results.json");
            var measured = new JArray();
            var report = new JObject { ["stage"]="multimodal_prompt_tokens_only",["success"]=false,
                ["encoders_validated"]=false,["backbone_validated"]=false,["end_to_end_validated"]=false,
                ["unity"]=Application.unityVersion,["cases"]=measured };
            try
            {
                var manifest=JObject.Parse(File.ReadAllText(Path.Combine(root,"prompt-validation.json")));
                Assert.That(manifest.Value<bool>("success"),Is.True);
                Assert.That(manifest.Value<string>("stage"),Is.EqualTo("multimodal_prompt_tokens_only"));
                foreach(var item in (JObject)manifest["files"])
                {
                    var path=Path.GetFullPath(Path.Combine(root,item.Key));
                    Assert.That(path.StartsWith(root+Path.DirectorySeparatorChar,StringComparison.Ordinal),Is.True);
                    Assert.That(new FileInfo(path).Length,Is.EqualTo(item.Value.Value<long>("bytes")));
                    Assert.That(Hash(path),Is.EqualTo(item.Value.Value<string>("sha256")));
                }
                var reference=JObject.Parse(File.ReadAllText(Path.Combine(root,"prompt-reference.json")));
                var metadata=reference["metadata"];
                Assert.That(metadata.Value<string>("model_revision"),Is.EqualTo("914f7f89142e33e77833254d9c9b90c3cef7303b"));
                Assert.That(metadata.Value<string>("source_commit"),Is.EqualTo(File.ReadAllText(Path.Combine(root,"expected-source.txt")).Trim()));
                report["reference_source"]=metadata["source_commit"];
                var type=AppDomain.CurrentDomain.GetAssemblies().Select(a=>a.GetType("EmbeddingGemma.MultimodalPromptBuilder")).FirstOrDefault(t=>t!=null);
                Assert.That(type,Is.Not.Null);
                Assert.That(type.Assembly.GetName().Name,Is.EqualTo("EmbeddingGemma.Runtime"),"Prototype source is not a distributed Runtime result.");
                report["runtime_assembly"]=type.Assembly.GetName().Name;
                var tokenizer=File.ReadAllText(Path.Combine(root,"tokenizer.json"));
                var configuration=File.ReadAllText(Path.Combine(root,"tokenizer_config.json"));
                var rows=(JArray)reference["cases"];
                Assert.That(rows.Count,Is.EqualTo(15));
                foreach(var row in rows)
                {
                    var image=row.Value<int>("image_feature_count"); var audio=row.Value<int>("audio_feature_count");
                    var profile=new MultimodalBackboneProfile(row.Value<int>("sequence_length"),metadata.Value<int>("image_token_id"),metadata.Value<int>("audio_token_id"),image,audio,metadata.Value<int>("vocabulary_size"),metadata.Value<int>("pad_token_id"));
                    var builder=Activator.CreateInstance(type,new object[] {tokenizer,configuration,profile});
                    var role=(TextRole)Enum.Parse(typeof(TextRole),row.Value<string>("role"),true);
                    var mask=row["audio_feature_mask"].Type==JTokenType.Null?null:row["audio_feature_mask"].ToObject<int[]>();
                    var prompt=type.GetMethod("Build").Invoke(builder,new object[] {row.Value<string>("text"),image,mask,role,row.Value<string>("title")});
                    var ids=(int[])prompt.GetType().GetProperty("InputIds").GetValue(prompt);
                    var attention=(int[])prompt.GetType().GetProperty("AttentionMask").GetValue(prompt);
                    Assert.That(ids,Is.EqualTo(row["input_ids"].ToObject<int[]>()),row.Value<string>("id"));
                    Assert.That(attention,Is.EqualTo(row["attention_mask"].ToObject<int[]>()),row.Value<string>("id"));
                    Assert.That(prompt.GetType().GetProperty("FormattedText").GetValue(prompt),Is.EqualTo(row.Value<string>("formatted_text")));
                    measured.Add(new JObject { ["id"]=row["id"],["success"]=true,["image_features"]=image,["audio_features"]=audio });
                }
                report["success"]=true;
            }
            catch(Exception error) { report["error"]=error.ToString(); throw; }
            finally
            {
                report["completed_utc"]=DateTime.UtcNow.ToString("O");
                Directory.CreateDirectory(Path.GetDirectoryName(resultPath));File.WriteAllText(resultPath,report.ToString());
            }
        }
        static string Hash(string path)
        {
            using var file=File.OpenRead(path);using var hash=SHA256.Create();
            return BitConverter.ToString(hash.ComputeHash(file)).Replace("-","").ToLowerInvariant();
        }
    }
}
