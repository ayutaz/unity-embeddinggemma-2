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
    public sealed class VisionOriginalRotaryGpuTests
    {
        [TestCase("cos","cat","8e7ad14c37f14fb5b11e70b08701181c67ad24f4a874385eee5dd63655b39b75")]
        [TestCase("sin","cat_1","1eeda3ae4a8b5bf7f09e29b6d6c763054d9b49ec6de551e09d54a080232a331e")]
        public void SavedRotaryConstantsMustMatchOriginalNativeHalves(string kind,string node,string expectedSha)
        {
            var source=Environment.GetEnvironmentVariable("EG_ACCEPTANCE_SOURCE");
            var output=Path.GetFullPath(Path.Combine(Application.dataPath,"../artifacts/vision-original-rotary-constant-audit-"+source+"-"+kind+".json"));
            Assert.That(File.Exists(output),Is.False);
            var report=new JObject{["success"]=false,["runtime_source"]=source,["scope"]="saved_rotary_constants_vs_original_native_halves_without_inference",
                ["gpu_validated"]=false,["model_inference_executed"]=false,["full_encoder_validated"]=false,["end_to_end_validated"]=false};
            void Save(){Directory.CreateDirectory(Path.GetDirectoryName(output));File.WriteAllText(output,report.ToString());}
            string Hash(byte[] bytes){using var stream=new MemoryStream(bytes,false);return PlayerFileHash.Compute(stream,out _);}
            void Check(string path,string sha)=>Assert.That(Hash(File.ReadAllBytes(path)),Is.EqualTo(sha));
            Save();
            try
            {
                var locked=JObject.Parse(File.ReadAllText(Path.Combine(Application.dataPath,"../Packages/packages-lock.json")));
                foreach(var package in new[]{"com.ayutaz.embeddinggemma","com.ayutaz.embeddinggemma.validation"})Assert.That(locked["dependencies"][package].Value<string>("hash"),Is.EqualTo(source));
                var path=Path.GetFullPath(Path.Combine(Application.dataPath,"../../k/artifacts/vision-encoder/square-fp32.sentis"));
                using(var stream=File.OpenRead(path))Assert.That(PlayerFileHash.Compute(stream,out _),Is.EqualTo("4ecd0106463e833588c7584a58ea4973fdbab1eb1a56389c2f1597d3e7df34e7"));
                var original=TextModelFile.Load(path);var array=typeof(Constant).GetField("array",BindingFlags.Instance|BindingFlags.NonPublic);
                var groups=original.constants.Where(c=>c.dataType==DataType.Float&&c.shape.length==2520*32).Select(c=>
                {
                    Assert.That(c.shape,Is.EqualTo(new TensorShape(1,2520,1,32)));
                    var segment=(ArraySegment<byte>)array.GetValue(c);var bytes=new byte[segment.Count];Buffer.BlockCopy(segment.Array,segment.Offset,bytes,0,bytes.Length);
                    return (constant:c,bytes,hash:Hash(bytes));
                }).GroupBy(value=>value.hash).ToArray();Assert.That(groups.Length,Is.EqualTo(4));
                foreach(var group in groups)Assert.That(group.Count(),Is.EqualTo(32),"Each position table is reused by Q/K across sixteen layers.");
                var expectedPath=Path.GetFullPath(Path.Combine(Application.dataPath,"../../numerical-acceptance-state/r38716679261/"+node+"-fp32.bin"));Check(expectedPath,expectedSha);
                var all=Floats(File.ReadAllBytes(expectedPath));Assert.That(all.Length,Is.EqualTo(2520*64));
                var rows=new JArray();report["comparisons"]=rows;var best=new[]{int.MaxValue,int.MaxValue};
                var folder=Path.Combine(Path.GetDirectoryName(output),"vision-original-rotary-constant-audit-"+source+"-captures");Directory.CreateDirectory(folder);
                foreach(var group in groups)
                {
                    var item=group.First();var file=Path.Combine(folder,"constant-"+item.hash+".bin");
                    if(File.Exists(file))Check(file,item.hash);else File.WriteAllBytes(file,item.bytes);
                    var values=Floats(item.bytes);
                    for(var half=0;half<2;half++)
                    {
                        var bits=0;double maximum=0;
                        for(var i=0;i<values.Length;i++)
                        {
                            var expected=all[(i/32)*64+half*32+i%32];
                            if(BitConverter.SingleToInt32Bits(values[i])!=BitConverter.SingleToInt32Bits(expected))bits++;
                            maximum=Math.Max(maximum,Math.Abs((double)values[i]-expected));
                        }
                        rows.Add(new JObject{["constant_sha256"]=item.hash,["occurrences"]=group.Count(),["native_half"]=half,["bit_differences"]=bits,["maximum_absolute_error"]=maximum});
                        best[half]=Math.Min(best[half],bits);
                    }
                }
                report["best_half_bit_differences"]=JArray.FromObject(best);Save();
                Assert.That(best,Is.EqualTo(new[]{0,0}),"Saved derived rotary tables must preserve native bits.");report["success"]=true;
            }
            catch(Exception error){report["error"]=error.ToString();throw;}
            finally{Save();}
        }
        static float[] Floats(byte[] raw){var values=new float[raw.Length/4];Buffer.BlockCopy(raw,0,values,0,raw.Length);return values;}
    }
}
