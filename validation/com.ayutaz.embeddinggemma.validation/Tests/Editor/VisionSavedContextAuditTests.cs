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
    public sealed class VisionSavedContextAuditTests
    {
        [Test]
        public void AuditOriginalAndTraceCoefficientsAndRotaryConstantsWithoutInference()
        {
            var source=Environment.GetEnvironmentVariable("EG_ACCEPTANCE_SOURCE");var output=Path.GetFullPath(Path.Combine(Application.dataPath,"../artifacts/vision-saved-context-audit-"+source+".json"));
            Assert.That(File.Exists(output),Is.False);var report=new JObject{["success"]=false,["runtime_source"]=source,["inference_executed"]=false,["scope"]="original_and_diagnostic_saved_constant_context_audit_only",
                ["full_encoder_validated"]=false,["end_to_end_validated"]=false};
            void Save(){Directory.CreateDirectory(Path.GetDirectoryName(output));File.WriteAllText(output,report.ToString());}
            string Hash(string path){using var stream=File.OpenRead(path);return PlayerFileHash.Compute(stream,out _);}
            var root=Path.GetFullPath(Path.Combine(Application.dataPath,"../../k/artifacts"));Save();
            try
            {
                var locked=JObject.Parse(File.ReadAllText(Path.Combine(Application.dataPath,"../Packages/packages-lock.json")));
                foreach(var package in new[]{"com.ayutaz.embeddinggemma","com.ayutaz.embeddinggemma.validation"})Assert.That(locked["dependencies"][package].Value<string>("hash"),Is.EqualTo(source));
                var originalPath=Path.Combine(root,"vision-encoder/square-fp32.sentis");var tracePath=Path.Combine(root,"vision-diagnostics/vision-stages-fp32.sentis");
                Assert.That(Hash(originalPath),Is.EqualTo("4ecd0106463e833588c7584a58ea4973fdbab1eb1a56389c2f1597d3e7df34e7"));
                Assert.That(Hash(tracePath),Is.EqualTo("63f25d442e6921529e009c4dbf9ca7cec7528499959aaf3a043f072723135508"));
                var original=TextModelFile.Load(originalPath);var trace=TextModelFile.Load(tracePath);
                var array=typeof(Constant).GetField("array",BindingFlags.Instance|BindingFlags.NonPublic);
                JArray Constants(Model model)
                {
                    var rows=new JArray();foreach(var constant in model.constants)
                    {
                        var raw=(ArraySegment<byte>)array.GetValue(constant);string hash;
                        using(var stream=new MemoryStream(raw.Array,raw.Offset,raw.Count,false))hash=PlayerFileHash.Compute(stream,out _);
                        rows.Add(new JObject{["index"]=constant.index,["dtype"]=constant.dataType.ToString(),["shape"]=JArray.FromObject(Enumerable.Range(0,constant.shape.rank).Select(i=>constant.shape[i]).ToArray()),
                            ["elements"]=constant.shape.length,["bytes"]=raw.Count,["sha256"]=hash});
                    }
                    return rows;
                }
                var originals=Constants(original);var traces=Constants(trace);report["original_constants"]=originals;report["trace_constants"]=traces;
                report["original_coefficients_sha256"]=ModelCoefficientHash.Compute(original,out _);report["trace_coefficients_sha256"]=ModelCoefficientHash.Compute(trace,out _);
                var originalHashes=originals.Select(row=>row.Value<string>("sha256")).ToHashSet();var traceHashes=traces.Select(row=>row.Value<string>("sha256")).ToHashSet();
                report["original_unshared_constants"]=new JArray(originals.Where(row=>!traceHashes.Contains(row.Value<string>("sha256"))));
                report["trace_unshared_constants"]=new JArray(traces.Where(row=>!originalHashes.Contains(row.Value<string>("sha256"))));
                var diagnostics=Path.Combine(root,"vision-diagnostics");var manifest=Path.Combine(diagnostics,"vision-diagnostics.json");
                Assert.That(Hash(manifest),Is.EqualTo("998b7d0d200b257b0bc19de32f45ef27be4328088d2d51706ea956cc3b4461ce"));var reference=JObject.Parse(File.ReadAllText(manifest));
                var rotations=new JArray();report["rotary_constant_comparisons"]=rotations;
                foreach(var item in new[]{("original",original),("trace",trace)})
                foreach(var constant in item.Item2.constants.Where(value=>value.dataType==DataType.Float&&value.shape.length==2520*64))
                {
                    var raw=(ArraySegment<byte>)array.GetValue(constant);Assert.That(raw.Count,Is.EqualTo(2520*64*4));
                    var values=new float[2520*64];Buffer.BlockCopy(raw.Array,raw.Offset,values,0,raw.Count);
                    var row=new JObject{["model"]=item.Item1,["index"]=constant.index};
                    foreach(var stage in new[]{1,2})
                    {
                        var path=Path.Combine(diagnostics,"native-stage-"+stage+".bin");Assert.That(Hash(path),Is.EqualTo(reference["files"][Path.GetFileName(path)].Value<string>("sha256")));
                        var bytes=File.ReadAllBytes(path);var expected=new float[bytes.Length/4];Buffer.BlockCopy(bytes,0,expected,0,bytes.Length);int bits=0;double maximum=0;
                        for(var i=0;i<values.Length;i++){if(BitConverter.SingleToInt32Bits(values[i])!=BitConverter.SingleToInt32Bits(expected[i]))bits++;maximum=Math.Max(maximum,Math.Abs((double)values[i]-expected[i]));}
                        row[stage==1?"native_cos":"native_sin"]=new JObject{["bit_differences"]=bits,["maximum_absolute_error"]=maximum};
                    }
                    rotations.Add(row);
                }
                report["original_layers"]=original.layers.Count;report["trace_layers"]=trace.layers.Count;report["success"]=true;
            }
            catch(Exception error){report["error"]=error.ToString();throw;}
            finally{Save();}
        }
    }
}
