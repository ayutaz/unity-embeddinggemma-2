using System;
using System.IO;
using System.Reflection;
using System.Security.Cryptography;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEngine;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class ModelCoefficientHashTests
    {
        static string Compute(Model model,out string backend,Func<Stream,string> accelerated=null)
        {
            var type=typeof(PlayerFileHash).Assembly.GetType("EmbeddingGemma.Validation.ModelCoefficientHash");
            Assert.That(type,Is.Not.Null,"Accelerated borrowed coefficient-byte audit is missing.");
            var arguments=new object[]{model,null,accelerated};var result=(string)type.GetMethod("Compute").Invoke(null,arguments);backend=(string)arguments[1];return result;
        }
        [TestCase(false)] [TestCase(true)]
        public void CoefficientAuditMatchesIndependentConcatenationAndPreservesBorrowedSegments(bool interrupted)
        {
            var model=new Model();model.constants.Add(new Constant(0,new TensorShape(2),new[]{1f,2f}));model.constants.Add(new Constant(1,new TensorShape(1),new[]{3f}));
            var field=typeof(Constant).GetField("array",BindingFlags.Instance|BindingFlags.NonPublic);
            var original=(ArraySegment<byte>)field.GetValue(model.constants[0]);var padded=new byte[13];Buffer.BlockCopy(original.Array,original.Offset,padded,3,8);
            field.SetValue(model.constants[0],new ArraySegment<byte>(padded,3,8));
            var expectedBytes=new byte[]{0,0,128,63,0,0,0,64,0,0,64,64};using var hash=SHA256.Create();var expected=BitConverter.ToString(hash.ComputeHash(expectedBytes)).Replace("-","").ToLowerInvariant();
            Func<Stream,string> fail=stream=>
            {
                Assert.That(stream.Length,Is.EqualTo(12));Assert.That(stream.CanWrite,Is.False);stream.Seek(-1,SeekOrigin.End);Assert.That(stream.ReadByte(),Is.EqualTo(64));
                throw new CryptographicException("Simulated interrupted native audit.");
            };
            var actual=Compute(model,out var backend,interrupted?fail:null);Assert.That(actual,Is.EqualTo(expected));
            if(interrupted)Assert.That(backend,Is.EqualTo("dotnet_sha256"));else Assert.That(backend=="windows_cng"||backend=="dotnet_sha256",Is.True);
            var after=(ArraySegment<byte>)field.GetValue(model.constants[0]);Assert.That(after.Array,Is.SameAs(padded));Assert.That(after.Offset,Is.EqualTo(3));Assert.That(after.Count,Is.EqualTo(8));
            Assert.That(padded[3],Is.Zero);Assert.That(padded[10],Is.EqualTo(64));
        }
        [Test]
        public void CoefficientAuditMatchesPinnedWholeVisionDigestWithoutModelInference()
        {
            var source=Environment.GetEnvironmentVariable("EG_ACCEPTANCE_SOURCE");
            var output=Path.GetFullPath(Path.Combine(Application.dataPath,"../artifacts/vision-coefficient-audit-"+source+".json"));Assert.That(File.Exists(output),Is.False);
            var path=Path.GetFullPath(Path.Combine(Application.dataPath,"../../k/artifacts/vision-encoder/square-fp32.sentis"));
            using(var stream=File.OpenRead(path))Assert.That(PlayerFileHash.Compute(stream,out _),Is.EqualTo("4ecd0106463e833588c7584a58ea4973fdbab1eb1a56389c2f1597d3e7df34e7"));
            var model=TextModelFile.Load(path);var timer=System.Diagnostics.Stopwatch.StartNew();
            var digest=Compute(model,out var backend);var elapsed=timer.Elapsed.TotalMilliseconds;
            Assert.That(digest,Is.EqualTo("3e14a1b4f9ab810af4fc118e1a8b6842615b6f2b9342dbc5348067f8c20f6886"));
            var report=new JObject{["success"]=true,["runtime_source"]=source,["scope"]="same_pinned_original_coefficient_bytes_hash_only_no_model_inference",
                ["digest"]=digest,["backend"]=backend,["audit_ms"]=elapsed,["model_inference_performed"]=false,["full_encoder_validated"]=false,["end_to_end_validated"]=false};
            Directory.CreateDirectory(Path.GetDirectoryName(output));File.WriteAllText(output,report.ToString());
        }
    }
}
