using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using NUnit.Framework;
using Unity.InferenceEngine;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class QualifiedVisionRotaryTableTests
    {
        static byte[] Bytes(Constant constant)
        {
            var segment=(ArraySegment<byte>)typeof(Constant).GetField("array",BindingFlags.Instance|BindingFlags.NonPublic).GetValue(constant);
            var result=new byte[segment.Count];Buffer.BlockCopy(segment.Array,segment.Offset,result,0,result.Length);return result;
        }
        static string Hash(byte[] bytes){using var stream=new MemoryStream(bytes,false);return PlayerFileHash.Compute(stream,out _);}
        static Model Source()
        {
            var graph=new FunctionalGraph();var shape=new TensorShape(1,2,1,32);var input=graph.AddInput<float>(shape,"input");
            return graph.Compile(Enumerable.Range(0,4).Select(group=>input*Functional.Constant(shape,
                Enumerable.Range(0,64).Select(i=>.1f+group*.15f+i*.00001f).ToArray())).ToArray());
        }
        static Dictionary<string,float[]> Corrections(Model model)
            =>model.constants.Where(c=>c.dataType==DataType.Float&&c.shape==new TensorShape(1,2,1,32)).ToDictionary(c=>Hash(Bytes(c)),c=>
            {
                var raw=Bytes(c);var values=new float[64];Buffer.BlockCopy(raw,0,values,0,raw.Length);
                for(var i=0;i<values.Length;i++)values[i]=BitConverter.Int32BitsToSingle(BitConverter.SingleToInt32Bits(values[i])+1);return values;
            });
        static Model Rebind(Model source,Dictionary<string,float[]> values)
        {
            var type=typeof(QualifiedVisionRmsOperation).Assembly.GetType("EmbeddingGemma.Validation.Tests.QualifiedVisionRotaryTables");
            Assert.That(type,Is.Not.Null,"Qualified derived rotary table correction is missing.");
            try{return (Model)type.GetMethod("Rebind").Invoke(null,new object[]{source,values});}
            catch(TargetInvocationException error){System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(error.InnerException).Throw();throw;}
        }
        [Test]
        public void RebindingPreservesBorrowedGraphAndOwnsCorrectedValues()
        {
            var source=Source();var corrections=Corrections(source);Assert.That(corrections.Count,Is.EqualTo(4));
            var hashes=source.constants.Select(c=>Hash(Bytes(c))).ToArray();var inputs=source.layers.Select(l=>l.inputs.ToArray()).ToArray();
            var result=Rebind(source,corrections);var expected=corrections.Values.Select(values=>Hash(FloatBytes(values))).ToArray();
            foreach(var values in corrections.Values)Array.Fill(values,99f);
            Assert.That(source.constants.Select(c=>Hash(Bytes(c))).ToArray(),Is.EqualTo(hashes));
            for(var i=0;i<inputs.Length;i++)Assert.That(source.layers[i].inputs,Is.EqualTo(inputs[i]));
            foreach(var constant in source.constants)Assert.That(result.constants.Contains(constant),Is.True);
            foreach(var hash in expected)Assert.That(result.constants.Any(c=>Hash(Bytes(c))==hash),Is.True,"Corrected bytes must be independently owned.");
            Assert.That(result.outputs.Select(o=>o.index),Is.EqualTo(source.outputs.Select(o=>o.index)));
        }
        [TestCase("missing")] [TestCase("nonfinite")] [TestCase("unbounded")]
        public void RebindingRejectsUnqualifiedCorrectionBeforeChangingSource(string problem)
        {
            var source=Source();var values=Corrections(source);var hashes=source.constants.Select(c=>Hash(Bytes(c))).ToArray();
            if(problem=="missing")values.Remove(values.Keys.First());
            else values.Values.First()[0]=problem=="nonfinite"?float.NaN:.9f;
            Assert.Throws<ArgumentException>(()=>Rebind(source,values));
            Assert.That(source.constants.Select(c=>Hash(Bytes(c))).ToArray(),Is.EqualTo(hashes));
        }
        static byte[] FloatBytes(float[] values){var bytes=new byte[values.Length*4];Buffer.BlockCopy(values,0,bytes,0,bytes.Length);return bytes;}
    }
}
