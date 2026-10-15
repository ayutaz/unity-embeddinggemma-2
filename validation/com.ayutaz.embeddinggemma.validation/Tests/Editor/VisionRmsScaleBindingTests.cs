using System;
using System.IO;
using System.Linq;
using System.Reflection;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEngine;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class VisionRmsScaleBindingTests
    {
        static byte[] Bytes(Model model){using var stream=new MemoryStream();ModelWriter.Save(stream,model);return stream.ToArray();}
        static Model Fixture(float[] scale)
        {
            var graph=new FunctionalGraph();var input=graph.AddInput<float>(new TensorShape(1,768),"input");
            return graph.Compile(input*Functional.Constant(new TensorShape(scale.Length),scale));
        }
        static MethodInfo Required()
        {
            var method=typeof(QualifiedVisionRmsOperation).GetMethod("BorrowScale");
            Assert.That(method,Is.Not.Null,"Original-scale borrowing is missing.");return method;
        }
        [TestCase(BackendType.CPU)] [TestCase(BackendType.GPUCompute)]
        public void OriginalScaleBindingBorrowsBytesAndPreservesTemplateAndCoefficientSource(BackendType backend)
        {
            if(backend==BackendType.GPUCompute)Assert.That(SystemInfo.supportsComputeShaders,Is.True,"Actual GPU required; never skip.");
            var template=Fixture(Enumerable.Repeat(3f,768).ToArray());var original=Fixture(Enumerable.Range(0,768).Select(i=>i/1024f-1f).ToArray());
            var scale=original.constants.Single(value=>value.shape==new TensorShape(768));var before=Bytes(template);var coefficientBefore=Bytes(original);
            var bound=(Model)Required().Invoke(null,new object[]{template,scale});var replacement=bound.constants.Single(value=>value.shape==new TensorShape(768));
            var field=typeof(Constant).GetField("array",BindingFlags.Instance|BindingFlags.NonPublic);
            Assert.That(((ArraySegment<byte>)field.GetValue(replacement)).Array,Is.SameAs(((ArraySegment<byte>)field.GetValue(scale)).Array));
            using var input=new Tensor<float>(new TensorShape(1,768),Enumerable.Repeat(2f,768).ToArray());if(backend==BackendType.GPUCompute)ComputeTensorData.Pin(input);
            using var worker=new Worker(bound,backend);worker.Schedule(input);
            Assert.That(((Tensor<float>)worker.PeekOutput()).DownloadToArray(),Is.EqualTo(Enumerable.Range(0,768).Select(i=>2f*(i/1024f-1f)).ToArray()));
            Assert.That(Bytes(template),Is.EqualTo(before));Assert.That(Bytes(original),Is.EqualTo(coefficientBefore));
        }
        [TestCase("width")] [TestCase("nonfinite")]
        public void OriginalScaleBindingRejectsInvalidCoefficientsWithoutChangingTemplate(string problem)
        {
            var template=Fixture(Enumerable.Repeat(3f,768).ToArray());var values=Enumerable.Repeat(1f,problem=="width"?767:768).ToArray();
            if(problem=="nonfinite")values[0]=float.NaN;
            var graph=new FunctionalGraph();var input=graph.AddInput<float>(new TensorShape(1,values.Length),"input");
            var source=graph.Compile(input*Functional.Constant(new TensorShape(values.Length),values));
            var scale=source.constants.Single(value=>value.shape==new TensorShape(values.Length));var before=Bytes(template);var method=Required();
            var error=Assert.Throws<TargetInvocationException>(()=>method.Invoke(null,new object[]{template,scale}));Assert.That(error.InnerException,Is.TypeOf<ArgumentException>());
            Assert.That(Bytes(template),Is.EqualTo(before));
        }
    }
}
