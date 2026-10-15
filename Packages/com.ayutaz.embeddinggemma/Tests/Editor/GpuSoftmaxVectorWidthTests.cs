using System;
using System.Linq;
using System.Reflection;
using System.Runtime.ExceptionServices;
using NUnit.Framework;
using Unity.InferenceEngine;

namespace EmbeddingGemma.Tests
{
    public sealed class GpuSoftmaxVectorWidthTests
    {
        static IGpuTensorOperation Create(int width)
        {
            var constructor=typeof(GpuSoftmax).GetConstructor(new[]{typeof(int)});
            Assert.That(constructor,Is.Not.Null,"Explicit softmax reduction vector width is missing.");
            try{return (IGpuTensorOperation)constructor.Invoke(new object[]{width});}
            catch(TargetInvocationException error){ExceptionDispatchInfo.Capture(error.InnerException).Throw();throw;}
        }

        [TestCase(8)] [TestCase(16)]
        public void ExplicitGpuSoftmaxKeepsInputAndOwnedOutputsAcrossReuseAndDispose(int width)
        {
            var values=Enumerable.Range(0,38).Select(i=>(float)(i%19-9)/3).ToArray();
            using var input=new Tensor<float>(new TensorShape(2,19),values);ComputeTensorData.Pin(input);
            using var operation=Create(width);using var output=operation.Execute(input);using var again=operation.Execute(input);
            var actual=output.DownloadToArray();Assert.That(again.DownloadToArray(),Is.EqualTo(actual));
            for(var row=0;row<2;row++)
            {
                var exponentials=values.Skip(row*19).Take(19).Select(value=>Math.Exp(value)).ToArray();var total=exponentials.Sum();
                for(var column=0;column<19;column++)Assert.That(Math.Abs(actual[row*19+column]-exponentials[column]/total),Is.LessThan(1e-6));
            }
            operation.Dispose();operation.Dispose();Assert.That(output.DownloadToArray(),Is.EqualTo(actual));
            Assert.That(input.DownloadToArray(),Is.EqualTo(values));Assert.That(output.backendType,Is.EqualTo(BackendType.GPUCompute));
            Assert.Throws<ObjectDisposedException>(()=>operation.Execute(input));
        }

        [Test]
        public void Vector16RejectsInvalidCallsAndRemainsUsableWhileDefaultKeepsVector8Bits()
        {
            foreach(var width in new[]{-1,0,4,32})Assert.Throws<ArgumentOutOfRangeException>(()=>Create(width));
            using var operation=Create(16);Assert.Throws<ArgumentNullException>(()=>operation.Execute(null));
            var values=Enumerable.Range(0,63).Select(i=>(float)(i%17-8)/7).ToArray();
            using var input=new Tensor<float>(new TensorShape(3,21),values);
            Assert.Throws<ArgumentException>(()=>operation.Execute(input));ComputeTensorData.Pin(input);
            using var output=operation.Execute(input);Assert.That(output.shape,Is.EqualTo(input.shape));
            using var original=new GpuSoftmax();using var eight=Create(8);
            using var first=original.Execute(input);using var second=eight.Execute(input);
            Assert.That(second.DownloadToArray().Select(BitConverter.SingleToInt32Bits),Is.EqualTo(first.DownloadToArray().Select(BitConverter.SingleToInt32Bits)));
        }
    }
}
