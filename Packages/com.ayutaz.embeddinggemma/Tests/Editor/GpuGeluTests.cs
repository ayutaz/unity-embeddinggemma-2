using System;
using System.Collections;
using System.Reflection;
using System.Runtime.ExceptionServices;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEngine;

namespace EmbeddingGemma.Tests
{
    public sealed class GpuGeluTests
    {
        [Test]
        public void ExplicitFusedInnerGeluPreservesDefaultAndOwnedResults()
        {
            // Fixed procedural-corpus inputs that distinguish the rounding paths.
            var values=new[]{-1.985658884048462f,-1.8951334953308105f,-1.2458938360214233f,-.9991238117218018f,
                -1.2450087070465088f,-1.072245478630066f,-1.2215272188186646f,-.5450138449668884f,
                -1.258405089378357f,-2.172506809234619f,-.9882657527923584f,-1.1261590719223022f};
            var shape=new TensorShape(1,values.Length);
            var constructor=typeof(GpuGelu).GetConstructor(new[]{typeof(TensorShape),typeof(bool),typeof(bool)});
            Assert.That(constructor,Is.Not.Null,"Explicit fused GELU inner expression is missing.");
            using var fused=(IGpuTensorOperation)constructor.Invoke(new object[]{shape,true,true});
            using var separate=(IGpuTensorOperation)constructor.Invoke(new object[]{shape,true,false});
            using var original=new GpuGelu(shape,true);
            using var input=new Tensor<float>(shape,values);ComputeTensorData.Pin(input);
            using var result=fused.Execute(input);using var baseline=separate.Execute(input);using var prior=original.Execute(input);
            var actual=result.DownloadToArray();var normal=baseline.DownloadToArray();var legacy=prior.DownloadToArray();var changed=0;
            for(var i=0;i<values.Length;i++)
            {
                Assert.That(BitConverter.SingleToInt32Bits(normal[i]),Is.EqualTo(BitConverter.SingleToInt32Bits(legacy[i])));
                if(BitConverter.SingleToInt32Bits(actual[i])!=BitConverter.SingleToInt32Bits(normal[i]))changed++;
                var x=(double)values[i];var expected=.5*x*(1+Math.Tanh(.7978845608028654*(x+.044715*x*x*x)));
                Assert.That(Math.Abs(actual[i]-expected),Is.LessThanOrEqualTo(1e-5+1e-4*Math.Abs(expected)));
            }
            Assert.That(changed,Is.GreaterThan(0),"Fixture must distinguish fused and separate rounding.");
            using var again=fused.Execute(input);Assert.That(again.DownloadToArray(),Is.EqualTo(actual));
            fused.Dispose();fused.Dispose();Assert.That(result.DownloadToArray(),Is.EqualTo(actual));Assert.That(input.DownloadToArray(),Is.EqualTo(values));
            Assert.Throws<ObjectDisposedException>(()=>fused.Execute(input));
        }

        [TestCase(false)] [TestCase(true)]
        public void OptInReleaseClearsBothInternalGeluCachesAndPreservesOwnedResults(bool release)
        {
            var shape=new TensorShape(1,9);
            var constructor=typeof(GpuGelu).GetConstructor(new[]{typeof(TensorShape),typeof(bool)});
            if(release)Assert.That(constructor,Is.Not.Null,"GELU internal GPU cache release is missing.");
            using var operation=release?(IGpuTensorOperation)constructor.Invoke(new object[]{shape,true}):Create(shape);
            using var input=new Tensor<float>(shape,Values);ComputeTensorData.Pin(input);
            using var output=operation.Execute(input);using var repeated=operation.Execute(input);
            const BindingFlags fields=BindingFlags.Instance|BindingFlags.NonPublic;
            var pipeline=(GpuProjectionPipeline)typeof(GpuGelu).GetField("pipeline",fields).GetValue(operation);
            var workers=(Worker[])typeof(GpuProjectionPipeline).GetField("workers",fields).GetValue(pipeline);
            Assert.That(workers.Length,Is.EqualTo(2));
            foreach(var worker in workers)
            {
                var storage=typeof(Worker).GetField("m_Storage",fields).GetValue(worker);
                var pool=storage.GetType().GetField("m_TensorPool",fields).GetValue(storage);
                var cache=pool.GetType().GetField("m_computeMemoryPool",fields).GetValue(pool);
                var count=((IDictionary)cache.GetType().GetField("freeBuffers",fields).GetValue(cache)).Count;
                if(release)Assert.That(count,Is.Zero);else Assert.That(count,Is.GreaterThan(0),"The existing default keeps both internal Worker caches.");
            }
            var actual=output.DownloadToArray();Assert.That(repeated.DownloadToArray(),Is.EqualTo(actual));operation.Dispose();operation.Dispose();
            Assert.That(output.DownloadToArray(),Is.EqualTo(actual));Assert.That(input.DownloadToArray(),Is.EqualTo(Values));
            for(var i=0;i<actual.Length;i++)Assert.That(Math.Abs((double)actual[i]-Expected[i]),Is.LessThanOrEqualTo(1e-5+1e-4*Math.Abs((double)Expected[i])));
        }

        static IGpuTensorOperation Create(TensorShape shape)
        {
            Assert.That(SystemInfo.supportsComputeShaders,Is.True,"Actual GPU required; no skip.");
            var type=typeof(IGpuTensorOperation).Assembly.GetType("EmbeddingGemma.GpuGelu");
            Assert.That(type,Is.Not.Null,"Fixed-shape GPU tanh GELU is missing.");
            try{return (IGpuTensorOperation)Activator.CreateInstance(type,new object[]{shape});}
            catch(TargetInvocationException error){ExceptionDispatchInfo.Capture(error.InnerException).Throw();throw;}
        }
        static readonly float[] Values={-10f,-3f,-1f,-.1f,0f,.1f,1f,3f,10f};
        // Independent torch 2.14.1 FP32 gelu(approximate="tanh") reference.
        static readonly float[] Expected={0f,-.0036374330520629883f,-.15880799293518066f,-.046017248183488846f,0f,.053982753306627274f,.8411920070648193f,2.9963626861572266f,10f};
        [TestCase(false)] [TestCase(true)]
        public void GpuGeluPreservesInputAndReturnsOwnedResultsAcrossReuseAndDispose(bool threeDimensions)
        {
            var shape=threeDimensions?new TensorShape(1,1,9):new TensorShape(1,9);using var operation=Create(shape);
            using var input=new Tensor<float>(shape,Values);ComputeTensorData.Pin(input);
            using var output=operation.Execute(input);using var repeated=operation.Execute(input);operation.Dispose();operation.Dispose();
            Assert.That(output.backendType,Is.EqualTo(BackendType.GPUCompute));Assert.That(output.shape,Is.EqualTo(shape));
            var actual=output.DownloadToArray();var again=repeated.DownloadToArray();Assert.That(again,Is.EqualTo(actual));
            for(var i=0;i<actual.Length;i++)Assert.That(Math.Abs((double)actual[i]-Expected[i]),Is.LessThanOrEqualTo(1e-5+1e-4*Math.Abs((double)Expected[i])));
            Assert.That(input.DownloadToArray(),Is.EqualTo(Values));
            Assert.Throws<ObjectDisposedException>(()=>operation.Execute(input));
        }
        [Test]
        public void GpuGeluRejectsInvalidInputsAndRemainsReusable()
        {
            using var operation=Create(new TensorShape(1,9));Assert.Throws<ArgumentNullException>(()=>operation.Execute(null));
            using var cpu=new Tensor<float>(new TensorShape(1,9),Values);Assert.Throws<ArgumentException>(()=>operation.Execute(cpu));
            using var wrong=new Tensor<float>(new TensorShape(1,8),new float[8]);ComputeTensorData.Pin(wrong);Assert.Throws<ArgumentException>(()=>operation.Execute(wrong));
            using var dead=new Tensor<float>(new TensorShape(1,9),Values);dead.Dispose();Assert.Throws<ObjectDisposedException>(()=>operation.Execute(dead));
            ComputeTensorData.Pin(cpu);using var result=operation.Execute(cpu);Assert.That(result.shape,Is.EqualTo(cpu.shape));
        }
        [Test]
        public void GpuGeluRejectsEmptyFixedShape()
            => Assert.Throws<ArgumentException>(()=>Create(new TensorShape(0,9)));
    }
}
