using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Runtime.ExceptionServices;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEngine;

namespace EmbeddingGemma.Tests
{
    public sealed class ImageFeatureEncoderTests
    {
        static Type RuntimeType(string name)
        {var type=typeof(ImagePreprocessor).Assembly.GetType("EmbeddingGemma."+name);Assert.That(type,Is.Not.Null,"Image feature encoder API is missing: "+name);return type;}
        static object Profile(int width=768,int height=768)=>Construct(RuntimeType("ImageEncoderProfile"),width,height);
        static object Construct(Type type,params object[] args)
        {try{return Activator.CreateInstance(type,args);}catch(TargetInvocationException error){ExceptionDispatchInfo.Capture(error.InnerException).Throw();throw;}}
        static object Encode(object encoder,ImagePreprocessingResult input)
        {try{return encoder.GetType().GetMethod("Encode").Invoke(encoder,new object[]{input});}catch(TargetInvocationException error){ExceptionDispatchInfo.Capture(error.InnerException).Throw();throw;}}
        static Model Model(int count=256,int width=512,int inputs=1)
        {
            var graph=new FunctionalGraph();var pixels=graph.AddInput<float>(new TensorShape(1,2520,768),"pixels");
            if(inputs==2)graph.AddInput<float>(new TensorShape(1),"unexpected");
            return graph.Compile(Functional.ReduceSum(pixels,new[]{0,1,2},false)/(2520*768f)*
                Functional.Constant(new TensorShape(count,width),Enumerable.Repeat(1f,count*width).ToArray()));
        }
        static int[] Positions(int width,int height)
        {
            var positions=Enumerable.Repeat(-1,5040).ToArray();var columns=width/16;var rows=height/16;
            for(var row=0;row<rows;row++)for(var column=0;column<columns;column++)
            {var index=2*(row*columns+column);positions[index]=column;positions[index+1]=row;}
            return positions;
        }
        static ImagePreprocessingResult Input(BackendType backend=BackendType.CPU,int width=768,int height=768,int count=256,int[] positions=null)
        {
            var pixels=new Tensor<float>(new TensorShape(1,2520,768),Enumerable.Repeat(0.25f,2520*768).ToArray());
            if(backend==BackendType.GPUCompute)ComputeTensorData.Pin(pixels);
            return (ImagePreprocessingResult)typeof(ImagePreprocessingResult).GetConstructors(BindingFlags.NonPublic|BindingFlags.Instance).Single().Invoke(
                new object[]{pixels,width,height,count,positions??Positions(width,height),new List<Tensor>{pixels},new List<ComputeBuffer>()});
        }
        static object Encoder(Model model,BackendType backend,object profile)=>Construct(RuntimeType("ImageFeatureEncoder"),model,backend,profile);
        static Tensor<float> Tokens(object result)=>(Tensor<float>)result.GetType().GetProperty("SoftTokens").GetValue(result);
        [TestCase(768,768,256)] [TestCase(1008,576,252)] [TestCase(576,1008,252)]
        public void RectangularProfilesUseWholeThreeByThreePatchPools(int width,int height,int count)
        {var profile=Profile(width,height);Assert.That(profile.GetType().GetProperty("FeatureCount").GetValue(profile),Is.EqualTo(count));}
        [TestCase(0,768)] [TestCase(-48,768)] [TestCase(752,768)] [TestCase(816,816)]
        public void InvalidGeometryIsRejected(int width,int height)
        {Assert.Throws<ArgumentException>(()=>Profile(width,height));}
        [TestCase(BackendType.CPU)] [TestCase(BackendType.GPUCompute)]
        public void ProjectedTokensRemainOnRequestedDeviceAndOutliveWorker(BackendType backend)
        {
            if(backend==BackendType.GPUCompute)Assert.That(SystemInfo.supportsComputeShaders,Is.True);
            var model=Model();var encoder=Encoder(model,backend,Profile());using var input=Input(backend);
            var result=Encode(encoder,input);var tokens=Tokens(result);
            Assert.That(tokens.shape,Is.EqualTo(new TensorShape(256,512)));Assert.That(tokens.backendType,Is.EqualTo(backend));
            ((IDisposable)encoder).Dispose();Assert.That(tokens.DownloadToArray(),Is.All.EqualTo(0.25f).Within(1e-5));
            Assert.That(input.Pixels.dataOnBackend,Is.Not.Null,"The caller retains its preprocessing result.");
            ((IDisposable)result).Dispose();((IDisposable)result).Dispose();
        }
        [TestCase("positions")] [TestCase("count")] [TestCase("geometry")]
        public void MetadataMustMatchTheBakedProfile(string problem)
        {
            var positions=Positions(768,768);if(problem=="positions")positions[0]=1;
            using var input=Input(width:problem=="geometry"?720:768,count:problem=="count"?252:256,positions:positions);
            using var encoder=(IDisposable)Encoder(Model(),BackendType.CPU,Profile());
            Assert.Throws<ArgumentException>(()=>Encode(encoder,input));
        }
        [Test] public void NullAndDisposedInputsAreRejected()
        {
            using var encoder=(IDisposable)Encoder(Model(),BackendType.CPU,Profile());Assert.Throws<ArgumentNullException>(()=>Encode(encoder,null));
            var input=Input();input.Dispose();Assert.Throws<ObjectDisposedException>(()=>Encode(encoder,input));
        }
        [Test] public void DisposedWorkerRejectsFurtherCalls()
        {
            var encoder=(IDisposable)Encoder(Model(),BackendType.CPU,Profile());encoder.Dispose();encoder.Dispose();using var input=Input();
            Assert.Throws<ObjectDisposedException>(()=>Encode(encoder,input));
        }
        [Test] public void WrongOutputShapeIsRejected()
        {using var encoder=(IDisposable)Encoder(Model(255),BackendType.CPU,Profile());using var input=Input();Assert.Throws<InvalidOperationException>(()=>Encode(encoder,input));}
        [Test] public void ExtraModelInputsAreRejected()
        {Assert.Throws<ArgumentException>(()=>Encoder(Model(inputs:2),BackendType.CPU,Profile()));}
    }
}
