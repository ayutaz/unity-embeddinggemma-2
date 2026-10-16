using System;
using System.Reflection;
using EmbeddingGemma.Tests;
using NUnit.Framework;
using Unity.InferenceEngine;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class AudioReluPathTests
    {
        [TestCase("valid")]
        [TestCase("wrong_activation")]
        [TestCase("ambiguous")]
        public void SubsampleActivationMustFollowOneTransposeReluPath(string scenario)
        {
            var method=typeof(AudioCoefficientLayerMap).GetMethod("ResolveReluOutput",BindingFlags.Static|BindingFlags.Public);
            Assert.That(method,Is.Not.Null,"Strict subsample norm-to-ReLU binding is missing.");
            var graph=new FunctionalGraph();
            var input=graph.AddInput<float>(new TensorShape(1,2),"input");
            var transposed=Functional.Transpose(input,0,1);
            var output=scenario=="wrong_activation"?Functional.Exp(transposed):Functional.Relu(transposed);
            var model=scenario=="ambiguous"?graph.Compile(output,Functional.Exp(transposed)):graph.Compile(output);
            if(scenario=="valid")
                Assert.That((int)method.Invoke(null,new object[]{model,model.inputs[0].index}),Is.EqualTo(model.outputs[0].index));
            else
            {
                var error=Assert.Throws<TargetInvocationException>(()=>method.Invoke(null,new object[]{model,model.inputs[0].index}));
                Assert.That(error.InnerException,Is.TypeOf<InvalidOperationException>());
            }
        }
    }
}
