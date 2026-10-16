using System;
using System.Reflection;
using EmbeddingGemma.Tests;
using NUnit.Framework;
using Unity.InferenceEngine;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class AudioActivationPathTests
    {
        [TestCase("valid")]
        [TestCase("wrong_offset")]
        [TestCase("wrong_exponential_input")]
        public void DivisionSiluPathMustPreserveNumeratorAndUnitDenominator(string scenario)
        {
            var method=typeof(AudioCoefficientLayerMap).GetMethod("ResolveSiluInput",BindingFlags.Public|BindingFlags.Static);
            Assert.That(method,Is.Not.Null,"Strict division-order SiLU path binding is missing.");
            var graph=new FunctionalGraph();
            var x=graph.AddInput<float>(new TensorShape(1,2),"input");
            var exponential=Functional.Exp(scenario=="wrong_exponential_input"?-(x+1f):-x);
            var output=x/(exponential+(scenario=="wrong_offset"?2f:1f));
            var model=graph.Compile(output);
            if(scenario=="valid")
                Assert.That((int)method.Invoke(null,new object[]{model,model.outputs[0].index}),Is.EqualTo(model.inputs[0].index));
            else
            {
                var error=Assert.Throws<TargetInvocationException>(()=>method.Invoke(null,new object[]{model,model.outputs[0].index}));
                Assert.That(error.InnerException,Is.TypeOf<InvalidOperationException>());
            }
        }
    }
}
