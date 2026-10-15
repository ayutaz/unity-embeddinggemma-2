using System;
using System.Linq;
using NUnit.Framework;
using Unity.InferenceEngine;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class VisionTailRmsShapeTests
    {
        static Model Source(int rows)
        {
            var graph=new FunctionalGraph();var input=graph.AddInput<float>(new TensorShape(1,rows,768),"input");
            var flat=Functional.Reshape(input,new[]{rows,768});
            return graph.Compile(flat*Functional.RSqrt(Functional.ReduceMean(Functional.Square(flat),1,true)+1e-6f));
        }
        // Shape/rebinding contract only; this synthetic scale is not a numerical RMS reference.
        static Model Template(int rows,int width=768)
        {
            var graph=new FunctionalGraph();var input=graph.AddInput<float>(new TensorShape(rows,width),"input");
            return graph.Compile(input*.5f);
        }
        [TestCase(252)] [TestCase(256)]
        public void TailTemplateMustMatchActualIntermediateShapeAndPreserveBorrowedGraph(int rows)
        {
            var source=Source(rows);var template=Template(rows);
            var layers=source.layers.ToArray();var constants=source.constants.ToArray();var templateInputs=template.inputs.ToArray();
            Assert.That(source.layers.Count(l=>l.GetType().Name=="Square"),Is.EqualTo(1),"Synthetic graph must exercise the original tail chain.");
            var result=QualifiedVisionRmsOperation.InlineTailRms(source,template);
            Assert.That(result.layers.Any(l=>l.GetType().Name=="Square"),Is.False);
            Assert.That(result.inputs[0].shape.ToTensorShape(),Is.EqualTo(new TensorShape(1,rows,768)));
            Assert.That(result.outputs[0].index,Is.EqualTo(source.outputs[0].index));
            Assert.That(source.layers.ToArray(),Is.EqualTo(layers));Assert.That(source.constants.ToArray(),Is.EqualTo(constants));
            Assert.That(template.inputs.ToArray(),Is.EqualTo(templateInputs));
        }
        [TestCase(252,256)] [TestCase(256,252)]
        public void WrongProfileTailTemplateMustBeRejectedBeforeGraphReplacement(int rows,int templateRows)
        {
            var source=Source(rows);var layers=source.layers.ToArray();
            Assert.Throws<ArgumentException>(()=>QualifiedVisionRmsOperation.InlineTailRms(source,Template(templateRows)));
            Assert.That(source.layers.ToArray(),Is.EqualTo(layers));
        }
        [Test]
        public void WrongWidthTailTemplateMustBeRejectedBeforeGraphReplacement()
            =>Assert.Throws<ArgumentException>(()=>QualifiedVisionRmsOperation.InlineTailRms(Source(252),Template(252,767)));
    }
}
