using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Unity.InferenceEngine;

namespace EmbeddingGemma.Validation.Tests
{
    /// <summary>Fixed first-RMS diagnostic operation; owns Workers/division, borrows the audited template.</summary>
    public sealed class QualifiedVisionRmsOperation : IGpuTensorOperation
    {
        GpuProjectionPipeline pipeline;
        GpuScalarDivide divide;
        public BackendType Backend=>BackendType.GPUCompute;
        public QualifiedVisionRmsOperation(Model model)
        {
            if(model==null)throw new ArgumentNullException(nameof(model));
            if(model.inputs.Count!=1||!model.inputs[0].shape.IsStatic()||model.inputs[0].shape.ToTensorShape()!=new TensorShape(1,2520,768))
                throw new ArgumentException("Expected the qualified fixed first vision RMS template.",nameof(model));
            var scalar=model.layers.Single(layer=>layer.GetType().Name=="ScalarMad"
                &&(float)layer.GetType().GetField("sFloat").GetValue(layer)==1f/768f
                &&(float)layer.GetType().GetField("bFloat").GetValue(layer)==1e-6f);
            var before=ModelTensorSlice.Extract(model,scalar.inputs[0],"total");
            var corrected=(Layer)typeof(object).GetMethod("MemberwiseClone",BindingFlags.Instance|BindingFlags.NonPublic).Invoke(scalar,null);
            corrected.GetType().GetField("sFloat").SetValue(corrected,1f);
            var changed=new Model{ProducerName=model.ProducerName+"/explicit-mean-division",inputs=new List<Model.Input>(model.inputs),
                outputs=new List<Model.Output>(model.outputs),constants=new List<Constant>(model.constants),
                layers=model.layers.Select(layer=>ReferenceEquals(layer,scalar)?corrected:layer).ToList()};
            var after=ModelTensorSlice.Extract(changed,changed.outputs[0].index,"normalized",new[]{new Model.Input{
                name="mean",index=scalar.inputs[0],dataType=DataType.Float,shape=new DynamicTensorShape(1,2520,1)}});
            try
            {
                divide=new GpuScalarDivide(768f);
                pipeline=new GpuProjectionPipeline(new[]{before,after},new[]{"total"},new[]{"mean"},new IGpuTensorOperation[]{divide});
            }
            catch{Dispose();throw;}
        }
        public Tensor<float> Execute(Tensor<float> input)
        {
            if(pipeline==null)throw new ObjectDisposedException(nameof(QualifiedVisionRmsOperation));
            return pipeline.Execute(input);
        }
        public void Dispose(){pipeline?.Dispose();pipeline=null;divide?.Dispose();divide=null;}
    }
}
