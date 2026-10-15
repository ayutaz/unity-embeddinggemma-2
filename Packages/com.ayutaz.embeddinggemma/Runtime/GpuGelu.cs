using System;
using Unity.InferenceEngine;
using UnityEngine;

namespace EmbeddingGemma
{
    /// <summary>
    /// Experimental fixed-shape FP32 tanh GELU. Owns two GPU Workers and tanh;
    /// borrows resident input and returns independent caller-owned GPU results.
    /// Fixed reference gates qualify numerical behavior on each supported GPU.
    /// </summary>
    public sealed class GpuGelu : IGpuTensorOperation
    {
        GpuProjectionPipeline pipeline;
        GpuTanh tanh;
        public BackendType Backend=>BackendType.GPUCompute;
        public GpuGelu(TensorShape shape)
            : this(shape,false) { }

        /// <summary>Optionally retires copied internal Worker outputs and releases unused GPU caches after each boundary.</summary>
        public GpuGelu(TensorShape shape,bool releaseUnusedGpuCache)
        {
            if(shape.rank<1||shape.length==0)throw new ArgumentException("Expected a nonempty fixed GELU shape.",nameof(shape));
            if(!SystemInfo.supportsComputeShaders)throw new NotSupportedException("GELU requires an actual compute GPU.");
            var first=new FunctionalGraph();var input=first.AddInput<float>(shape,"input");
            var before=first.Compile(.7978845608028654f*(input+.044715f*(input*input*input)));
            var boundary=before.outputs[0];boundary.name="tanh_input";before.outputs[0]=boundary;
            var last=new FunctionalGraph();var original=last.AddInput<float>(shape,"input");var activated=last.AddInput<float>(shape,"tanh_output");
            var after=last.Compile((.5f*original)*(1f+activated));
            try
            {
                tanh=new GpuTanh();
                pipeline=new GpuProjectionPipeline(new[]{before,after},new[]{new[]{"tanh_input"}},new[]{"tanh_output"},new IGpuTensorOperation[]{tanh},releaseUnusedGpuCache);
            }
            catch{Dispose();throw;}
        }
        public Tensor<float> Execute(Tensor<float> input)
        {
            if(pipeline==null)throw new ObjectDisposedException(nameof(GpuGelu));
            return pipeline.Execute(input);
        }
        public void Dispose(){pipeline?.Dispose();pipeline=null;tanh?.Dispose();tanh=null;}
    }
}
