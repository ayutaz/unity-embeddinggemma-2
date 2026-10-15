using System;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.Rendering;

namespace EmbeddingGemma
{
    /// <summary>
    /// Experimental fixed-shape FP32 tanh GELU. Owns its GPU resources;
    /// borrows resident input and returns independent caller-owned GPU results.
    /// Fixed reference gates qualify numerical behavior on each supported GPU.
    /// </summary>
    public sealed class GpuGelu : IGpuTensorOperation
    {
        GpuProjectionPipeline pipeline;
        GpuTanh tanh;
        ComputeShader nativeShader;
        readonly int nativeKernel;
        readonly TensorShape fixedShape;
        readonly bool fuseNativeInner;
        public BackendType Backend=>BackendType.GPUCompute;
        public GpuGelu(TensorShape shape)
            : this(shape,false) { }

        /// <summary>Optionally retires copied internal Worker outputs and releases unused GPU caches after each boundary.</summary>
        public GpuGelu(TensorShape shape,bool releaseUnusedGpuCache)
            : this(shape,releaseUnusedGpuCache,false) { }

        /// <summary>Explicitly opts into one-rounding kappa*cube+input; all arithmetic remains FP32 on GPU.</summary>
        public GpuGelu(TensorShape shape,bool releaseUnusedGpuCache,bool fuseInner)
            :this(shape,releaseUnusedGpuCache,fuseInner,false) { }

        /// <summary>Explicitly selects an ordered FP32 kernel with paired-exponential SLEEF tanh.</summary>
        public GpuGelu(TensorShape shape,bool releaseUnusedGpuCache,bool fuseInner,bool sleefTanh)
        {
            if(shape.rank<1||shape.length==0)throw new ArgumentException("Expected a nonempty fixed GELU shape.",nameof(shape));
            if(!SystemInfo.supportsComputeShaders)throw new NotSupportedException("GELU requires an actual compute GPU.");
            fixedShape=shape;fuseNativeInner=fuseInner;
            if(sleefTanh)
            {
                // The graph compiler changes rounding inside the compensated prelude.
                // Keep explicit FMA and non-FMA boundaries in one precise FP32 kernel.
                var resource=Resources.Load<ComputeShader>("EmbeddingGemmaNativeTanh");
                if(resource==null)throw new InvalidOperationException("GPU GELU shader is missing.");
                try{nativeShader=UnityEngine.Object.Instantiate(resource);nativeKernel=nativeShader.FindKernel("SleefGelu");}
                catch{Dispose();throw;}
                return;
            }
            var first=new FunctionalGraph();var input=first.AddInput<float>(shape,"input");
            var inner=input+.044715f*(input*input*input);
            if(fuseInner)
            {
                // Bounded FP32 Dekker/TwoSum, as in the qualified export helper.
                // This is not a general IEEE FMA implementation for extreme inputs.
                const float a=.044715f;var cube=input*input*input;var product=a*cube;
                var aSplit=a*4097f;var aHigh=aSplit-(aSplit-a);var aLow=a-aHigh;
                var bSplit=cube*4097f;var bHigh=bSplit-(bSplit-cube);var bLow=cube-bHigh;
                var productError=((aHigh*bHigh-product)+aHigh*bLow+aLow*bHigh)+aLow*bLow;
                var total=product+input;var recovered=total-product;
                var sumError=(product-(total-recovered))+(input-recovered);
                inner=total+(productError+sumError);
            }
            var before=first.Compile(.7978845608028654f*inner);
            var boundary=before.outputs[0];boundary.name="tanh_input";before.outputs[0]=boundary;
            var last=new FunctionalGraph();var original=last.AddInput<float>(shape,"input");var activated=last.AddInput<float>(shape,"tanh_output");
            var after=last.Compile((.5f*original)*(1f+activated));
            try
            {
                tanh=new GpuTanh(sleefTanh);
                pipeline=new GpuProjectionPipeline(new[]{before,after},new[]{new[]{"tanh_input"}},new[]{"tanh_output"},new IGpuTensorOperation[]{tanh},releaseUnusedGpuCache);
            }
            catch{Dispose();throw;}
        }
        public Tensor<float> Execute(Tensor<float> input)
        {
            if(nativeShader!=null)return ExecuteNative(input);
            if(pipeline==null)throw new ObjectDisposedException(nameof(GpuGelu));
            return pipeline.Execute(input);
        }
        Tensor<float> ExecuteNative(Tensor<float> input)
        {
            if(input==null)throw new ArgumentNullException(nameof(input));
            if(input.dataOnBackend==null)throw new ObjectDisposedException(nameof(input));
            if(input.shape!=fixedShape||input.backendType!=BackendType.GPUCompute||!(input.dataOnBackend is ComputeTensorData))
                throw new ArgumentException("Expected the fixed tensor shape already on GPUCompute; no implicit transfer.");
            var output=new Tensor<float>(input.shape);
            try
            {
                var destination=ComputeTensorData.Pin(output);var groups=(input.shape.length+63L)/64;var groupsX=(int)Math.Min(65535,groups);var groupsY=(int)((groups+groupsX-1)/groupsX);
                using var commands=new CommandBuffer{name="EmbeddingGemma ordered native GPU GELU"};
                commands.SetComputeIntParam(nativeShader,"Count",input.shape.length);commands.SetComputeIntParam(nativeShader,"DispatchWidth",groupsX*64);
                commands.SetComputeIntParam(nativeShader,"FuseGeluInner",fuseNativeInner?1:0);
                commands.SetComputeBufferParam(nativeShader,nativeKernel,"Input",((ComputeTensorData)input.dataOnBackend).buffer);
                commands.SetComputeBufferParam(nativeShader,nativeKernel,"Output",destination.buffer);
                commands.DispatchCompute(nativeShader,nativeKernel,groupsX,groupsY,1);Graphics.ExecuteCommandBuffer(commands);return output;
            }
            catch{output.Dispose();throw;}
        }
        public void Dispose()
        {
            pipeline?.Dispose();pipeline=null;tanh?.Dispose();tanh=null;
            if(nativeShader!=null){if(Application.isPlaying)UnityEngine.Object.Destroy(nativeShader);else UnityEngine.Object.DestroyImmediate(nativeShader);nativeShader=null;}
        }
    }
}
