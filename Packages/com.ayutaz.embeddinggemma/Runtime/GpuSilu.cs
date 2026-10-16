using System;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.Rendering;

namespace EmbeddingGemma
{
    /// <summary>
    /// Experimental FP32 SiLU on GPU. Borrows a GPU tensor and returns an owned
    /// tensor of the same shape. The source-derived exp rounding is qualified
    /// against bounded fixed inputs on each GPU; it is not a portable bit guarantee.
    /// </summary>
    public sealed class GpuSilu : IGpuTensorOperation
    {
        ComputeShader shader;
        readonly int kernel;
        bool disposed;
        public BackendType Backend => BackendType.GPUCompute;

        public GpuSilu()
        {
            if(!SystemInfo.supportsComputeShaders)throw new NotSupportedException("SiLU requires an actual compute GPU.");
            var resource=Resources.Load<ComputeShader>("EmbeddingGemmaNativeSilu");
            if(resource==null)throw new InvalidOperationException("GPU SiLU shader is missing.");
            try {shader=UnityEngine.Object.Instantiate(resource);kernel=shader.FindKernel("NativeSilu");}
            catch {Dispose();throw;}
        }

        public Tensor<float> Execute(Tensor<float> input)
        {
            if(disposed)throw new ObjectDisposedException(nameof(GpuSilu));
            if(input==null)throw new ArgumentNullException(nameof(input));
            if(input.dataOnBackend==null)throw new ObjectDisposedException(nameof(input));
            if(input.backendType!=BackendType.GPUCompute || !(input.dataOnBackend is ComputeTensorData) || input.shape.length==0)
                throw new ArgumentException("Expected a nonempty tensor already on GPUCompute; no implicit transfer.",nameof(input));
            var output=new Tensor<float>(input.shape);
            try
            {
                var source=(ComputeTensorData)input.dataOnBackend;var destination=ComputeTensorData.Pin(output);
                var groups=(input.shape.length+63L)/64;var groupsX=(int)Math.Min(65535,groups);var groupsY=(int)((groups+groupsX-1)/groupsX);
                using var commands=new CommandBuffer{name="EmbeddingGemma native-rounding GPU SiLU"};
                commands.SetComputeIntParam(shader,"Count",input.shape.length);commands.SetComputeIntParam(shader,"DispatchWidth",groupsX*64);
                commands.SetComputeBufferParam(shader,kernel,"Input",source.buffer);commands.SetComputeBufferParam(shader,kernel,"Output",destination.buffer);
                commands.DispatchCompute(shader,kernel,groupsX,groupsY,1);Graphics.ExecuteCommandBuffer(commands);return output;
            }
            catch {output.Dispose();throw;}
        }

        public void Dispose()
        {
            if(disposed)return;disposed=true;
            if(shader!=null){if(Application.isPlaying)UnityEngine.Object.Destroy(shader);else UnityEngine.Object.DestroyImmediate(shader);shader=null;}
        }
    }
}
