using System;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.Rendering;

namespace EmbeddingGemma
{
    /// <summary>
    /// Experimental native-rounding FP32 tanh using paired FP32 residuals.
    /// Borrows an already resident GPU tensor and returns an independently owned output.
    /// Fixed reference results qualify rounding on each GPU, not all devices or inputs.
    /// </summary>
    public sealed class GpuTanh : IGpuTensorOperation
    {
        ComputeShader shader;
        readonly int kernel;
        bool disposed;
        public BackendType Backend => BackendType.GPUCompute;
        public GpuTanh()
            :this(false) { }
        /// <summary>Explicit paired-exponential SLEEF candidate; the existing table remains the default.</summary>
        public GpuTanh(bool useSleef)
        {
            if(!SystemInfo.supportsComputeShaders)throw new NotSupportedException("Tanh requires an actual compute GPU.");
            var resource=Resources.Load<ComputeShader>("EmbeddingGemmaNativeTanh");
            if(resource==null)throw new InvalidOperationException("GPU tanh shader is missing.");
            try{shader=UnityEngine.Object.Instantiate(resource);kernel=shader.FindKernel(useSleef?"SleefTanh":"NativeTanh");}
            catch{Dispose();throw;}
        }
        public Tensor<float> Execute(Tensor<float> input)
        {
            if(disposed)throw new ObjectDisposedException(nameof(GpuTanh));
            if(input==null)throw new ArgumentNullException(nameof(input));
            if(input.dataOnBackend==null)throw new ObjectDisposedException(nameof(input));
            if(input.backendType!=BackendType.GPUCompute||!(input.dataOnBackend is ComputeTensorData)||input.shape.length==0)
                throw new ArgumentException("Expected a nonempty tensor already on GPUCompute; no implicit transfer.");
            var output=new Tensor<float>(input.shape);
            try
            {
                var destination=ComputeTensorData.Pin(output);var groups=(output.shape.length+63L)/64;var groupsX=(int)Math.Min(65535,groups);var groupsY=(int)((groups+groupsX-1)/groupsX);
                using var commands=new CommandBuffer{name="EmbeddingGemma native-rounding GPU tanh"};
                commands.SetComputeIntParam(shader,"Count",input.shape.length);commands.SetComputeIntParam(shader,"DispatchWidth",groupsX*64);
                commands.SetComputeBufferParam(shader,kernel,"Input",((ComputeTensorData)input.dataOnBackend).buffer);commands.SetComputeBufferParam(shader,kernel,"Output",destination.buffer);
                commands.DispatchCompute(shader,kernel,groupsX,groupsY,1);Graphics.ExecuteCommandBuffer(commands);return output;
            }
            catch{output.Dispose();throw;}
        }
        public void Dispose()
        {
            if(disposed)return;disposed=true;
            if(shader!=null){if(Application.isPlaying)UnityEngine.Object.Destroy(shader);else UnityEngine.Object.DestroyImmediate(shader);shader=null;}
        }
    }
}
