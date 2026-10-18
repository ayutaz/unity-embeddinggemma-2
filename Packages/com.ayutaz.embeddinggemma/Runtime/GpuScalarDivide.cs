using System;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.Rendering;

namespace EmbeddingGemma
{
    /// <summary>Experimental rounded FP32 scalar division on resident GPU tensors. Fixed references qualify rounding per device.</summary>
    public sealed class GpuScalarDivide : IGpuTensorOperation
    {
        ComputeShader shader;
        readonly int kernel;
        readonly float divisor;
        bool disposed;
        public BackendType Backend => BackendType.GPUCompute;
        public GpuScalarDivide(float divisor)
        {
            if(divisor==0||float.IsNaN(divisor)||float.IsInfinity(divisor))throw new ArgumentException("Expected a finite nonzero FP32 divisor.",nameof(divisor));
            if(!SystemInfo.supportsComputeShaders)throw new NotSupportedException("Rounded scalar division requires an actual compute GPU.");
            this.divisor=divisor;var resource=Resources.Load<ComputeShader>("EmbeddingGemmaScalarDivide");
            if(resource==null)throw new InvalidOperationException("GPU scalar division shader is missing.");
            try{shader=UnityEngine.Object.Instantiate(resource);kernel=shader.FindKernel("RoundedDivide");}
            catch{Dispose();throw;}
        }
        public Tensor<float> Execute(Tensor<float> input)
        {
            if(disposed)throw new ObjectDisposedException(nameof(GpuScalarDivide));
            if(input==null)throw new ArgumentNullException(nameof(input));
            if(input.dataOnBackend==null)throw new ObjectDisposedException(nameof(input));
            if(input.backendType!=BackendType.GPUCompute||!(input.dataOnBackend is ComputeTensorData)||input.shape.length==0)
                throw new ArgumentException("Expected a nonempty tensor already on GPUCompute; no implicit transfer.");
            var output=GpuOutputTensor.Allocate(input.shape);
            try
            {
                var destination=ComputeTensorData.Pin(output);var groups=(output.shape.length+63L)/64;var groupsX=(int)Math.Min(65535,groups);var groupsY=(int)((groups+groupsX-1)/groupsX);
                using var commands=new CommandBuffer{name="EmbeddingGemma rounded FP32 GPU scalar division"};
                commands.SetComputeIntParam(shader,"Count",input.shape.length);commands.SetComputeIntParam(shader,"DispatchWidth",groupsX*64);commands.SetComputeFloatParam(shader,"Divisor",divisor);
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
