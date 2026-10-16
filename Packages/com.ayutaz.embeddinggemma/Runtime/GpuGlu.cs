using System;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.Rendering;

namespace EmbeddingGemma
{
    /// <summary>Experimental native-order last-axis FP32 GLU. Borrows resident GPU input and returns an owned tensor with half the last width.</summary>
    public sealed class GpuGlu : IGpuTensorOperation
    {
        ComputeShader shader;
        readonly int kernel;
        bool disposed;
        public BackendType Backend => BackendType.GPUCompute;
        public GpuGlu()
        {
            if(!SystemInfo.supportsComputeShaders)throw new NotSupportedException("GLU requires an actual compute GPU.");
            var resource=Resources.Load<ComputeShader>("EmbeddingGemmaNativeGlu");
            if(resource==null)throw new InvalidOperationException("GPU GLU shader is missing.");
            try{shader=UnityEngine.Object.Instantiate(resource);kernel=shader.FindKernel("NativeGlu");}
            catch{Dispose();throw;}
        }
        public Tensor<float> Execute(Tensor<float> input)
        {
            if(disposed)throw new ObjectDisposedException(nameof(GpuGlu));
            if(input==null)throw new ArgumentNullException(nameof(input));
            if(input.dataOnBackend==null)throw new ObjectDisposedException(nameof(input));
            if(input.backendType!=BackendType.GPUCompute||!(input.dataOnBackend is ComputeTensorData)||input.shape.length==0||input.shape.rank<1||input.shape[input.shape.rank-1]%2!=0)
                throw new ArgumentException("Expected a nonempty even last-axis tensor already on GPUCompute; no implicit transfer.");
            var dimensions=new int[input.shape.rank];for(var axis=0;axis<dimensions.Length;axis++)dimensions[axis]=input.shape[axis];dimensions[dimensions.Length-1]/=2;
            var output=new Tensor<float>(new TensorShape(dimensions));
            try
            {
                var destination=ComputeTensorData.Pin(output);var groups=(output.shape.length+63L)/64;var groupsX=(int)Math.Min(65535,groups);var groupsY=(int)((groups+groupsX-1)/groupsX);
                using var commands=new CommandBuffer{name="EmbeddingGemma native-order GPU GLU"};
                commands.SetComputeIntParam(shader,"Count",output.shape.length);commands.SetComputeIntParam(shader,"Columns",dimensions[dimensions.Length-1]);commands.SetComputeIntParam(shader,"DispatchWidth",groupsX*64);
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
