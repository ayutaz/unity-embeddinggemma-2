using System;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.Rendering;

namespace EmbeddingGemma
{
    /// <summary>Experimental last-axis FP32 softmax. Defaults to eight-lane native order; sixteen-lane order is explicit. Fixed inputs qualify rounding per GPU.</summary>
    public sealed class GpuSoftmax : IGpuTensorOperation
    {
        ComputeShader shader;
        readonly int kernel;
        bool disposed;
        public BackendType Backend => BackendType.GPUCompute;
        public GpuSoftmax()
            :this(8) { }
        public GpuSoftmax(int vectorWidth)
        {
            if(vectorWidth!=8&&vectorWidth!=16)throw new ArgumentOutOfRangeException(nameof(vectorWidth),"Expected eight or sixteen FP32 reduction lanes.");
            if(!SystemInfo.supportsComputeShaders)throw new NotSupportedException("Softmax requires an actual compute GPU.");
            var resource=Resources.Load<ComputeShader>("EmbeddingGemmaNativeSoftmax");
            if(resource==null)throw new InvalidOperationException("GPU softmax shader is missing.");
            try{shader=UnityEngine.Object.Instantiate(resource);kernel=shader.FindKernel(vectorWidth==8?"NativeSoftmax":"NativeSoftmax16");}
            catch{Dispose();throw;}
        }
        public Tensor<float> Execute(Tensor<float> input)
        {
            if(disposed)throw new ObjectDisposedException(nameof(GpuSoftmax));
            if(input==null)throw new ArgumentNullException(nameof(input));
            if(input.dataOnBackend==null)throw new ObjectDisposedException(nameof(input));
            if(input.backendType!=BackendType.GPUCompute||!(input.dataOnBackend is ComputeTensorData)||input.shape.length==0||input.shape.rank<1)
                throw new ArgumentException("Expected a nonempty rank 1..8 tensor already on GPUCompute; no implicit transfer.");
            var width=input.shape[input.shape.rank-1];var rows=input.shape.length/width;var output=new Tensor<float>(input.shape);
            try
            {
                var destination=ComputeTensorData.Pin(output);var groups=(rows+63L)/64;var groupsX=(int)Math.Min(65535,groups);var groupsY=(int)((groups+groupsX-1)/groupsX);
                using var commands=new CommandBuffer{name="EmbeddingGemma native-order GPU softmax"};
                commands.SetComputeIntParam(shader,"Rows",rows);commands.SetComputeIntParam(shader,"Width",width);commands.SetComputeIntParam(shader,"DispatchWidth",groupsX*64);
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
