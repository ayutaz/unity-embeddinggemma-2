using System;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.Rendering;

namespace EmbeddingGemma
{
    /// <summary>Two separately rounded FP32 multiplies on GPU, preserving native query scale ordering.</summary>
    public sealed class GpuSequentialScale : IGpuTensorOperation
    {
        ComputeShader shader;
        ComputeBuffer coefficients;
        readonly int width,kernel;
        readonly float firstScale;
        bool disposed;
        public BackendType Backend => BackendType.GPUCompute;
        public GpuSequentialScale(float firstScale,float[] secondScale)
        {
            if(float.IsNaN(firstScale)||float.IsInfinity(firstScale)||secondScale==null||secondScale.Length==0)
                throw new ArgumentException("Expected finite scalar and nonempty scale vector.");
            foreach(var value in secondScale)if(float.IsNaN(value)||float.IsInfinity(value))throw new ArgumentException("Scale vector must be finite.");
            if(!SystemInfo.supportsComputeShaders)throw new NotSupportedException("Sequential scale requires an actual compute GPU.");
            this.firstScale=firstScale;width=secondScale.Length;
            try
            {
                var resource=Resources.Load<ComputeShader>("EmbeddingGemmaSequentialScale");
                if(resource==null)throw new InvalidOperationException("GPU sequential scale shader is missing.");
                shader=UnityEngine.Object.Instantiate(resource);kernel=shader.FindKernel("SequentialScale");
                coefficients=new ComputeBuffer(width,sizeof(float));coefficients.SetData(secondScale);
            }
            catch{Dispose();throw;}
        }
        public Tensor<float> Execute(Tensor<float> input)
        {
            if(disposed)throw new ObjectDisposedException(nameof(GpuSequentialScale));
            if(input==null)throw new ArgumentNullException(nameof(input));
            if(input.dataOnBackend==null)throw new ObjectDisposedException(nameof(input));
            if(input.backendType!=BackendType.GPUCompute||!(input.dataOnBackend is ComputeTensorData)||input.shape.length==0||input.shape.rank<1||input.shape[input.shape.rank-1]!=width)
                throw new ArgumentException("Expected a nonempty matching-width tensor already on GPUCompute.");
            var output=GpuOutputTensor.Allocate(input.shape);
            try
            {
                var destination=ComputeTensorData.Pin(output);var groups=(output.shape.length+63L)/64;var groupsX=(int)Math.Min(65535,groups);var groupsY=(int)((groups+groupsX-1)/groupsX);
                using var commands=new CommandBuffer{name="EmbeddingGemma separately rounded GPU query scaling"};
                commands.SetComputeIntParam(shader,"Count",output.shape.length);commands.SetComputeIntParam(shader,"Width",width);commands.SetComputeIntParam(shader,"DispatchWidth",groupsX*64);
                commands.SetComputeFloatParam(shader,"FirstScale",firstScale);commands.SetComputeBufferParam(shader,kernel,"Input",((ComputeTensorData)input.dataOnBackend).buffer);
                commands.SetComputeBufferParam(shader,kernel,"Coefficients",coefficients);commands.SetComputeBufferParam(shader,kernel,"Output",destination.buffer);
                commands.DispatchCompute(shader,kernel,groupsX,groupsY,1);Graphics.ExecuteCommandBuffer(commands);return output;
            }
            catch{output.Dispose();throw;}
        }
        public void Dispose()
        {
            if(disposed)return;disposed=true;coefficients?.Dispose();coefficients=null;
            if(shader!=null){if(Application.isPlaying)UnityEngine.Object.Destroy(shader);else UnityEngine.Object.DestroyImmediate(shader);shader=null;}
        }
    }
}
