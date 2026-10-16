using System;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.Rendering;

namespace EmbeddingGemma
{
    /// <summary>
    /// Experimental main-thread GPU projection with block192 accumulation.
    /// Copies coefficients at construction, borrows GPU inputs, returns owned GPU outputs.
    /// Native MAD rounding must be qualified on each supported GPU; no CPU fallback.
    /// </summary>
    public sealed class GpuBlockedMatmul : IGpuTensorOperation
    {
        readonly int width, columns, blockSize;
        ComputeBuffer weights;
        ComputeShader shader;
        int kernel;
        bool disposed;
        public BackendType Backend => BackendType.GPUCompute;

        public GpuBlockedMatmul(int width,int columns,float[] coefficients)
            :this(width,columns,coefficients,192) { }

        /// <summary>Explicit reference-qualified accumulation order; existing callers retain block192.</summary>
        public GpuBlockedMatmul(int width,int columns,float[] coefficients,int blockSize)
        {
            if(blockSize<1 || blockSize>4096)throw new ArgumentOutOfRangeException(nameof(blockSize),"Expected a bounded accumulation block from 1 to 4096.");
            if(width<1 || columns<1 || coefficients==null || (long)width*columns!=coefficients.Length)
                throw new ArgumentException("Expected a nonempty fixed KxN coefficient matrix.");
            foreach(var value in coefficients)
                if(float.IsNaN(value) || float.IsInfinity(value) || Math.Abs(value)>100)
                    throw new ArgumentException("Expected bounded finite FP32 coefficients.",nameof(coefficients));
            if(!SystemInfo.supportsComputeShaders)
                throw new NotSupportedException("This projection requires an actual compute GPU.");
            this.width=width;this.columns=columns;this.blockSize=blockSize;
            try
            {
                var resource=Resources.Load<ComputeShader>("EmbeddingGemmaBlockedMatmul");
                if(resource==null)throw new InvalidOperationException("GPU projection shader is missing.");
                shader=UnityEngine.Object.Instantiate(resource);
                kernel=shader.FindKernel(blockSize==192?"Block192":"VariableBlock");
                weights=new ComputeBuffer(coefficients.Length,sizeof(float));
                weights.SetData(coefficients);
            }
            catch { Dispose();throw; }
        }

        public Tensor<float> Execute(Tensor<float> input)
        {
            if(disposed)throw new ObjectDisposedException(nameof(GpuBlockedMatmul));
            if(input==null)throw new ArgumentNullException(nameof(input));
            if(input.dataOnBackend==null)throw new ObjectDisposedException(nameof(input));
            if(input.backendType!=BackendType.GPUCompute || !(input.dataOnBackend is ComputeTensorData))
                throw new ArgumentException("Input must already be on GPUCompute; no implicit transfer.",nameof(input));
            var rank=input.shape.rank;
            if((rank!=2 && rank!=3) || input.shape[rank-1]!=width || input.shape.length==0)
                throw new ArgumentException("Expected a nonempty matching fixed-width matrix.",nameof(input));
            var shape=rank==2?new TensorShape(input.shape[0],columns):new TensorShape(input.shape[0],input.shape[1],columns);
            var output=new Tensor<float>(shape);
            try
            {
                var source=(ComputeTensorData)input.dataOnBackend;
                var destination=ComputeTensorData.Pin(output);
                var groups=(output.shape.length+63L)/64;
                var groupsX=(int)Math.Min(65535,groups);
                var groupsY=(int)((groups+groupsX-1)/groupsX);
                using var commands=new CommandBuffer { name="EmbeddingGemma block192 GPU projection" };
                commands.SetComputeIntParam(shader,"Width",width);
                commands.SetComputeIntParam(shader,"Columns",columns);
                commands.SetComputeIntParam(shader,"Rows",input.shape.length/width);
                commands.SetComputeIntParam(shader,"DispatchWidth",groupsX*64);
                if(blockSize!=192)commands.SetComputeIntParam(shader,"BlockSize",blockSize);
                commands.SetComputeBufferParam(shader,kernel,"Input",source.buffer);
                commands.SetComputeBufferParam(shader,kernel,"Weights",weights);
                commands.SetComputeBufferParam(shader,kernel,"Output",destination.buffer);
                commands.DispatchCompute(shader,kernel,groupsX,groupsY,1);
                Graphics.ExecuteCommandBuffer(commands);
                return output;
            }
            catch { output.Dispose();throw; }
        }

        public void Dispose()
        {
            if(disposed)return;
            disposed=true;weights?.Dispose();weights=null;
            if(shader!=null)
            {
                if(Application.isPlaying)UnityEngine.Object.Destroy(shader);
                else UnityEngine.Object.DestroyImmediate(shader);
                shader=null;
            }
        }
    }
}
