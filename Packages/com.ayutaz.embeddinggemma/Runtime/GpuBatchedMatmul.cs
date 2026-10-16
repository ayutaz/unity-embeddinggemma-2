using System;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.Rendering;

namespace EmbeddingGemma
{
    /// <summary>
    /// Experimental FP32 batch-broadcast matmul. Borrows already resident GPU
    /// inputs, returns an owned GPU output, and uses block192 native MAD ordering.
    /// Fixed-reference GPU gates qualify rounding on each supported device.
    /// </summary>
    public sealed class GpuBatchedMatmul : IGpuBinaryTensorOperation
    {
        ComputeShader shader;
        readonly int kernel;
        bool disposed;
        public BackendType Backend => BackendType.GPUCompute;
        public GpuBatchedMatmul()
        {
            if(!SystemInfo.supportsComputeShaders)throw new NotSupportedException("Batched matmul requires an actual compute GPU.");
            var resource=Resources.Load<ComputeShader>("EmbeddingGemmaBatchedMatmul");
            if(resource==null)throw new InvalidOperationException("GPU batched matmul shader is missing.");
            try{shader=UnityEngine.Object.Instantiate(resource);kernel=shader.FindKernel("Block192");}
            catch{Dispose();throw;}
        }
        public Tensor<float> Execute(Tensor<float> input)
            =>throw new NotSupportedException("Batched matmul requires both GPU operands.");

        static void Validate(Tensor<float> input)
        {
            if(input==null)throw new ArgumentNullException(nameof(input));
            if(input.dataOnBackend==null)throw new ObjectDisposedException(nameof(input));
            if(input.backendType!=BackendType.GPUCompute || !(input.dataOnBackend is ComputeTensorData) || input.shape.length==0 || input.shape.rank<2 || input.shape.rank>8)
                throw new ArgumentException("Expected a nonempty rank 2..8 tensor already on GPUCompute.");
        }
        void Vector(CommandBuffer commands,string name,int[] values,int offset)
            =>commands.SetComputeIntParams(shader,name,new[]{values[offset],values[offset+1],values[offset+2],values[offset+3]});

        public Tensor<float> Execute(Tensor<float> left,Tensor<float> right)
        {
            if(disposed)throw new ObjectDisposedException(nameof(GpuBatchedMatmul));
            Validate(left);Validate(right);
            var a=left.shape;var b=right.shape;int width=a[a.rank-1],rows=a[a.rank-2],columns=b[b.rank-1];
            if(width!=b[b.rank-2])throw new ArgumentException("Matrix inner dimensions differ.");
            int rank=Math.Max(a.rank,b.rank),batchRank=rank-2;var dimensions=new int[rank];var packed=new int[8];var strideA=new int[8];var strideB=new int[8];
            int runningA=checked(rows*width),runningB=checked(width*columns);
            for(var axis=batchRank-1;axis>=0;axis--)
            {
                var indexA=axis-(rank-a.rank);var indexB=axis-(rank-b.rank);var dimA=indexA<0?1:a[indexA];var dimB=indexB<0?1:b[indexB];
                if(dimA!=dimB && dimA!=1 && dimB!=1)throw new ArgumentException("Matrix batch dimensions cannot broadcast.");
                dimensions[axis]=Math.Max(dimA,dimB);packed[axis]=dimensions[axis];strideA[axis]=dimA==1?0:runningA;strideB[axis]=dimB==1?0:runningB;
                runningA=checked(runningA*dimA);runningB=checked(runningB*dimB);
            }
            dimensions[rank-2]=rows;dimensions[rank-1]=columns;var output=new Tensor<float>(new TensorShape(dimensions));
            try
            {
                var destination=ComputeTensorData.Pin(output);var groups=(output.shape.length+63L)/64;var groupsX=(int)Math.Min(65535,groups);var groupsY=(int)((groups+groupsX-1)/groupsX);
                using var commands=new CommandBuffer{name="EmbeddingGemma native-order GPU batched matmul"};
                commands.SetComputeIntParam(shader,"Rows",rows);commands.SetComputeIntParam(shader,"Columns",columns);commands.SetComputeIntParam(shader,"Width",width);
                commands.SetComputeIntParam(shader,"BatchRank",batchRank);commands.SetComputeIntParam(shader,"Count",output.shape.length);commands.SetComputeIntParam(shader,"DispatchWidth",groupsX*64);
                Vector(commands,"BatchShapeLo",packed,0);Vector(commands,"BatchShapeHi",packed,4);Vector(commands,"StrideALo",strideA,0);Vector(commands,"StrideAHi",strideA,4);Vector(commands,"StrideBLo",strideB,0);Vector(commands,"StrideBHi",strideB,4);
                commands.SetComputeBufferParam(shader,kernel,"Left",((ComputeTensorData)left.dataOnBackend).buffer);commands.SetComputeBufferParam(shader,kernel,"Right",((ComputeTensorData)right.dataOnBackend).buffer);
                commands.SetComputeBufferParam(shader,kernel,"Output",destination.buffer);commands.DispatchCompute(shader,kernel,groupsX,groupsY,1);Graphics.ExecuteCommandBuffer(commands);return output;
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
