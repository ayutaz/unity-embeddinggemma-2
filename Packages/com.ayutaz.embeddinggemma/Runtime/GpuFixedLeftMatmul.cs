using System;
using Unity.InferenceEngine;
using UnityEngine;

namespace EmbeddingGemma
{
    /// <summary>
    /// Owns fixed rank-2 or batch-one rank-3 left coefficients and uses the
    /// qualified block192 GPU matmul. Borrows GPU right operands and transfers
    /// ownership of each output to the caller. No implicit CPU execution.
    /// </summary>
    public sealed class GpuFixedLeftMatmul : IGpuTensorOperation
    {
        Tensor<float> coefficients;
        GpuBatchedMatmul operation;
        bool disposed;
        public BackendType Backend=>BackendType.GPUCompute;

        public GpuFixedLeftMatmul(TensorShape shape,float[] values)
        {
            if((shape.rank!=2&&shape.rank!=3)||(shape.rank==3&&shape[0]!=1)||shape.length==0
                ||values==null||shape.length!=values.Length)
                throw new ArgumentException("Expected fixed rank-2 or batch-one rank-3 FP32 left coefficients.");
            foreach(var value in values)
                if(float.IsNaN(value)||float.IsInfinity(value)||Math.Abs(value)>100)
                    throw new ArgumentException("Expected bounded finite FP32 coefficients.",nameof(values));
            if(!SystemInfo.supportsComputeShaders)throw new NotSupportedException("Fixed-left matmul requires an actual compute GPU.");
            try
            {
                operation=new GpuBatchedMatmul();
                coefficients=new Tensor<float>(shape,(float[])values.Clone());ComputeTensorData.Pin(coefficients);
            }
            catch{Dispose();throw;}
        }

        public Tensor<float> Execute(Tensor<float> input)
        {
            if(disposed)throw new ObjectDisposedException(nameof(GpuFixedLeftMatmul));
            return operation.Execute(coefficients,input);
        }

        public void Dispose()
        {
            if(disposed)return;disposed=true;
            try{operation?.Dispose();}finally{operation=null;coefficients?.Dispose();coefficients=null;}
        }
    }
}
