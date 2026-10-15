using System;
using Unity.InferenceEngine;

namespace EmbeddingGemma.Validation.Tests
{
    public static class SavedProjectionPartition
    {
        public sealed class Plan
        {
            public Model[] Models;
            public string[][] Inputs;
            public string[] Outputs;
            public int[] Widths,Columns,Rows;
            public float[][] Weights;
        }
        public static Plan Create(Model source,int[] selectedOutputs)
            =>throw new NotImplementedException("Live-residual saved projection partitions are missing.");
    }
}
