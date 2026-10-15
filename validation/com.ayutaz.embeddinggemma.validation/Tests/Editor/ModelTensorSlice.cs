using System;
using Unity.InferenceEngine;

namespace EmbeddingGemma.Validation.Tests
{
    public static class ModelTensorSlice
    {
        public static Model Extract(Model source,int tensorIndex,string outputName)
            =>throw new NotImplementedException("Immutable saved-model tensor slicing is missing.");
    }
}
