using System;
using System.Linq;
using Unity.InferenceEngine;
using UnityEngine;

namespace EmbeddingGemma
{
    /// <summary>
    /// Synchronous main-thread inference from a fixed token layout and projected soft tokens.
    /// Owns the Worker and per-call IDs/mask. Borrows the Model and feature tensors;
    /// feature tensor values are never downloaded. Only the final vector is copied to CPU.
    /// Processor prompt construction and encoder execution are separate contracts.
    /// </summary>
    public sealed class MultimodalBackbone : IDisposable, IMultimodalBackbone
    {
        public const int EmbeddingDimension = 768;
        public BackendType Backend { get; }
        public MultimodalBackboneProfile Profile { get; }
        Worker worker;

        public MultimodalBackbone(Model model, MultimodalBackboneProfile profile, BackendType backend)
        {
            if (model == null) throw new ArgumentNullException(nameof(model));
            if (profile == null) throw new ArgumentNullException(nameof(profile));
            if (backend != BackendType.CPU && backend != BackendType.GPUCompute)
                throw new ArgumentOutOfRangeException(nameof(backend), "Use CPU or GPUCompute explicitly.");
            if (backend == BackendType.GPUCompute && !SystemInfo.supportsComputeShaders)
                throw new NotSupportedException("This device cannot run GPUCompute.");
            if (model.inputs.Count != 4 || model.outputs.Count != 1 ||
                !Matches(model, "input_ids", DataType.Int, new TensorShape(1, profile.SequenceLength)) ||
                !Matches(model, "attention_mask", DataType.Int, new TensorShape(1, profile.SequenceLength)) ||
                !Matches(model, "image_features", DataType.Float, new TensorShape(profile.ImageFeatureCount, MultimodalBackboneProfile.FeatureWidth)) ||
                !Matches(model, "audio_features", DataType.Float, new TensorShape(profile.AudioFeatureCount, MultimodalBackboneProfile.FeatureWidth)))
                throw new ArgumentException("Expected four named static inputs matching the fixed multimodal profile.", nameof(model));
            Profile = profile;
            Backend = backend;
            worker = new Worker(model, backend);
            if (worker.backendType != backend)
            {
                Dispose();
                throw new NotSupportedException("The requested backend was not created.");
            }
        }

        static bool Matches(Model model, string name, DataType dataType, TensorShape shape) =>
            model.inputs.Any(input => input.name == name && input.dataType == dataType &&
                input.shape.IsStatic() && input.shape.ToTensorShape() == shape);

        public float[] Embed(int[] inputIds, int[] attentionMask, Tensor<float> imageFeatures, Tensor<float> audioFeatures)
        {
            if (worker == null) throw new ObjectDisposedException(nameof(MultimodalBackbone));
            Profile.ValidateTokens(inputIds, attentionMask);
            ValidateFeatures(imageFeatures, Profile.ImageFeatureCount, nameof(imageFeatures));
            ValidateFeatures(audioFeatures, Profile.AudioFeatureCount, nameof(audioFeatures));
            using var ids = new Tensor<int>(new TensorShape(1, Profile.SequenceLength), inputIds);
            using var mask = new Tensor<int>(new TensorShape(1, Profile.SequenceLength), attentionMask);
            worker.SetInput("input_ids", ids);
            worker.SetInput("attention_mask", mask);
            worker.SetInput("image_features", imageFeatures);
            worker.SetInput("audio_features", audioFeatures);
            worker.Schedule();
            var output = worker.PeekOutput() as Tensor<float>;
            if (output == null || output.shape != new TensorShape(1, EmbeddingDimension) || output.backendType != Backend)
                throw new InvalidOperationException("Expected a float32 [1,768] output on the requested backend.");
            var result = output.DownloadToArray();
            if (result.Any(value => float.IsNaN(value) || float.IsInfinity(value)))
                throw new InvalidOperationException("Embedding contains non-finite values.");
            if (Math.Abs(result.Sum(value => (double)value * value) - 1) > 0.002)
                throw new InvalidOperationException("Exported embedding is not L2 normalized.");
            return result;
        }

        void ValidateFeatures(Tensor<float> tensor, int count, string name)
        {
            if (tensor == null) throw new ArgumentNullException(name);
            if (tensor.shape != new TensorShape(count, MultimodalBackboneProfile.FeatureWidth))
                throw new ArgumentException("Feature shape must match the fixed profile.", name);
            if (count != 0 && tensor.dataOnBackend == null)
                throw new ArgumentException("Nonempty features must have live allocated tensor data.", name);
            if (count != 0 && tensor.backendType != Backend)
                throw new ArgumentException("Nonempty features must already reside on the requested backend.", name);
        }

        public void Dispose()
        {
            worker?.Dispose();
            worker = null;
        }
    }
}
