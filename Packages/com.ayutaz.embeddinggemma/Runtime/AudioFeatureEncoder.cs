using System;
using Unity.InferenceEngine;
using UnityEngine;

namespace EmbeddingGemma
{
    /// <summary>Fixed prefix baked into an audited audio encoder/projector model.</summary>
    public sealed class AudioEncoderProfile
    {
        public const int FrameCount = AudioPcmInput.ProfileFrameCount;
        public const int MelBinCount = 128;
        public const int FeatureWidth = MultimodalBackboneProfile.FeatureWidth;
        public int ValidFrameCount { get; }
        public int FeatureCount { get; }

        public AudioEncoderProfile(int validFrameCount)
        {
            if (validFrameCount < 1 || validFrameCount > FrameCount)
                throw new ArgumentOutOfRangeException(nameof(validFrameCount), "Expected a nonempty prefix within 1119 frames.");
            ValidFrameCount = validFrameCount;
            FeatureCount = (validFrameCount + 3) / 4;
        }
    }

    /// <summary>
    /// Synchronous main-thread execution. Borrows a model exported and audited for exactly Profile.ValidFrameCount.
    /// Owns its Worker, borrows preprocessing tensors, and returns independently owned
    /// projected soft tokens on the requested device. This is not a final embedding.
    /// </summary>
    public sealed class AudioFeatureEncoder : IDisposable, IAudioFeatureEncoder
    {
        public BackendType Backend { get; }
        public AudioEncoderProfile Profile { get; }
        readonly string outputName;
        Worker worker;

        public AudioFeatureEncoder(Model model, BackendType backend, AudioEncoderProfile profile)
        {
            if (model == null) throw new ArgumentNullException(nameof(model));
            if (profile == null) throw new ArgumentNullException(nameof(profile));
            if (backend != BackendType.CPU && backend != BackendType.GPUCompute)
                throw new ArgumentOutOfRangeException(nameof(backend), "Use CPU or GPUCompute explicitly.");
            if (backend == BackendType.GPUCompute && !SystemInfo.supportsComputeShaders)
                throw new NotSupportedException("This device cannot run GPUCompute.");
            if (model.inputs.Count != 1 || model.outputs.Count != 1 ||
                model.inputs[0].name != "input" || model.inputs[0].dataType != DataType.Float ||
                !model.inputs[0].shape.IsStatic() ||
                model.inputs[0].shape.ToTensorShape() != new TensorShape(1, AudioEncoderProfile.FrameCount, AudioEncoderProfile.MelBinCount))
                throw new ArgumentException("Expected one fixed float [1,1119,128] input named input and one projected-feature output.", nameof(model));
            Profile = profile; Backend = backend; outputName = model.outputs[0].name;
            worker = new Worker(model, backend);
            if (worker.backendType != backend)
            {
                Dispose();
                throw new NotSupportedException("The requested backend was not created.");
            }
        }

        public AudioEncodingResult Encode(AudioPreprocessingResult input)
        {
            if (worker == null) throw new ObjectDisposedException(nameof(AudioFeatureEncoder));
            if (input == null) throw new ArgumentNullException(nameof(input));
            var features = input.Features; var mask = input.FrameMask;
            if (features.dataOnBackend == null || mask.dataOnBackend == null)
                throw new ObjectDisposedException(nameof(input));
            if (input.ValidFrameCount != Profile.ValidFrameCount ||
                features.shape != new TensorShape(1, AudioEncoderProfile.FrameCount, AudioEncoderProfile.MelBinCount) ||
                mask.shape != new TensorShape(1, AudioEncoderProfile.FrameCount) ||
                features.backendType != Backend || mask.backendType != Backend)
                throw new ArgumentException("Preprocessing prefix, tensor shapes and devices must match the fixed encoder profile.", nameof(input));
            // The audited encoder export has a fixed mask. Validate its exact prefix
            // through preprocessing metadata without downloading the GPU mask.
            worker.Schedule(features);
            var output = worker.PeekOutput(outputName) as Tensor<float>;
            if (output == null || output.shape != new TensorShape(Profile.FeatureCount, AudioEncoderProfile.FeatureWidth) ||
                output.backendType != Backend)
                throw new InvalidOperationException("Expected projected float soft tokens on the requested backend.");
            Tensor owned = null;
            try
            {
                worker.CopyOutput(outputName, ref owned);
                return new AudioEncodingResult((Tensor<float>)owned, Profile.ValidFrameCount);
            }
            catch { owned?.Dispose(); throw; }
        }

        public void Dispose() { worker?.Dispose(); worker = null; }
    }

    /// <summary>Owns projected soft tokens. Borrow SoftTokens only until Dispose.</summary>
    public sealed class AudioEncodingResult : IDisposable
    {
        Tensor<float> softTokens;
        public Tensor<float> SoftTokens => softTokens ?? throw new ObjectDisposedException(nameof(AudioEncodingResult));
        public int ValidFrameCount { get; }

        internal AudioEncodingResult(Tensor<float> softTokens, int validFrameCount)
        {
            this.softTokens = softTokens; ValidFrameCount = validFrameCount;
        }
        public void Dispose() { softTokens?.Dispose(); softTokens = null; }
    }
}
