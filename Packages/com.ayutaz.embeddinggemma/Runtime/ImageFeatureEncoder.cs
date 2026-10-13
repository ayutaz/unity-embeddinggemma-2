using System;
using Unity.InferenceEngine;
using UnityEngine;

namespace EmbeddingGemma
{
    /// <summary>Image geometry baked into an audited batch-one encoder/projector.</summary>
    public sealed class ImageEncoderProfile
    {
        public const int PatchCount = 2520;
        public const int PatchWidth = 768;
        public const int FeatureWidth = MultimodalBackboneProfile.FeatureWidth;
        public int ResizedWidth { get; }
        public int ResizedHeight { get; }
        public int FeatureCount { get; }

        public ImageEncoderProfile(int resizedWidth, int resizedHeight)
        {
            if (resizedWidth <= 0 || resizedHeight <= 0 || resizedWidth % 48 != 0 ||
                resizedHeight % 48 != 0 || (long)resizedWidth * resizedHeight > PatchCount * 256L)
                throw new ArgumentException("Expected positive dimensions divisible by 48 within the 2520-patch budget.");
            ResizedWidth = resizedWidth;
            ResizedHeight = resizedHeight;
            FeatureCount = resizedWidth / 48 * (resizedHeight / 48);
        }

        internal void ValidatePositions(ReadOnlySpan<int> positions)
        {
            if (positions.Length != PatchCount * 2)
                throw new ArgumentException("Expected all patch coordinates, including padded positions.", nameof(positions));
            var columns = ResizedWidth / 16;
            var count = columns * (ResizedHeight / 16);
            for (var patch = 0; patch < PatchCount; patch++)
            {
                var x = patch < count ? patch % columns : -1;
                var y = patch < count ? patch / columns : -1;
                if (positions[patch * 2] != x || positions[patch * 2 + 1] != y)
                    throw new ArgumentException("Patch coordinates must match the exact row-major baked profile.", nameof(positions));
            }
        }
    }

    /// <summary>
    /// Synchronous main-thread execution of a model exported and audited for exactly Profile.
    /// Owns its Worker, borrows preprocessing tensors, and returns independently owned
    /// projected soft tokens on the requested device. This is not a final embedding.
    /// </summary>
    public sealed class ImageFeatureEncoder : IDisposable, IImageFeatureEncoder
    {
        public BackendType Backend { get; }
        public ImageEncoderProfile Profile { get; }
        readonly string outputName;
        Worker worker;

        public ImageFeatureEncoder(Model model, BackendType backend, ImageEncoderProfile profile)
        {
            if (model == null) throw new ArgumentNullException(nameof(model));
            if (profile == null) throw new ArgumentNullException(nameof(profile));
            if (backend != BackendType.CPU && backend != BackendType.GPUCompute)
                throw new ArgumentOutOfRangeException(nameof(backend), "Use CPU or GPUCompute explicitly.");
            if (backend == BackendType.GPUCompute && !SystemInfo.supportsComputeShaders)
                throw new NotSupportedException("This device cannot run GPUCompute.");
            if (model.inputs.Count != 1 || model.outputs.Count != 1 ||
                model.inputs[0].name != "pixels" || model.inputs[0].dataType != DataType.Float ||
                !model.inputs[0].shape.IsStatic() || model.inputs[0].shape.ToTensorShape() !=
                new TensorShape(1, ImageEncoderProfile.PatchCount, ImageEncoderProfile.PatchWidth))
                throw new ArgumentException("Expected one fixed float [1,2520,768] input named pixels and one projected-feature output.", nameof(model));
            Backend = backend;
            Profile = profile;
            outputName = model.outputs[0].name;
            worker = new Worker(model, backend);
            if (worker.backendType != backend)
            {
                Dispose();
                throw new NotSupportedException("The requested backend was not created.");
            }
        }

        public ImageEncodingResult Encode(ImagePreprocessingResult input)
        {
            if (worker == null) throw new ObjectDisposedException(nameof(ImageFeatureEncoder));
            if (input == null) throw new ArgumentNullException(nameof(input));
            var pixels = input.Pixels;
            if (pixels.dataOnBackend == null) throw new ObjectDisposedException(nameof(input));
            if (input.ResizedWidth != Profile.ResizedWidth || input.ResizedHeight != Profile.ResizedHeight ||
                input.SoftTokenCount != Profile.FeatureCount || pixels.shape !=
                new TensorShape(1, ImageEncoderProfile.PatchCount, ImageEncoderProfile.PatchWidth) || pixels.backendType != Backend)
                throw new ArgumentException("Preprocessing geometry, feature count, tensor shape and device must match the fixed profile.", nameof(input));
            // Coordinates are CPU metadata owned by preprocessing; no pixel readback is required.
            Profile.ValidatePositions(input.Positions);
            worker.Schedule(pixels);
            var output = worker.PeekOutput(outputName) as Tensor<float>;
            if (output == null || output.shape != new TensorShape(Profile.FeatureCount, ImageEncoderProfile.FeatureWidth) ||
                output.backendType != Backend)
                throw new InvalidOperationException("Expected projected float soft tokens on the requested backend.");
            Tensor owned = null;
            try
            {
                worker.CopyOutput(outputName, ref owned);
                return new ImageEncodingResult((Tensor<float>)owned, Profile.ResizedWidth, Profile.ResizedHeight);
            }
            catch { owned?.Dispose(); throw; }
        }

        public void Dispose() { worker?.Dispose(); worker = null; }
    }

    /// <summary>Owns projected soft tokens. Borrow SoftTokens only until Dispose.</summary>
    public sealed class ImageEncodingResult : IDisposable
    {
        Tensor<float> softTokens;
        public Tensor<float> SoftTokens => softTokens ?? throw new ObjectDisposedException(nameof(ImageEncodingResult));
        public int ResizedWidth { get; }
        public int ResizedHeight { get; }

        internal ImageEncodingResult(Tensor<float> softTokens, int width, int height)
        {
            this.softTokens = softTokens;
            ResizedWidth = width;
            ResizedHeight = height;
        }

        public void Dispose() { softTokens?.Dispose(); softTokens = null; }
    }
}
