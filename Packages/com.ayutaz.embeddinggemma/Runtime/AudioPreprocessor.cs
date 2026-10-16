using System;
using System.Linq;
using Unity.InferenceEngine;
using UnityEngine;

namespace EmbeddingGemma
{
    /// <summary>
    /// Main-thread execution of the pinned PCM preprocessing graph. Borrows the Model,
    /// owns its Worker and per-call input tensors, and returns independently owned outputs.
    /// GPU processing never downloads intermediate features or masks.
    /// </summary>
    public sealed class AudioPreprocessor : IDisposable, IAudioPreprocessor
    {
        public BackendType Backend { get; }
        readonly string featureOutput, maskOutput;
        Worker worker;

        public AudioPreprocessor(Model model, BackendType backend)
        {
            if (model == null) throw new ArgumentNullException(nameof(model));
            if (backend != BackendType.CPU && backend != BackendType.GPUCompute)
                throw new ArgumentOutOfRangeException(nameof(backend), "Use CPU or GPUCompute explicitly.");
            if (backend == BackendType.GPUCompute && !SystemInfo.supportsComputeShaders)
                throw new NotSupportedException("This device cannot run GPUCompute.");
            var shape = new TensorShape(1, AudioPcmInput.ProfileSampleCount);
            if (model.inputs.Count != 2 || model.outputs.Count != 2 ||
                model.outputs.Select(output => output.name).Distinct().Count() != 2 ||
                !Matches(model, "waveform", DataType.Float, shape) ||
                !Matches(model, "sample_mask", DataType.Int, shape))
                throw new ArgumentException("Expected fixed waveform / sample_mask inputs and two distinct ordered outputs.", nameof(model));
            featureOutput = model.outputs[0].name;
            maskOutput = model.outputs[1].name;
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

        public AudioPreprocessingResult Process(AudioPcmInput input)
        {
            if (worker == null) throw new ObjectDisposedException(nameof(AudioPreprocessor));
            if (input == null) throw new ArgumentNullException(nameof(input));
            using var waveform = new Tensor<float>(new TensorShape(1, AudioPcmInput.ProfileSampleCount), input.Waveform.ToArray());
            using var sampleMask = new Tensor<int>(new TensorShape(1, AudioPcmInput.ProfileSampleCount), input.SampleMask.ToArray());
            if (Backend == BackendType.GPUCompute)
            {
                // PCM upload is the explicit CPU input boundary, before preprocessing.
                ComputeTensorData.Pin(waveform);
                ComputeTensorData.Pin(sampleMask);
            }
            worker.SetInput("waveform", waveform);
            worker.SetInput("sample_mask", sampleMask);
            worker.Schedule();
            var features = worker.PeekOutput(featureOutput) as Tensor<float>;
            var frameMask = worker.PeekOutput(maskOutput) as Tensor<int>;
            if (features == null || frameMask == null ||
                features.shape != new TensorShape(1, AudioPcmInput.ProfileFrameCount, 128) ||
                frameMask.shape != new TensorShape(1, AudioPcmInput.ProfileFrameCount) ||
                features.backendType != Backend || frameMask.backendType != Backend)
                throw new InvalidOperationException("Expected float [1,1119,128] features and int [1,1119] mask on the requested backend.");
            Tensor ownedFeatures = null, ownedMask = null;
            try
            {
                // Device copies preserve prior results when the Worker is reused or disposed.
                worker.CopyOutput(featureOutput, ref ownedFeatures);
                worker.CopyOutput(maskOutput, ref ownedMask);
                return new AudioPreprocessingResult((Tensor<float>)ownedFeatures,
                    (Tensor<int>)ownedMask, input.ValidFrameCount);
            }
            catch
            {
                ownedFeatures?.Dispose(); ownedMask?.Dispose();
                throw;
            }
        }

        public void Dispose()
        {
            worker?.Dispose();
            worker = null;
        }
    }

    /// <summary>Owns both output tensors. Callers may borrow them for subsequent inference until Dispose.</summary>
    public sealed class AudioPreprocessingResult : IDisposable
    {
        Tensor<float> features;
        Tensor<int> frameMask;
        public Tensor<float> Features => features ?? throw new ObjectDisposedException(nameof(AudioPreprocessingResult));
        public Tensor<int> FrameMask => frameMask ?? throw new ObjectDisposedException(nameof(AudioPreprocessingResult));
        public int ValidFrameCount { get; }

        internal AudioPreprocessingResult(Tensor<float> features, Tensor<int> frameMask, int validFrameCount)
        {
            this.features = features;
            this.frameMask = frameMask;
            ValidFrameCount = validFrameCount;
        }

        public void Dispose()
        {
            features?.Dispose(); frameMask?.Dispose();
            features = null; frameMask = null;
        }
    }
}
