using System;
using System.Linq;
using Unity.InferenceEngine;

namespace EmbeddingGemma
{
    public interface IAudioEmbedder : IDisposable
    {
        BackendType Backend { get; }
        float[] Embed(AudioPcmInput input, TextRole role = TextRole.Raw, string title = null);
    }

    /// <summary>
    /// Synchronous main-thread audio-only composition. Borrows all stages; owns per-call
    /// intermediate results and returns an independent CPU copy of the final 768 vector.
    /// Dispose releases borrowed references, never the caller's workers.
    /// </summary>
    public sealed class AudioEmbedder : IAudioEmbedder
    {
        IAudioPreprocessor preprocessor;
        IAudioFeatureEncoder encoder;
        IMultimodalBackbone backbone;
        MultimodalPromptBuilder prompts;
        bool disposed;
        public BackendType Backend { get; }
        public MultimodalBackboneProfile Profile { get; }

        public AudioEmbedder(IAudioPreprocessor preprocessor, IAudioFeatureEncoder encoder,
            IMultimodalBackbone backbone, MultimodalPromptBuilder prompts)
        {
            if (preprocessor == null) throw new ArgumentNullException(nameof(preprocessor));
            if (encoder == null) throw new ArgumentNullException(nameof(encoder));
            if (backbone == null) throw new ArgumentNullException(nameof(backbone));
            if (prompts == null) throw new ArgumentNullException(nameof(prompts));
            if (preprocessor.Backend != BackendType.CPU && preprocessor.Backend != BackendType.GPUCompute)
                throw new ArgumentOutOfRangeException(nameof(preprocessor), "Use CPU or GPUCompute explicitly.");
            if (preprocessor.Backend != encoder.Backend || encoder.Backend != backbone.Backend)
                throw new ArgumentException("Every audio stage must use the same backend.");
            if (encoder.Profile == null || backbone.Profile == null || backbone.Profile.ImageFeatureCount != 0 ||
                backbone.Profile.AudioFeatureCount != encoder.Profile.FeatureCount || !SameProfile(backbone.Profile, prompts.Profile))
                throw new ArgumentException("Audio-only backbone, encoder and prompt profiles must agree.");
            this.preprocessor = preprocessor; this.encoder = encoder; this.backbone = backbone; this.prompts = prompts;
            Backend = preprocessor.Backend; Profile = backbone.Profile;
        }

        static bool SameProfile(MultimodalBackboneProfile left, MultimodalBackboneProfile right) =>
            left.SequenceLength == right.SequenceLength && left.ImageTokenId == right.ImageTokenId &&
            left.AudioTokenId == right.AudioTokenId && left.ImageFeatureCount == right.ImageFeatureCount &&
            left.AudioFeatureCount == right.AudioFeatureCount && left.VocabularySize == right.VocabularySize && left.PadTokenId == right.PadTokenId;

        public float[] Embed(AudioPcmInput input, TextRole role = TextRole.Raw, string title = null)
        {
            if (disposed) throw new ObjectDisposedException(nameof(AudioEmbedder));
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (input.ValidFrameCount != encoder.Profile.ValidFrameCount)
                throw new ArgumentException("PCM prefix must match the fixed encoder profile.", nameof(input));
            // Prompt validation precedes GPU work; its mask comes from immutable PCM metadata.
            var prompt = prompts.BuildAudio(input.ValidFrameCount, role, title);
            using var processed = preprocessor.Process(input);
            using var encoded = encoder.Encode(processed);
            using var emptyImage = new Tensor<float>(new TensorShape(0, MultimodalBackboneProfile.FeatureWidth));
            var vector = backbone.Embed(prompt.InputIds, prompt.AttentionMask, emptyImage, encoded.SoftTokens);
            if (vector == null || vector.Length != MultimodalBackbone.EmbeddingDimension ||
                vector.Any(value => float.IsNaN(value) || float.IsInfinity(value)) ||
                Math.Abs(vector.Sum(value => (double)value * value) - 1) > 0.002)
                throw new InvalidOperationException("Expected a finite L2-normalized 768-dimensional audio embedding.");
            return (float[])vector.Clone();
        }

        public void Dispose()
        {
            disposed = true;
            preprocessor = null; encoder = null; backbone = null; prompts = null;
        }
    }
}
