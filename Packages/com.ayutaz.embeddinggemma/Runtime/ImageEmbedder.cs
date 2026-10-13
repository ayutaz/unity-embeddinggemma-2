using System;
using System.Linq;
using Unity.InferenceEngine;
using UnityEngine;

namespace EmbeddingGemma
{
    public interface IImageEmbedder : IDisposable
    {
        BackendType Backend { get; }
        float[] Embed(Texture input, TextRole role = TextRole.Raw, string title = null);
    }

    /// <summary>
    /// Synchronous main-thread image-only composition. Borrows the texture and all stages;
    /// owns per-call intermediate results and returns an independent CPU copy of the final
    /// 768 vector. Dispose releases references without disposing the caller's workers.
    /// The supplied preprocessor determines supported textures and input access requirements.
    /// </summary>
    public sealed class ImageEmbedder : IImageEmbedder
    {
        IImagePreprocessor preprocessor;
        IImageFeatureEncoder encoder;
        IMultimodalBackbone backbone;
        MultimodalPromptBuilder prompts;
        bool disposed;
        public BackendType Backend { get; }
        public MultimodalBackboneProfile Profile { get; }

        public ImageEmbedder(IImagePreprocessor preprocessor, IImageFeatureEncoder encoder,
            IMultimodalBackbone backbone, MultimodalPromptBuilder prompts)
        {
            if (preprocessor == null) throw new ArgumentNullException(nameof(preprocessor));
            if (encoder == null) throw new ArgumentNullException(nameof(encoder));
            if (backbone == null) throw new ArgumentNullException(nameof(backbone));
            if (prompts == null) throw new ArgumentNullException(nameof(prompts));
            if (preprocessor.Backend != BackendType.CPU && preprocessor.Backend != BackendType.GPUCompute)
                throw new ArgumentOutOfRangeException(nameof(preprocessor), "Use CPU or GPUCompute explicitly.");
            if (preprocessor.Backend != encoder.Backend || encoder.Backend != backbone.Backend)
                throw new ArgumentException("Every image stage must use the same backend.");
            if (encoder.Profile == null || backbone.Profile == null || backbone.Profile.AudioFeatureCount != 0 ||
                backbone.Profile.ImageFeatureCount != encoder.Profile.FeatureCount || !SameProfile(backbone.Profile, prompts.Profile))
                throw new ArgumentException("Image-only backbone, encoder and prompt profiles must agree.");
            this.preprocessor = preprocessor;
            this.encoder = encoder;
            this.backbone = backbone;
            this.prompts = prompts;
            Backend = preprocessor.Backend;
            Profile = backbone.Profile;
        }

        static bool SameProfile(MultimodalBackboneProfile left, MultimodalBackboneProfile right) =>
            left.SequenceLength == right.SequenceLength && left.ImageTokenId == right.ImageTokenId &&
            left.AudioTokenId == right.AudioTokenId && left.ImageFeatureCount == right.ImageFeatureCount &&
            left.AudioFeatureCount == right.AudioFeatureCount && left.VocabularySize == right.VocabularySize && left.PadTokenId == right.PadTokenId;

        public float[] Embed(Texture input, TextRole role = TextRole.Raw, string title = null)
        {
            if (disposed) throw new ObjectDisposedException(nameof(ImageEmbedder));
            if (ReferenceEquals(input, null)) throw new ArgumentNullException(nameof(input));
            if (input == null) throw new ArgumentException("The borrowed texture has been destroyed.", nameof(input));
            // Prompt validation precedes any GPU work. Its feature count comes from the fixed profile.
            var prompt = prompts.BuildImage(Profile.ImageFeatureCount, role, title);
            using var processed = preprocessor.Process(input);
            using var encoded = encoder.Encode(processed);
            using var emptyAudio = new Tensor<float>(new TensorShape(0, MultimodalBackboneProfile.FeatureWidth));
            var vector = backbone.Embed(prompt.InputIds, prompt.AttentionMask, encoded.SoftTokens, emptyAudio);
            if (vector == null || vector.Length != MultimodalBackbone.EmbeddingDimension ||
                vector.Any(value => float.IsNaN(value) || float.IsInfinity(value)) ||
                Math.Abs(vector.Sum(value => (double)value * value) - 1) > 0.002)
                throw new InvalidOperationException("Expected a finite L2-normalized 768-dimensional image embedding.");
            return (float[])vector.Clone();
        }

        public void Dispose()
        {
            disposed = true;
            preprocessor = null; encoder = null; backbone = null; prompts = null;
        }
    }
}
