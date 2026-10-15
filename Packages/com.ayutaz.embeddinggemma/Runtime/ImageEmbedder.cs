using System;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Unity.InferenceEngine;
using UnityEngine;

namespace EmbeddingGemma
{
    public interface IImageEmbedder : IDisposable
    {
        BackendType Backend { get; }
        float[] Embed(Texture input, TextRole role = TextRole.Raw, string title = null);
    }

    public interface IAsyncImageEmbedder : IImageEmbedder
    {
        Task<float[]> EmbedAsync(Texture input, TextRole role = TextRole.Raw, string title = null, CancellationToken cancellationToken = default);
    }

    /// <summary>
    /// Main-thread image-only composition. Borrows the texture and all stages;
    /// owns per-call intermediate results and returns an independent CPU copy of the final
    /// 768 vector. Dispose releases references without disposing the caller's workers.
    /// The supplied preprocessor determines supported textures and input access requirements.
    /// </summary>
    public sealed class ImageEmbedder : IAsyncImageEmbedder
    {
        IImagePreprocessor preprocessor;
        IImageFeatureEncoder encoder;
        IMultimodalBackbone backbone;
        MultimodalPromptBuilder prompts;
        bool disposed;
        bool executing;
        CancellationTokenSource pendingCancellation;
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

        MultimodalPrompt Prepare(Texture input, TextRole role, string title)
        {
            if (disposed) throw new ObjectDisposedException(nameof(ImageEmbedder));
            if (executing) throw new InvalidOperationException("Only one image embedding call may be active.");
            if (ReferenceEquals(input, null)) throw new ArgumentNullException(nameof(input));
            if (input == null) throw new ArgumentException("The borrowed texture has been destroyed.", nameof(input));
            // Prompt validation precedes any GPU work. Its feature count comes from the fixed profile.
            return prompts.BuildImage(Profile.ImageFeatureCount, role, title);
        }

        static float[] OwnedVector(float[] vector)
        {
            if (vector == null || vector.Length != MultimodalBackbone.EmbeddingDimension ||
                vector.Any(value => float.IsNaN(value) || float.IsInfinity(value)) ||
                Math.Abs(vector.Sum(value => (double)value * value) - 1) > 0.002)
                throw new InvalidOperationException("Expected a finite L2-normalized 768-dimensional image embedding.");
            return (float[])vector.Clone();
        }

        public float[] Embed(Texture input, TextRole role = TextRole.Raw, string title = null)
        {
            var prompt = Prepare(input, role, title);
            var preprocessing = preprocessor; var encoding = encoder; var body = backbone;
            executing = true;
            try
            {
                using var processed = preprocessing.Process(input);
                if (disposed) throw new ObjectDisposedException(nameof(ImageEmbedder));
                using var encoded = encoding.Encode(processed);
                if (disposed) throw new ObjectDisposedException(nameof(ImageEmbedder));
                using var emptyAudio = new Tensor<float>(new TensorShape(0, MultimodalBackboneProfile.FeatureWidth));
                return OwnedVector(body.Embed(prompt.InputIds, prompt.AttentionMask, encoded.SoftTokens, emptyAudio));
            }
            finally { executing = false; }
        }

        /// <summary>
        /// Awaits a staged encoder on Unity's main thread. Keeps owned inputs and
        /// features alive through the backbone, downloads only its final vector,
        /// and cancels an active call when this facade is disposed. Keep borrowed
        /// textures/stages alive until the returned task completes, including on
        /// cancellation or Dispose; the facade never disposes borrowed stages.
        /// </summary>
        public async Task<float[]> EmbedAsync(Texture input, TextRole role = TextRole.Raw, string title = null, CancellationToken cancellationToken = default)
        {
            var prompt = Prepare(input, role, title);
            cancellationToken.ThrowIfCancellationRequested();
            if (!(encoder is IAsyncImageFeatureEncoder encoding))
                throw new NotSupportedException("Asynchronous image embedding requires a staged image feature encoder.");
            var preprocessing = preprocessor; var body = backbone;
            using var cancellation = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
            executing = true; pendingCancellation = cancellation;
            try
            {
                using var processed = preprocessing.Process(input);
                cancellation.Token.ThrowIfCancellationRequested();
                using var encoded = await encoding.EncodeAsync(processed, cancellation.Token);
                cancellation.Token.ThrowIfCancellationRequested();
                using var emptyAudio = new Tensor<float>(new TensorShape(0, MultimodalBackboneProfile.FeatureWidth));
                var vector = body.Embed(prompt.InputIds, prompt.AttentionMask, encoded.SoftTokens, emptyAudio);
                cancellation.Token.ThrowIfCancellationRequested();
                return OwnedVector(vector);
            }
            finally { pendingCancellation = null; executing = false; }
        }

        public void Dispose()
        {
            if (disposed) return;
            disposed = true;
            try { pendingCancellation?.Cancel(); }
            finally { preprocessor = null; encoder = null; backbone = null; prompts = null; }
        }
    }
}
