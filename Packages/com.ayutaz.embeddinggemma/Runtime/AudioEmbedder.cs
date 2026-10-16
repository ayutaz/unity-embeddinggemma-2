using System;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Unity.InferenceEngine;

namespace EmbeddingGemma
{
    public interface IAudioEmbedder : IDisposable
    {
        BackendType Backend { get; }
        float[] Embed(AudioPcmInput input, TextRole role = TextRole.Raw, string title = null);
    }
    public interface IAsyncAudioEmbedder : IAudioEmbedder
    {
        Task<float[]> EmbedAsync(AudioPcmInput input,TextRole role=TextRole.Raw,string title=null,CancellationToken cancellationToken=default);
    }

    /// <summary>
    /// Main-thread audio-only composition. Borrows all stages; owns per-call
    /// intermediate results and returns an independent CPU copy of the final 768 vector.
    /// Dispose releases borrowed references, never the caller's workers.
    /// </summary>
    public sealed class AudioEmbedder : IAsyncAudioEmbedder
    {
        IAudioPreprocessor preprocessor;
        IAudioFeatureEncoder encoder;
        IMultimodalBackbone backbone;
        MultimodalPromptBuilder prompts;
        bool disposed;
        bool executing;
        CancellationTokenSource pendingCancellation;
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

        MultimodalPrompt Prepare(AudioPcmInput input,TextRole role,string title)
        {
            if (disposed) throw new ObjectDisposedException(nameof(AudioEmbedder));
            if (executing) throw new InvalidOperationException("Only one audio embedding call may be active.");
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (input.ValidFrameCount != encoder.Profile.ValidFrameCount)
                throw new ArgumentException("PCM prefix must match the fixed encoder profile.", nameof(input));
            // Prompt validation precedes GPU work; its mask comes from immutable PCM metadata.
            return prompts.BuildAudio(input.ValidFrameCount, role, title);
        }

        static float[] OwnedVector(float[] vector)
        {
            if (vector == null || vector.Length != MultimodalBackbone.EmbeddingDimension ||
                vector.Any(value => float.IsNaN(value) || float.IsInfinity(value)) ||
                Math.Abs(vector.Sum(value => (double)value * value) - 1) > 0.002)
                throw new InvalidOperationException("Expected a finite L2-normalized 768-dimensional audio embedding.");
            return (float[])vector.Clone();
        }

        public float[] Embed(AudioPcmInput input,TextRole role=TextRole.Raw,string title=null)
        {
            var prompt=Prepare(input,role,title);
            var preprocessing=preprocessor;var encoding=encoder;var body=backbone;
            executing=true;
            try
            {
                using var processed=preprocessing.Process(input);
                if(disposed)throw new ObjectDisposedException(nameof(AudioEmbedder));
                using var encoded=encoding.Encode(processed);
                if(disposed)throw new ObjectDisposedException(nameof(AudioEmbedder));
                using var emptyImage=new Tensor<float>(new TensorShape(0,MultimodalBackboneProfile.FeatureWidth));
                return OwnedVector(body.Embed(prompt.InputIds,prompt.AttentionMask,emptyImage,encoded.SoftTokens));
            }
            finally{executing=false;}
        }

        /// <summary>
        /// Awaits a staged GPU encoder on Unity's main thread. Holds per-call inputs
        /// across GPU fences; cancellation/Dispose releases them after the stage ends.
        /// Only the final vector is downloaded. Await before disposing borrowed stages.
        /// </summary>
        public async Task<float[]> EmbedAsync(AudioPcmInput input,TextRole role=TextRole.Raw,string title=null,CancellationToken cancellationToken=default)
        {
            var prompt=Prepare(input,role,title);cancellationToken.ThrowIfCancellationRequested();
            if(!(encoder is IAsyncAudioFeatureEncoder encoding))throw new NotSupportedException("Asynchronous audio embedding requires a staged audio feature encoder.");
            var preprocessing=preprocessor;var body=backbone;
            using var cancellation=CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
            executing=true;pendingCancellation=cancellation;
            try
            {
                using var processed=preprocessing is IAsyncAudioPreprocessor stagedPreprocessing
                    ? await stagedPreprocessing.ProcessAsync(input,cancellation.Token)
                    : preprocessing.Process(input);
                cancellation.Token.ThrowIfCancellationRequested();
                using var encoded=await encoding.EncodeAsync(processed,cancellation.Token);cancellation.Token.ThrowIfCancellationRequested();
                using var emptyImage=new Tensor<float>(new TensorShape(0,MultimodalBackboneProfile.FeatureWidth));
                var vector=body.Embed(prompt.InputIds,prompt.AttentionMask,emptyImage,encoded.SoftTokens);
                cancellation.Token.ThrowIfCancellationRequested();return OwnedVector(vector);
            }
            finally{pendingCancellation=null;executing=false;}
        }

        public void Dispose()
        {
            if(disposed)return;
            disposed = true;
            try{pendingCancellation?.Cancel();}
            finally{preprocessor = null; encoder = null; backbone = null; prompts = null;}
        }
    }
}
