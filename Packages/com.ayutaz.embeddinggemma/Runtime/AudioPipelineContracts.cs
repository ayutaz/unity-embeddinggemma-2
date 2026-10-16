using Unity.InferenceEngine;
using System.Threading;
using System.Threading.Tasks;

namespace EmbeddingGemma
{
    public interface IAudioPreprocessor
    {
        BackendType Backend { get; }
        AudioPreprocessingResult Process(AudioPcmInput input);
    }
    public interface IAsyncAudioPreprocessor : IAudioPreprocessor
    {
        Task<AudioPreprocessingResult> ProcessAsync(AudioPcmInput input, CancellationToken cancellationToken = default);
    }
    public interface IAudioFeatureEncoder
    {
        BackendType Backend { get; }
        AudioEncoderProfile Profile { get; }
        AudioEncodingResult Encode(AudioPreprocessingResult input);
    }
    public interface IAsyncAudioFeatureEncoder : IAudioFeatureEncoder
    {
        Task<AudioEncodingResult> EncodeAsync(AudioPreprocessingResult input,CancellationToken cancellationToken=default);
    }
    public interface IMultimodalBackbone
    {
        BackendType Backend { get; }
        MultimodalBackboneProfile Profile { get; }
        float[] Embed(int[] inputIds, int[] attentionMask, Tensor<float> imageFeatures, Tensor<float> audioFeatures);
    }
}
