using Unity.InferenceEngine;

namespace EmbeddingGemma
{
    public interface IAudioPreprocessor
    {
        BackendType Backend { get; }
        AudioPreprocessingResult Process(AudioPcmInput input);
    }
    public interface IAudioFeatureEncoder
    {
        BackendType Backend { get; }
        AudioEncoderProfile Profile { get; }
        AudioEncodingResult Encode(AudioPreprocessingResult input);
    }
    public interface IMultimodalBackbone
    {
        BackendType Backend { get; }
        MultimodalBackboneProfile Profile { get; }
        float[] Embed(int[] inputIds, int[] attentionMask, Tensor<float> imageFeatures, Tensor<float> audioFeatures);
    }
}
