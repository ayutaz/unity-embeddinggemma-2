using Unity.InferenceEngine;
using UnityEngine;

namespace EmbeddingGemma
{
    public interface IImagePreprocessor
    {
        BackendType Backend { get; }
        ImagePreprocessingResult Process(Texture input);
    }

    public interface IImageFeatureEncoder
    {
        BackendType Backend { get; }
        ImageEncoderProfile Profile { get; }
        ImageEncodingResult Encode(ImagePreprocessingResult input);
    }
}
