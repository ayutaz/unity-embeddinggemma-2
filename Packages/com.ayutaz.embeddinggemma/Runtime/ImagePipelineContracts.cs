using Unity.InferenceEngine;
using UnityEngine;
using System.Threading;
using System.Threading.Tasks;

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

    public interface IAsyncImageFeatureEncoder : IImageFeatureEncoder
    {
        Task<ImageEncodingResult> EncodeAsync(ImagePreprocessingResult input, CancellationToken cancellationToken = default);
    }
}
