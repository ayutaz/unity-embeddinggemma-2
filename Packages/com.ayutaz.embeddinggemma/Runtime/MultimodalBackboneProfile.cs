using System;

namespace EmbeddingGemma
{
    /// <summary>Fixed batch-one contract for already projected soft-token features.</summary>
    public sealed class MultimodalBackboneProfile
    {
        public const int FeatureWidth = 512;
        public int SequenceLength { get; }
        public int ImageTokenId { get; }
        public int AudioTokenId { get; }
        public int ImageFeatureCount { get; }
        public int AudioFeatureCount { get; }
        public int VocabularySize { get; }
        public int PadTokenId { get; }

        public MultimodalBackboneProfile(int sequenceLength, int imageTokenId, int audioTokenId,
            int imageFeatureCount, int audioFeatureCount, int vocabularySize, int padTokenId)
        {
            if (sequenceLength < 2 || sequenceLength > 8192 || imageFeatureCount < 0 || audioFeatureCount < 0 ||
                (long)imageFeatureCount + audioFeatureCount > sequenceLength || vocabularySize <= 0 ||
                padTokenId < 0 || padTokenId >= vocabularySize || imageTokenId < 0 || audioTokenId < 0 ||
                imageTokenId == audioTokenId || imageTokenId == padTokenId || audioTokenId == padTokenId)
                throw new ArgumentException("Invalid fixed multimodal profile, token IDs or feature budget.");
            SequenceLength = sequenceLength;
            ImageTokenId = imageTokenId;
            AudioTokenId = audioTokenId;
            ImageFeatureCount = imageFeatureCount;
            AudioFeatureCount = audioFeatureCount;
            VocabularySize = vocabularySize;
            PadTokenId = padTokenId;
        }

        public void ValidateTokens(int[] inputIds, int[] attentionMask)
        {
            if (inputIds == null) throw new ArgumentNullException(nameof(inputIds));
            if (attentionMask == null) throw new ArgumentNullException(nameof(attentionMask));
            if (inputIds.Length != SequenceLength || attentionMask.Length != SequenceLength)
                throw new ArgumentException("Token IDs and mask must match the fixed sequence length.");
            int images = 0, audio = 0, valid = 0;
            for (var index = 0; index < SequenceLength; index++)
            {
                var id = inputIds[index]; var mask = attentionMask[index];
                if (mask != 0 && mask != 1) throw new ArgumentException("Attention mask must be binary.");
                valid += mask;
                if (id == ImageTokenId || id == AudioTokenId)
                {
                    if (mask == 0) throw new ArgumentException("Feature placeholders cannot be padding.");
                    if (id == ImageTokenId) images++; else audio++;
                }
                else if (id < 0 || id >= VocabularySize)
                    throw new ArgumentException("Text token ID is outside the fixed vocabulary.");
            }
            if (valid == 0 || images != ImageFeatureCount || audio != AudioFeatureCount)
                throw new ArgumentException("Nonempty mask and exact feature placeholder counts are required.");
        }
    }
}
