using System;
using Newtonsoft.Json.Linq;
using Unity.InferenceEngine.Tokenization;
using Unity.InferenceEngine.Tokenization.Parsers.HuggingFace;

namespace EmbeddingGemma
{
    /// <summary>
    /// Sentis tokenizer for the pinned text model, including its empty-input contract.
    /// Sentis 2.6.1 rejects empty input in AddedVocabulary.Split. For an empty sequence,
    /// preserve the configured post-processing and padding with a token-free pipeline.
    /// </summary>
    public sealed class TextTokenizer
    {
        readonly ITokenizer tokenizer;
        readonly ITokenizer emptyTokenizer;

        public TextTokenizer(string tokenizerJson)
        {
            if (tokenizerJson == null) throw new ArgumentNullException(nameof(tokenizerJson));
            tokenizer = HuggingFaceParser.GetDefault().Parse(tokenizerJson);
            var config = JObject.Parse(tokenizerJson);
            var emptyConfig = new JObject
            {
                ["added_tokens"] = new JArray(),
                ["normalizer"] = null,
                ["pre_tokenizer"] = new JObject { ["type"] = "WhitespaceSplit" },
                ["post_processor"] = config["post_processor"]?.DeepClone(),
                ["truncation"] = config["truncation"]?.DeepClone(),
                ["padding"] = config["padding"]?.DeepClone(),
                ["decoder"] = null,
                ["model"] = new JObject
                {
                    ["type"] = "WordLevel", ["unk_token"] = "[UNK]",
                    ["vocab"] = new JObject { ["[UNK]"] = 0 }
                }
            };
            emptyTokenizer = HuggingFaceParser.GetDefault().Parse(emptyConfig.ToString());
        }

        public IEncoding Encode(string text)
        {
            if (text == null) throw new ArgumentNullException(nameof(text));
            // WhitespaceSplit produces zero ordinary tokens. Sentis still applies the actual
            // configured special-token template and padding; no model token IDs are hardcoded.
            return text.Length == 0 ? emptyTokenizer.Encode(" ") : tokenizer.Encode(text);
        }
    }
}
