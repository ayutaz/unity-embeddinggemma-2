using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Newtonsoft.Json.Linq;

namespace EmbeddingGemma
{
    /// <summary>Owned fixed-profile input IDs and mask; returned arrays are independent copies.</summary>
    public sealed class MultimodalPrompt
    {
        readonly int[] ids, mask;
        public string FormattedText { get; }
        public int ImageFeatureCount { get; }
        public int AudioFeatureCount { get; }
        public int[] InputIds => (int[])ids.Clone();
        public int[] AttentionMask => (int[])mask.Clone();

        internal MultimodalPrompt(string text, int[] inputIds, int[] attentionMask, int images, int audio)
        {
            FormattedText = text; ids = inputIds; mask = attentionMask;
            ImageFeatureCount = images; AudioFeatureCount = audio;
        }
    }

    /// <summary>
    /// Builds batch-one processor-compatible prompts for one image and/or one audio segment.
    /// Uses the tokenizer's special-token template, then pads to the fixed backbone profile.
    /// Does not truncate inputs, run encoders, or read feature tensor data.
    /// </summary>
    public sealed class MultimodalPromptBuilder
    {
        readonly TextTokenizer tokenizer;
        readonly string imageToken, imageStart, imageEnd, audioToken, audioStart, audioEnd;
        public MultimodalBackboneProfile Profile { get; }

        public MultimodalPromptBuilder(string tokenizerJson, string tokenizerConfigurationJson, MultimodalBackboneProfile profile)
        {
            if (tokenizerJson == null) throw new ArgumentNullException(nameof(tokenizerJson));
            if (tokenizerConfigurationJson == null) throw new ArgumentNullException(nameof(tokenizerConfigurationJson));
            if (profile == null) throw new ArgumentNullException(nameof(profile));
            var configuration = JObject.Parse(tokenizerConfigurationJson);
            imageToken = ReadToken(configuration, "image_token"); imageStart = ReadToken(configuration, "boi_token"); imageEnd = ReadToken(configuration, "eoi_token");
            audioToken = ReadToken(configuration, "audio_token"); audioStart = ReadToken(configuration, "boa_token"); audioEnd = ReadToken(configuration, "eoa_token");
            var markers = new[] {imageToken,imageStart,imageEnd,audioToken,audioStart,audioEnd};
            if (new HashSet<string>(markers, StringComparer.Ordinal).Count != markers.Length)
                throw new ArgumentException("Multimodal special tokens must be distinct.", nameof(tokenizerConfigurationJson));
            var raw = JObject.Parse(tokenizerJson);
            var added = raw["added_tokens"] as JArray ?? throw new ArgumentException("Special token mappings are required.", nameof(tokenizerJson));
            for (var index = 0; index < markers.Length; index++)
            {
                var matching = added.Where(token => token.Value<string>("content") == markers[index]).ToArray();
                if (matching.Length != 1 || matching[0].Value<bool?>("special") != true || matching[0].Value<int?>("id") == null)
                    throw new ArgumentException("Every multimodal marker must have a unique special token mapping.", nameof(tokenizerJson));
                var id = matching[0].Value<int>("id");
                if (index == 0 ? id != profile.ImageTokenId : index == 3 ? id != profile.AudioTokenId : id < 0 || id >= profile.VocabularySize)
                    throw new ArgumentException("Special token IDs do not match the fixed backbone profile.", nameof(profile));
            }
            // Text-only saved tokenizers may contain a 128-token truncation budget.
            // Apply this profile only after the complete prompt has been tokenized.
            raw["truncation"] = null; raw["padding"] = null;
            tokenizer = new TextTokenizer(raw.ToString());
            Profile = profile;
        }

        public MultimodalPrompt Build(string text, int imageFeatureCount = 0, int[] audioFeatureMask = null,
            TextRole role = TextRole.Raw, string title = null)
        {
            if (text == null) throw new ArgumentNullException(nameof(text));
            if (imageFeatureCount < 0 || imageFeatureCount != Profile.ImageFeatureCount)
                throw new ArgumentException("Image feature count must match the fixed profile.", nameof(imageFeatureCount));
            var audioCount = audioFeatureMask == null ? 0 : AudioFeatureCount(audioFeatureMask);
            if (audioCount != Profile.AudioFeatureCount)
                throw new ArgumentException("Audio feature count must match the fixed profile.", nameof(audioFeatureMask));
            foreach (var marker in new[] {imageStart,imageEnd,audioStart,audioEnd})
                if (text.Contains(marker)) throw new ArgumentException("Supply unexpanded modality placeholders.", nameof(text));
            if (title != null && new[] {imageToken,imageStart,imageEnd,audioToken,audioStart,audioEnd}.Any(title.Contains))
                throw new ArgumentException("Titles cannot introduce modality markers.", nameof(title));
            if (Occurrences(text,imageToken) != (imageFeatureCount == 0 ? 0 : 1) || Occurrences(text,audioToken) != (audioCount == 0 ? 0 : 1))
                throw new ArgumentException("One placeholder is required for each supplied modality.", nameof(text));
            var expanded = text.Replace(imageToken, Expand(imageStart,imageToken,imageEnd,imageFeatureCount))
                .Replace(audioToken, Expand(audioStart,audioToken,audioEnd,audioCount));
            var formatted = TextPrompts.Format(expanded,role,title);
            var encoding = tokenizer.Encode(formatted);
            var tokenIds = encoding.GetIds().ToArray(); var tokenMask = encoding.GetAttentionMask().ToArray();
            if (tokenIds.Length > Profile.SequenceLength || tokenIds.Length != tokenMask.Length)
                throw new ArgumentException("The complete prompt exceeds the fixed sequence budget; truncation is disabled.", nameof(text));
            var ids = Enumerable.Repeat(Profile.PadTokenId,Profile.SequenceLength).ToArray();
            var mask = new int[Profile.SequenceLength];
            Array.Copy(tokenIds,ids,tokenIds.Length); Array.Copy(tokenMask,mask,tokenMask.Length);
            Profile.ValidateTokens(ids,mask);
            return new MultimodalPrompt(formatted,ids,mask,imageFeatureCount,audioCount);
        }

        /// <summary>Audio-only prompt from trusted PCM prefix metadata, without GPU mask readback.</summary>
        public MultimodalPrompt BuildAudio(int validFrameCount, TextRole role = TextRole.Raw, string title = null)
        {
            var profile = new AudioEncoderProfile(validFrameCount);
            if (Profile.ImageFeatureCount != 0 || Profile.AudioFeatureCount != profile.FeatureCount)
                throw new ArgumentException("Audio-only prompt must match the fixed encoder feature count.", nameof(validFrameCount));
            var mask = new int[AudioEncoderProfile.FrameCount];
            Array.Fill(mask, 1, 0, validFrameCount);
            return Build(audioToken, audioFeatureMask: mask, role: role, title: title);
        }

        static int AudioFeatureCount(int[] mask)
        {
            if (mask.Length == 0 || mask.Any(value => value != 0 && value != 1))
                throw new ArgumentException("Audio mask must be nonempty and binary.", nameof(mask));
            int count = 0;
            // Two official stride-2 reductions retain indices 0,4,8,... including the tail.
            for (var index = 0; index < mask.Length; index += 4) count += mask[index];
            if (count == 0) throw new ArgumentException("Audio must retain at least one valid feature.", nameof(mask));
            return count;
        }
        static string ReadToken(JObject configuration, string name)
        {
            var item = configuration[name];
            var value = item?.Type == JTokenType.String ? item.Value<string>() : (item as JObject)?.Value<string>("content");
            if (string.IsNullOrEmpty(value)) throw new ArgumentException("Missing multimodal token: " + name, nameof(configuration));
            return value;
        }
        static string Expand(string start, string marker, string end, int count)
        {
            var result = new StringBuilder(start);
            for (var index = 0; index < count; index++) result.Append(marker);
            return result.Append(end).ToString();
        }
        static int Occurrences(string text,string marker)
        {
            int count=0, index=0;
            while ((index=text.IndexOf(marker,index,StringComparison.Ordinal)) >= 0) { count++; index+=marker.Length; }
            return count;
        }
    }
}
