using System;
using System.Linq;
using NUnit.Framework;

namespace EmbeddingGemma.Tests
{
    public sealed class TextTokenizerTests
    {
        const string Json = @"{
          ""added_tokens"": [], ""normalizer"": null,
          ""pre_tokenizer"": {""type"": ""WhitespaceSplit""}, ""decoder"": null,
          ""truncation"": null,
          ""padding"": {""strategy"": {""Fixed"": 6}, ""direction"": ""Right"", ""pad_to_multiple_of"": null, ""pad_id"": 7, ""pad_type_id"": 0, ""pad_token"": ""[PAD]""},
          ""post_processor"": {""type"": ""TemplateProcessing"",
            ""single"": [{""SpecialToken"": {""id"": ""<start>"", ""type_id"": 0}}, {""Sequence"": {""id"": ""A"", ""type_id"": 0}}, {""SpecialToken"": {""id"": ""<end>"", ""type_id"": 0}}],
            ""pair"": [{""Sequence"": {""id"": ""A"", ""type_id"": 0}}, {""Sequence"": {""id"": ""B"", ""type_id"": 1}}],
            ""special_tokens"": {""<start>"": {""id"": ""<start>"", ""ids"": [11], ""tokens"": [""<start>""]}, ""<end>"": {""id"": ""<end>"", ""ids"": [12], ""tokens"": [""<end>""]}}},
          ""model"": {""type"": ""WordLevel"", ""vocab"": {""[UNK]"": 0, ""hello"": 5}, ""unk_token"": ""[UNK]""}
        }";

        [Test]
        public void EmptyInputUsesConfiguredSpecialTokensAndPadding()
        {
            var encoded = new TextTokenizer(Json).Encode("");
            Assert.That(encoded.GetIds().ToArray(), Is.EqualTo(new[] { 11, 12, 7, 7, 7, 7 }));
            Assert.That(encoded.GetAttentionMask().ToArray(), Is.EqualTo(new[] { 1, 1, 0, 0, 0, 0 }));
        }

        [Test]
        public void NonEmptyInputStillUsesTheOriginalVocabulary()
        {
            var encoded = new TextTokenizer(Json).Encode("hello");
            Assert.That(encoded.GetIds().ToArray(), Is.EqualTo(new[] { 11, 5, 12, 7, 7, 7 }));
        }

        [Test]
        public void NullInputAndNullConfigurationAreRejected()
        {
            Assert.Throws<ArgumentNullException>(() => new TextTokenizer(null));
            Assert.Throws<ArgumentNullException>(() => new TextTokenizer(Json).Encode(null));
        }
    }
}
