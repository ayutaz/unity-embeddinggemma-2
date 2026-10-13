using System;
using System.Linq;
using System.Reflection;
using NUnit.Framework;

namespace EmbeddingGemma.Tests
{
    public sealed class MultimodalPromptBuilderTests
    {
        const string Configuration = "{\"image_token\":\"<image>\",\"boi_token\":\"<boi>\",\"eoi_token\":\"<eoi>\",\"audio_token\":\"<audio>\",\"boa_token\":\"<boa>\",\"eoa_token\":\"<eoa>\"}";
        const string Json = @"{
          ""added_tokens"": [
            {""id"":10,""content"":""<image>"",""special"":true,""single_word"":false,""lstrip"":false,""rstrip"":false,""normalized"":false},
            {""id"":11,""content"":""<audio>"",""special"":true,""single_word"":false,""lstrip"":false,""rstrip"":false,""normalized"":false},
            {""id"":12,""content"":""<boi>"",""special"":true,""single_word"":false,""lstrip"":false,""rstrip"":false,""normalized"":false},
            {""id"":13,""content"":""<eoi>"",""special"":true,""single_word"":false,""lstrip"":false,""rstrip"":false,""normalized"":false},
            {""id"":14,""content"":""<boa>"",""special"":true,""single_word"":false,""lstrip"":false,""rstrip"":false,""normalized"":false},
            {""id"":15,""content"":""<eoa>"",""special"":true,""single_word"":false,""lstrip"":false,""rstrip"":false,""normalized"":false}],
          ""normalizer"":null,""pre_tokenizer"":{""type"":""WhitespaceSplit""},""decoder"":null,
          ""truncation"":{""direction"":""Right"",""max_length"":6,""strategy"":""LongestFirst"",""stride"":0},
          ""padding"":{""strategy"":{""Fixed"":6},""direction"":""Right"",""pad_to_multiple_of"":null,""pad_id"":0,""pad_type_id"":0,""pad_token"":""<pad>""},
          ""post_processor"":{""type"":""TemplateProcessing"",
            ""single"":[{""SpecialToken"":{""id"":""<bos>"",""type_id"":0}},{""Sequence"":{""id"":""A"",""type_id"":0}},{""SpecialToken"":{""id"":""<eos>"",""type_id"":0}}],
            ""pair"":[{""Sequence"":{""id"":""A"",""type_id"":0}},{""Sequence"":{""id"":""B"",""type_id"":1}}],
            ""special_tokens"":{""<bos>"":{""id"":""<bos>"",""ids"":[1],""tokens"":[""<bos>""]},""<eos>"":{""id"":""<eos>"",""ids"":[2],""tokens"":[""<eos>""]}}},
          ""model"":{""type"":""WordLevel"",""vocab"":{""[UNK]"":3,""hello"":5,""then"":6},""unk_token"":""[UNK]""}}
        ";

        [TestCase(false)]
        [TestCase(true)]
        public void MixedPromptKeepsFeatureOrderAndIgnoresOldTextTruncation(bool audioFirst)
        {
            var builder = Make(image: 2, audio: 2);
            var result = Build(builder, audioFirst ? "<audio> then <image>" : "<image> then <audio>", 2, new[] {1,1,1,1,1,1,1});
            var expected = audioFirst ? new[] {1,14,11,11,15,6,12,10,10,13,2,0,0,0,0,0}
                : new[] {1,12,10,10,13,6,14,11,11,15,2,0,0,0,0,0};
            Assert.That(Ids(result), Is.EqualTo(expected));
            Assert.That(Mask(result), Is.EqualTo(Enumerable.Repeat(1,11).Concat(Enumerable.Repeat(0,5)).ToArray()));
        }

        [TestCase(256)]
        [TestCase(252)]
        public void ImageUsesActualFeatureCountRatherThanTheSynthetic280(int features)
        {
            var result = Build(Make(length: 264, image: features), "<image>", features);
            Assert.That(Ids(result).Count(id => id == 10), Is.EqualTo(features));
            Assert.That(Ids(result).Take(3), Is.EqualTo(new[] {1,12,10}));
            Assert.That(Ids(result)[features+2], Is.EqualTo(13));
            Assert.That(Ids(result)[features+3], Is.EqualTo(2));
        }

        [TestCase(1,1)] [TestCase(4,1)] [TestCase(5,2)] [TestCase(8,2)] [TestCase(9,3)]
        public void AudioUsesTwoStride2MaskReductions(int frames, int features)
        {
            var result = Build(Make(audio: features), "<audio>", 0, Enumerable.Repeat(1,frames).ToArray());
            Assert.That(Ids(result).Take(features+4), Is.EqualTo(new[] {1,14}.Concat(Enumerable.Repeat(11,features)).Concat(new[] {15,2})));
        }

        [Test]
        public void AudioCountsTheSubsampledMaskRatherThanTheOriginalSum()
        {
            var result = Build(Make(audio: 2), "<audio>", 0, new[] {1,1,1,1,0,0,0,0,1});
            Assert.That(Ids(result).Count(id => id == 11), Is.EqualTo(2));
        }

        [Test]
        public void OverlongPromptIsRejectedRatherThanTruncated()
        {
            Assert.Throws<ArgumentException>(() => Build(Make(length: 8, image: 2), "<image> " + string.Join(" ", Enumerable.Repeat("hello",20)), 2));
        }

        [TestCase("<image> <image>",2)]
        [TestCase("hello",2)]
        [TestCase("<image>",1)]
        [TestCase("<boi><image><eoi>",2)]
        public void InvalidImageLayoutOrProfileIsRejected(string template, int features) =>
            Assert.Throws<ArgumentException>(() => Build(Make(image: 2), template, features));

        [TestCase(new[] {1,2,1,1,1})]
        [TestCase(new[] {0,0,0,0,0})]
        [TestCase(new int[0])]
        [TestCase(null)]
        public void InvalidOrMissingAudioMaskIsRejected(int[] mask) =>
            Assert.Throws<ArgumentException>(() => Build(Make(audio: 1), "<audio>", 0, mask));

        [Test]
        public void WrongAudioProfileCountIsRejected() =>
            Assert.Throws<ArgumentException>(() => Build(Make(audio: 1), "<audio>", 0, Enumerable.Repeat(1,9).ToArray()));

        [Test]
        public void DocumentRoleUsesTheExistingOfficialTextPrefix()
        {
            var result = Build(Make(image: 2), "<image>", 2, null, TextRole.Document, "hello");
            Assert.That(Property<string>(result,"FormattedText"), Is.EqualTo("title: hello | text: <boi><image><image><eoi>"));
        }

        [Test]
        public void PromptResultOwnsItsArraysAndCanBeReused()
        {
            var builder = Make(image: 2);
            var result = Build(builder,"<image>",2);
            var ids = Ids(result); var mask = Mask(result); ids[0]=31; mask[0]=0;
            Build(builder,"hello <image>",2);
            Assert.That(Ids(result)[0], Is.EqualTo(1));
            Assert.That(Mask(result)[0], Is.EqualTo(1));
        }

        [Test]
        public void InvalidSpecialTokenConfigurationAndWrongProfileIdsAreRejected()
        {
            Assert.Throws<ArgumentException>(() => Make(configuration: Configuration.Replace("<image>", "")));
            Assert.Throws<ArgumentException>(() => Make(configuration: Configuration.Replace("<audio>", "<image>")));
            Assert.Throws<ArgumentException>(() => Make(profileImageId: 20));
        }

        [Test]
        public void NullInputsAreRejected()
        {
            Assert.Throws<ArgumentNullException>(() => Make(json: null));
            Assert.Throws<ArgumentNullException>(() => Make(configuration: null));
            Assert.Throws<ArgumentNullException>(() => Build(Make(), null));
        }

        static object Make(string json=Json, string configuration=Configuration, int length=16, int image=0, int audio=0, int profileImageId=10)
        {
            var type = AppDomain.CurrentDomain.GetAssemblies().Select(a => a.GetType("EmbeddingGemma.MultimodalPromptBuilder")).FirstOrDefault(t => t != null);
            Assert.That(type, Is.Not.Null, "The multimodal prompt Runtime API is missing.");
            return Unwrap(() => Activator.CreateInstance(type, new object[] {json, configuration, new MultimodalBackboneProfile(length,profileImageId,11,image,audio,32,0)}));
        }
        static object Build(object builder, string text, int image=0, int[] audio=null, TextRole role=TextRole.Raw, string title=null) =>
            Unwrap(() => builder.GetType().GetMethod("Build").Invoke(builder,new object[] {text,image,audio,role,title}));
        static object Unwrap(Func<object> call)
        {
            try { return call(); }
            catch (TargetInvocationException error) { System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(error.InnerException).Throw(); throw; }
        }
        static T Property<T>(object result,string name) => (T)result.GetType().GetProperty(name).GetValue(result);
        static int[] Ids(object result) => Property<int[]>(result,"InputIds");
        static int[] Mask(object result) => Property<int[]>(result,"AttentionMask");
    }
}
