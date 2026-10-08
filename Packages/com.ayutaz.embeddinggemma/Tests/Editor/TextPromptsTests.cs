using System;
using NUnit.Framework;

namespace EmbeddingGemma.Tests
{
    public sealed class TextPromptsTests
    {
        [TestCase("猫", TextRole.Query, null, "task: search result | query: 猫")]
        [TestCase("", TextRole.Query, null, "task: search result | query: ")]
        [TestCase("猫", TextRole.Document, null, "title: none | text: 猫")]
        [TestCase("猫", TextRole.Document, "天気", "title: 天気 | text: 猫")]
        [TestCase("猫", TextRole.Document, "", "title:  | text: 猫")]
        [TestCase("", TextRole.Raw, null, "")]
        [TestCase("  \t\n猫🐈 é", TextRole.Raw, null, "  \t\n猫🐈 é")]
        public void PromptMatchesPythonWithoutChangingInput(string text, TextRole role, string title, string expected)
        {
            Assert.That(TextPrompts.Format(text, role, title), Is.EqualTo(expected));
        }

        [TestCase(TextRole.Query)]
        [TestCase(TextRole.Document)]
        [TestCase(TextRole.Raw)]
        public void NullTextIsRejected(TextRole role)
        {
            Assert.Throws<ArgumentNullException>(() => TextPrompts.Format(null, role));
        }

        [TestCase(TextRole.Query)]
        [TestCase(TextRole.Raw)]
        public void TitleOutsideDocumentsIsRejected(TextRole role)
        {
            Assert.Throws<ArgumentException>(() => TextPrompts.Format("text", role, "title"));
        }

        [Test]
        public void UnknownRoleIsRejected()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => TextPrompts.Format("text", (TextRole)999));
        }
    }
}
