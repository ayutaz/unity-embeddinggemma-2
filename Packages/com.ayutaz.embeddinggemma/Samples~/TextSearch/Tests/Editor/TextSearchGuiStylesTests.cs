using System;
using NUnit.Framework;
using UnityEngine;

namespace EmbeddingGemma.Samples.Tests
{
    public sealed class TextSearchGuiStylesTests
    {
        Font font;
        GUISkin skin;
        TextSearchGuiStyles styles;
        Font originalSkinFont;
        int originalSkinSize;
        [SetUp] public void Setup()
        {
            font = TextSearchGuiFont.Acquire();
            skin = ScriptableObject.CreateInstance<GUISkin>();
            originalSkinFont = skin.label.font; originalSkinSize = skin.label.fontSize;
            styles = new TextSearchGuiStyles(skin, font);
        }
        [TearDown] public void Cleanup() { styles.Dispose(); UnityEngine.Object.DestroyImmediate(skin); TextSearchGuiFont.Release(font); }
        float GlyphSpan(string text, int size)
        {
            // The legacy glyph API needs a fresh font after a domain reload.
            // Measure the same OS family without passing this temporary probe
            // to GUIStyle, which would register it in the persistent cache.
            var probe = Font.CreateDynamicFontFromOSFont(font.name, size);
            try
            {
                probe.RequestCharactersInTexture(text, size);
                var low = int.MaxValue; var high = int.MinValue;
                foreach (var c in text)
                    if (probe.GetCharacterInfo(c, out var info, size)) { low = Math.Min(low, info.minY); high = Math.Max(high, info.maxY); }
                Assert.That(high, Is.GreaterThan(low), "The actual OS font must provide test glyphs.");
                return high - low;
            }
            finally { UnityEngine.Object.DestroyImmediate(probe); }
        }
        [TestCase("label")] [TestCase("field")] [TestCase("button")] [TestCase("heading")]
        public void LayoutFitsTheActualFontGlyphBounds(string name)
        {
            var style = name == "field" ? styles.TextField : name == "button" ? styles.Button : name == "heading" ? styles.Heading : styles.Label;
            var size = name == "heading" ? 22 : 16;
            const string text = "Agjp 猫を健康に";
            var required = GlyphSpan(text, size) + style.padding.vertical;
            var actual = style.CalcSize(new GUIContent(text)).y;
            Assert.That(actual, Is.GreaterThanOrEqualTo(required), name + " row must fit ascenders and descenders of the rendered font.");
        }
        [Test] public void WrappedDocumentRowsFitBothLinesOfTheActualFont()
        {
            const string line = "Agjp 猫を健康に";
            var required = 2 * GlyphSpan(line, 16) + styles.Label.padding.vertical;
            Assert.That(styles.Label.CalcHeight(new GUIContent(line + "\n" + line), 800), Is.GreaterThanOrEqualTo(required));
            Assert.That(styles.Label.wordWrap, Is.True, "Long document text must wrap in a narrow viewport.");
        }
        [Test] public void DoesNotChangeSharedSkinAndClearsFontReferencesBeforeDestroy()
        {
            Assert.That(skin.label.font, Is.SameAs(originalSkinFont)); Assert.That(skin.label.fontSize, Is.EqualTo(originalSkinSize));
            styles.Dispose();
            foreach (var style in new[] { styles.Label, styles.Heading, styles.Error, styles.TextField, styles.Button })
                Assert.That(style.font, Is.Null, "Cached GUIStyles must release native font references.");
            Assert.That(font != null, Is.True, "Disposing styles must not destroy the owner-managed font.");
        }
    }
}
