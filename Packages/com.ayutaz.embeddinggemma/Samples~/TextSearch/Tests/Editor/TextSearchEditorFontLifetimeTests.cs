using System;
using System.Collections;
using System.Reflection;
using NUnit.Framework;
using UnityEditor;
using UnityEngine;
using UnityEngine.TestTools;

namespace EmbeddingGemma.Samples.Tests
{
    public sealed class TextSearchEditorFontLifetimeTests
    {
        const string FontIdKey = "EmbeddingGemma.FontLifetimeTest.FirstFontId";
        static readonly FieldInfo SampleFont = typeof(TextSearchSample).GetField("font", BindingFlags.Instance | BindingFlags.NonPublic);
        static Font FontOf(TextSearchSample sample) => (Font)SampleFont.GetValue(sample);

        static Font CachedFont(int id)
        {
            // Observe the pinned Editor cache; production code must not mutate it.
            var settingsType = Type.GetType("UnityEngine.TextCore.Text.TextSettings,UnityEngine.TextCoreTextEngineModule");
            Assert.That(settingsType, Is.Not.Null);
            var references = settingsType.GetField("m_FontReferences", BindingFlags.Instance | BindingFlags.NonPublic);
            Assert.That(references, Is.Not.Null);
            foreach (var settings in Resources.FindObjectsOfTypeAll<ScriptableObject>())
            {
                if (settings.GetType().FullName != "UnityEditor.EditorTextSettings") continue;
                foreach (var row in (IEnumerable)references.GetValue(settings))
                {
                    var font = (Font)row.GetType().GetField("font").GetValue(row);
                    if (!ReferenceEquals(font, null) && font.GetInstanceID() == id) return font;
                }
            }
            Assert.Fail("The sample font must have been registered by actual IMGUI rendering.");
            return null;
        }

        static void ShowGameView() => EditorWindow.GetWindow(Type.GetType("UnityEditor.GameView,UnityEditor")).Focus();

        [UnityTest]
        public IEnumerator TwoSampleViewsShareOneEditorFont()
        {
            yield return new EnterPlayMode();
            ShowGameView();
            var first = new GameObject("First font owner").AddComponent<TextSearchSample>();
            var second = new GameObject("Second font owner").AddComponent<TextSearchSample>();
            for (var frame = 0; frame < 30 && (FontOf(first) == null || FontOf(second) == null); frame++) yield return null;
            try
            {
                Assert.That(FontOf(first) != null && FontOf(second) != null, Is.True, "Both views must really render.");
                Assert.That(first.IsReady || second.IsReady, Is.False, "No model preparation in this test.");
                Assert.That(FontOf(first), Is.SameAs(FontOf(second)), "Editor cache ownership must be bounded to one shared font.");
            }
            finally { UnityEngine.Object.Destroy(first.gameObject); UnityEngine.Object.Destroy(second.gameObject); }
            yield return new ExitPlayMode();
        }

        [UnityTest]
        public IEnumerator EditorFontCacheStaysValidAcrossTwoPlaySessions()
        {
            yield return new EnterPlayMode();
            ShowGameView();
            var first = new GameObject("First Play font").AddComponent<TextSearchSample>();
            for (var frame = 0; frame < 30 && FontOf(first) == null; frame++) yield return null;
            Assert.That(FontOf(first) != null, Is.True, "Actual IMGUI must create the font.");
            SessionState.SetInt(FontIdKey, FontOf(first).GetInstanceID());
            Assert.That(CachedFont(SessionState.GetInt(FontIdKey, 0)), Is.SameAs(FontOf(first)));
            yield return new ExitPlayMode();
            Assert.That(CachedFont(SessionState.GetInt(FontIdKey, 0)) != null, Is.True,
                "Stop must not leave a destroyed font in the persistent Editor cache.");

            yield return new EnterPlayMode();
            ShowGameView();
            var second = new GameObject("Second Play font").AddComponent<TextSearchSample>();
            for (var frame = 0; frame < 30 && FontOf(second) == null; frame++) yield return null;
            Assert.That(FontOf(second) != null, Is.True);
            Assert.That(FontOf(second).GetInstanceID(), Is.EqualTo(SessionState.GetInt(FontIdKey, 0)),
                "A domain reload must reuse the same native font, rather than retain another font each Play.");
            yield return new ExitPlayMode();
            Assert.That(CachedFont(SessionState.GetInt(FontIdKey, 0)) != null, Is.True);
            SessionState.EraseInt(FontIdKey);
        }
    }
}
