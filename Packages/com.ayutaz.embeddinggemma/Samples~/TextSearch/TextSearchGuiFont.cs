using UnityEngine;

namespace EmbeddingGemma.Samples
{
#if UNITY_EDITOR
    [UnityEditor.InitializeOnLoad]
#endif
    internal static class TextSearchGuiFont
    {
#if UNITY_EDITOR
        const string EditorFontIdKey = "EmbeddingGemma.TextSearch.SharedEditorFont.InstanceId";
        static Font editorFont;

        static TextSearchGuiFont() => UnityEditor.EditorApplication.quitting += DestroyEditorFont;

        static Font FindEditorFont()
        {
            var id = UnityEditor.SessionState.GetInt(EditorFontIdKey, 0);
            foreach (var candidate in Resources.FindObjectsOfTypeAll<Font>())
                if (candidate != null && candidate.GetInstanceID() == id && candidate.hideFlags == HideFlags.HideAndDontSave)
                    return candidate;
            return null;
        }

        static void DestroyEditorFont()
        {
            var owned = editorFont != null ? editorFont : FindEditorFont();
            editorFont = null;
            UnityEditor.SessionState.EraseInt(EditorFontIdKey);
            if (owned != null) Object.DestroyImmediate(owned);
        }
#endif

        internal static Font Acquire()
        {
#if UNITY_EDITOR
            if (editorFont == null) editorFont = FindEditorFont();
            if (editorFont != null) return editorFont;
#endif
            var font = Font.CreateDynamicFontFromOSFont(new[] { "Yu Gothic UI", "Noto Sans CJK JP", "Hiragino Sans", "Arial" }, 16);
#if UNITY_EDITOR
            // IMGUI registers this font in the persistent EditorTextSettings cache.
            // Keep one Editor-owned native font across Play/domain reloads, and
            // destroy it at Editor quit. A shared font does not belong to a view.
            font.hideFlags = HideFlags.HideAndDontSave;
            UnityEditor.SessionState.SetInt(EditorFontIdKey, font.GetInstanceID());
            editorFont = font;
#endif
            return font;
        }

        internal static void Release(Font font)
        {
#if !UNITY_EDITOR
            // Player fonts retain the view's original lifetime and ownership.
            if (font != null) Object.Destroy(font);
#endif
        }
    }
}
