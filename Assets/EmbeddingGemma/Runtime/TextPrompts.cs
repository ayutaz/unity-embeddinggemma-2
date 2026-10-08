using System;

namespace EmbeddingGemma
{
    public enum TextRole { Query, Document, Raw }

    /// <summary>Formats inputs exactly as the pinned Python reference.</summary>
    public static class TextPrompts
    {
        public static string Format(string text, TextRole role, string title = null)
        {
            if (text == null) throw new ArgumentNullException(nameof(text));
            if (role != TextRole.Query && role != TextRole.Document && role != TextRole.Raw)
                throw new ArgumentOutOfRangeException(nameof(role));
            if (title != null && role != TextRole.Document)
                throw new ArgumentException("Titles are only supported for documents.", nameof(title));
            switch (role)
            {
                case TextRole.Query: return "task: search result | query: " + text;
                case TextRole.Document: return "title: " + (title ?? "none") + " | text: " + text;
                default: return text;
            }
        }
    }
}
