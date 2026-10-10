using System;
using UnityEngine;

namespace EmbeddingGemma.Samples
{
    internal sealed class TextSearchGuiStyles : IDisposable
    {
        internal readonly GUIStyle Label, Heading, Error, TextField, Button;
        internal TextSearchGuiStyles(GUISkin skin, Font font)
        {
            if (skin == null) throw new ArgumentNullException(nameof(skin));
            if (font == null) throw new ArgumentNullException(nameof(font));
            // Inherited styles with a null font can measure the built-in font while
            // rendering the OS font. Explicit font/size keeps layout and glyphs aligned.
            GUIStyle Body(GUIStyle prototype) => new(prototype) { font = font, fontSize = 16 };
            Label = Body(skin.label); Label.wordWrap = true;
            Heading = new GUIStyle(Label) { fontSize = 22 };
            Error = new GUIStyle(Label); Error.normal.textColor = Color.red;
            TextField = Body(skin.textField);
            Button = Body(skin.button);
        }
        public void Dispose()
        {
            foreach (var style in new[] { Label, Heading, Error, TextField, Button }) style.font = null;
        }
    }
}
