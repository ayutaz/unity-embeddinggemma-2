using System;
using System.IO;
using Unity.InferenceEngine;

namespace EmbeddingGemma
{
    /// <summary>Sentis serialization; Float16 quantizes stored weights, not every operation.</summary>
    public static class TextModelFile
    {
        public static void Save(Model model, string path, bool float16 = false)
        {
            if (model == null) throw new ArgumentNullException(nameof(model));
            if (string.IsNullOrWhiteSpace(path) || !path.EndsWith(".sentis", StringComparison.OrdinalIgnoreCase))
                throw new ArgumentException("A .sentis destination is required.", nameof(path));
            path = Path.GetFullPath(path);
            Directory.CreateDirectory(Path.GetDirectoryName(path));
            var pending = path + "." + Guid.NewGuid().ToString("N") + ".pending";
            try
            {
                if (float16)
                {
                    // Quantization is destructive. Clone through disk to avoid an additional
                    // full-size MemoryStream and to keep the caller's fp32 Model unchanged.
                    ModelWriter.Save(pending, model);
                    var copy = ModelLoader.Load(pending);
                    ModelQuantizer.QuantizeWeights(QuantizationType.Float16, ref copy);
                    ModelWriter.Save(pending, copy);
                }
                else ModelWriter.Save(pending, model);
                // Never replace the previous destination with an unreadable serialization.
                ModelLoader.Load(pending);
                if (File.Exists(path)) File.Replace(pending, path, null);
                else File.Move(pending, path);
            }
            finally { if (File.Exists(pending)) File.Delete(pending); }
        }

        public static Model Load(string path) => ModelLoader.Load(path);
    }
}
