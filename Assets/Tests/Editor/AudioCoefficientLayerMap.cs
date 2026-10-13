using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Security.Cryptography;
using Newtonsoft.Json.Linq;
using Unity.InferenceEngine;

namespace EmbeddingGemma.Tests
{
    public static class AudioCoefficientLayerMap
    {
        public static Dictionary<string, int> Resolve(Model model, JArray anchors)
        {
            var result = new Dictionary<string, int>();
            var arrayField = typeof(Constant).GetField("array", BindingFlags.Instance | BindingFlags.NonPublic);
            if (arrayField == null) throw new NotSupportedException("Pinned Sentis constant bytes are unavailable.");
            foreach (var anchor in anchors)
            {
                var stage = anchor.Value<string>("stage");
                if (string.IsNullOrEmpty(stage) || result.ContainsKey(stage)) throw new InvalidOperationException("Stage is missing or duplicated.");
                var bytes = anchor.Value<int>("coefficient_bytes");
                if (bytes <= 0 || bytes > 1024 * 1024) throw new InvalidOperationException("Coefficient byte budget is invalid.");
                var wanted = anchor.Value<string>("coefficient_sha256");
                var constants = new HashSet<int>();
                foreach (var constant in model.constants.Where(value => value.dataType == DataType.Float && value.lengthBytes == bytes))
                {
                    var data = (ArraySegment<byte>)arrayField.GetValue(constant);
                    using var hash = SHA256.Create();
                    var sha = BitConverter.ToString(hash.ComputeHash(data.Array, data.Offset, data.Count)).Replace("-", "").ToLowerInvariant();
                    if (sha == wanted) constants.Add(constant.index);
                }
                var layers = model.layers.Where(layer => layer.GetType().Name == "Mul" && layer.inputs.Any(constants.Contains)).ToArray();
                if (layers.Length != 1 || layers[0].outputs.Length != 1)
                    throw new InvalidOperationException("Coefficient layer is missing or ambiguous: " + stage);
                result.Add(stage, layers[0].outputs[0]);
            }
            return result;
        }
    }
}
