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
        public static Dictionary<string, int> ResolveNormInputs(Model model, JArray anchors)
        {
            var outputs = Resolve(model, anchors);
            var result = new Dictionary<string, int>();
            var constants = new HashSet<int>(model.constants.Select(value => value.index));
            int Producer(int index)
            {
                var matches = model.layers.Select((layer, number) => new { layer, number }).Where(value => value.layer.outputs.Contains(index)).ToArray();
                if (matches.Length != 1) throw new InvalidOperationException("Norm producer is missing or ambiguous.");
                return matches[0].number;
            }
            foreach (var row in outputs)
            {
                var scale = model.layers[Producer(row.Value)];
                var normalized = scale.inputs.Where(index => !constants.Contains(index)).ToArray();
                if (normalized.Length != 1) throw new InvalidOperationException("Norm scale input is ambiguous.");
                var multiply = model.layers[Producer(normalized[0])];
                if (multiply.GetType().Name != "Mul" || multiply.inputs.Length != 2)
                    throw new InvalidOperationException("Expected a normalized product.");
                var inverses = multiply.inputs.Where(index => model.layers.Any(layer => layer.GetType().Name == "Rsqrt" && layer.outputs.Contains(index))).ToArray();
                if (inverses.Length != 1) throw new InvalidOperationException("Expected one norm inverse square root.");
                var input = multiply.inputs.Single(index => index != inverses[0]);
                var inverse = model.layers[Producer(inverses[0])];
                var sum = model.layers[Producer(inverse.inputs[0])];
                int[] moments;
                if (sum.GetType().Name == "ScalarMad" && sum.inputs.Length == 1)
                {
                    var multiplier = (float)sum.GetType().GetField("sFloat").GetValue(sum);
                    var epsilon = (float)sum.GetType().GetField("bFloat").GetValue(sum);
                    if (multiplier != 1f || epsilon <= 0 || float.IsInfinity(epsilon) || float.IsNaN(epsilon))
                        throw new InvalidOperationException("Contracted norm epsilon is invalid.");
                    moments = sum.inputs;
                }
                else if (sum.GetType().Name == "Add" && sum.inputs.Length == 2)
                {
                    moments = sum.inputs.Where(index => !constants.Contains(index)).ToArray();
                    if (moments.Length != 1) throw new InvalidOperationException("Norm epsilon must be a constant.");
                }
                else throw new InvalidOperationException("Expected a norm epsilon sum.");
                var mean = model.layers[Producer(moments[0])];
                if (mean.GetType().Name != "ReduceMean") throw new InvalidOperationException("Expected a squared mean.");
                var square = model.layers[Producer(mean.inputs[0])];
                if (square.GetType().Name != "Square" || square.inputs.Length != 1 || square.inputs[0] != input)
                    throw new InvalidOperationException("Squared norm input identity differs.");
                result.Add(row.Key, input);
            }
            return result;
        }

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
