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
        public static float[] RestoreHalfScale(float[] values)
        {
            if (values == null) throw new ArgumentNullException(nameof(values));
            var restored = new float[values.Length];
            for (var index = 0; index < values.Length; index++)
            {
                var value = values[index];
                if (float.IsNaN(value) || float.IsInfinity(value) || (value != 0f && Math.Abs(value) < 1.17549435e-38f))
                    throw new ArgumentException("Half-scale restoration requires finite normal values or zero.", nameof(values));
                restored[index] = value * 2f;
                if (float.IsInfinity(restored[index])) throw new ArgumentException("Half-scale restoration overflows.", nameof(values));
            }
            return restored;
        }

        public static Dictionary<string, int> ResolveFeedForward(Model model, JArray anchors)
        {
            var norms = Resolve(model, anchors);
            var inputs = ResolveNormInputs(model, anchors);
            if (!norms.Keys.OrderBy(value => value).SequenceEqual(new[] { "post_layer_norm", "pre_layer_norm" }))
                throw new InvalidOperationException("Expected feed-forward pre/post norm identities.");
            var constants = new HashSet<int>(model.constants.Select(value => value.index));
            int Producer(int index)
            {
                var matches = model.layers.Select((layer, number) => new { layer, number }).Where(value => value.layer.outputs.Contains(index)).ToArray();
                if (matches.Length != 1) throw new InvalidOperationException("Feed-forward producer is missing or ambiguous.");
                return matches[0].number;
            }
            int Unclip(int index)
            {
                for (var count = 0; count < 5; count++)
                {
                    var layer = model.layers[Producer(index)];
                    if (layer.GetType().Name != "Clip") return index;
                    if (layer.inputs.Length != 3 || !constants.Contains(layer.inputs[1]) || !constants.Contains(layer.inputs[2]))
                        throw new InvalidOperationException("Feed-forward clamp bounds must be constants.");
                    index = layer.inputs[0];
                }
                throw new InvalidOperationException("Feed-forward clamp chain exceeded the bound.");
            }
            int MatrixInput(int index)
            {
                var layer = model.layers[Producer(Unclip(index))];
                if (layer.GetType().Name != "MatMul" && layer.GetType().Name != "MatMul2D")
                    throw new InvalidOperationException("Expected a feed-forward projection.");
                var data = layer.inputs.Where(value => !constants.Contains(value)).ToArray();
                if (layer.inputs.Length != 2 || data.Length != 1)
                    throw new InvalidOperationException("Feed-forward projection requires one constant weight.");
                return data[0];
            }
            var second = inputs["post_layer_norm"];
            var activationIndex = Unclip(MatrixInput(second));
            var activation = model.layers[Producer(activationIndex)];
            if (activation.GetType().Name != "Swish" || activation.inputs.Length != 1)
                throw new InvalidOperationException("Expected the native SiLU/Swish path.");
            var first = activation.inputs[0];
            if (Unclip(MatrixInput(first)) != norms["pre_layer_norm"])
                throw new InvalidOperationException("Feed-forward projection is not bound to the pre norm.");
            return new Dictionary<string, int> { ["pre_layer_norm"] = norms["pre_layer_norm"],
                ["ffw_layer_1"] = first, ["act_fn"] = activationIndex,
                ["ffw_layer_2"] = second, ["post_layer_norm"] = norms["post_layer_norm"] };
        }

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
