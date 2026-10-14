using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Unity.InferenceEngine;

namespace EmbeddingGemma.Tests
{
    /// <summary>Editor candidate only. Replaces static-weight projections without changing borrowed graph objects.</summary>
    public static class CompensatedModel
    {
        static readonly MethodInfo MemberCopy = typeof(object).GetMethod("MemberwiseClone", BindingFlags.Instance | BindingFlags.NonPublic);
        static readonly FieldInfo ConstantBytes = typeof(Constant).GetField("array", BindingFlags.Instance | BindingFlags.NonPublic);

        public static Model Create(Model source, int chunk)
        {
            if (source == null) throw new ArgumentNullException(nameof(source));
            if (chunk <= 0) throw new ArgumentOutOfRangeException(nameof(chunk));
            var assembly = typeof(Worker).Assembly;
            var inference = assembly.GetType("Unity.InferenceEngine.Compiler.Analyser.PartialInferenceAnalysis");
            if (inference == null) throw new NotSupportedException("Pinned Sentis shape analysis is unavailable.");
            var context = inference.GetMethod("InferModelPartialTensors").Invoke(null, new object[] { source });
            var getPartial = context.GetType().GetMethod("GetPartialTensor");
            var constants = source.constants.ToDictionary(value => value.index);
            var result = new Model { ProducerName = source.ProducerName + "/compensated-fp32",
                inputs = new List<Model.Input>(source.inputs), outputs = new List<Model.Output>(source.outputs),
                constants = new List<Constant>(source.constants), layers = new List<Layer>() };
            var next = source.inputs.Select(value => value.index).Concat(source.constants.Select(value => value.index))
                .Concat(source.layers.SelectMany(value => value.outputs)).DefaultIfEmpty(0).Max() + 1;
            foreach (var layer in source.layers)
            {
                var kind = layer.GetType().Name;
                if ((kind != "MatMul" && kind != "MatMul2D" && kind != "Dense") || layer.inputs.Length < 2 ||
                    !constants.TryGetValue(layer.inputs[1], out var weight) || constants.ContainsKey(layer.inputs[0]))
                { result.layers.Add(layer); continue; }
                if (weight.dataType != DataType.Float || weight.shape.rank != 2 || weight.lengthBytes != weight.shape.length * 4)
                    throw new NotSupportedException("Candidate requires audited FP32 matrix weights.");
                var transposeA = layer.GetType().GetField("transposeA");
                if (transposeA != null && (bool)transposeA.GetValue(layer)) throw new NotSupportedException("Transposed data input needs independent acceptance.");
                var partial = getPartial.Invoke(context, new object[] { layer.inputs[0] });
                var shape = (DynamicTensorShape)partial.GetType().GetProperty("shape").GetValue(partial);
                if (!shape.IsStatic()) throw new NotSupportedException("Candidate requires a fixed input shape.");
                var tensorShape = shape.ToTensorShape();
                var dimensions = Enumerable.Range(0, tensorShape.rank).Select(index => tensorShape[index]).ToArray();
                var transposeB = layer.GetType().GetField("transposeB");
                var transposed = transposeB != null && (bool)transposeB.GetValue(layer);
                var width = dimensions[dimensions.Length - 1];
                var columns = weight.shape[transposed ? 0 : 1];
                if (weight.shape[transposed ? 1 : 0] != width) throw new InvalidOperationException("Projection shape identity does not match.");
                var values = Floats(weight);
                if (transposed)
                {
                    var ordered = new float[values.Length];
                    for (var row = 0; row < width; row++) for (var column = 0; column < columns; column++)
                        ordered[row * columns + column] = values[column * width + row];
                    values = ordered;
                }
                var replacement = CompensatedProjection.BuildForShape(values, columns, chunk, dimensions);
                var dense = kind == "Dense";
                var activation = dense ? layer.GetType().GetField("fusedActivation").GetValue(layer).ToString() : "None";
                if (activation != "None" && activation != "Relu") throw new NotSupportedException("Unknown fused activation.");
                var destination = dense ? next++ : layer.outputs[0];
                var map = new Dictionary<int, int> { [replacement.inputs[0].index] = layer.inputs[0] };
                if (replacement.outputs[0].index == replacement.inputs[0].index)
                {
                    result.layers.Add(NewLayer("Identity", destination, layer.inputs[0]));
                }
                else
                {
                    map[replacement.outputs[0].index] = destination;
                    foreach (var value in replacement.constants)
                        if (!map.ContainsKey(value.index)) map[value.index] = next++;
                    foreach (var value in replacement.layers.SelectMany(value => value.outputs))
                        if (value >= 0 && !map.ContainsKey(value)) map[value] = next++;
                    foreach (var value in replacement.constants)
                    {
                        var copy = (Constant)MemberCopy.Invoke(value, null);
                        copy.index = map[value.index]; result.constants.Add(copy);
                    }
                    foreach (var value in replacement.layers)
                    {
                        var copy = (Layer)MemberCopy.Invoke(value, null);
                        copy.inputs = value.inputs.Select(index => index < 0 ? index : map[index]).ToArray();
                        copy.outputs = value.outputs.Select(index => index < 0 ? index : map[index]).ToArray(); result.layers.Add(copy);
                    }
                }
                if (dense)
                {
                    if (layer.inputs.Length != 3 || !constants.ContainsKey(layer.inputs[2])) throw new NotSupportedException("Expected constant dense bias.");
                    var afterBias = activation == "None" ? layer.outputs[0] : next++;
                    result.layers.Add(NewLayer("Add", afterBias, destination, layer.inputs[2]));
                    if (activation == "Relu") result.layers.Add(NewLayer("Relu", layer.outputs[0], afterBias));
                }
            }
            var live = new HashSet<int>(result.layers.SelectMany(value => value.inputs).Concat(result.outputs.Select(value => value.index)));
            result.constants = result.constants.Where(value => live.Contains(value.index)).ToList();
            return result;
        }

        static Layer NewLayer(string name, int output, params int[] inputs)
        {
            var type = typeof(Worker).Assembly.GetTypes().Single(value => value.Name == name && typeof(Layer).IsAssignableFrom(value));
            var layer = (Layer)Activator.CreateInstance(type); layer.inputs = inputs; layer.outputs = new[] { output }; return layer;
        }
        static float[] Floats(Constant value)
        {
            var data = (ArraySegment<byte>)ConstantBytes.GetValue(value); var values = new float[value.lengthBytes / 4];
            Buffer.BlockCopy(data.Array, data.Offset, values, 0, value.lengthBytes); return values;
        }
    }
}
