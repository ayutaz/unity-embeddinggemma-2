using System;
using System.Collections.Generic;
using System.Linq;
using Newtonsoft.Json.Linq;
using Unity.InferenceEngine;

namespace EmbeddingGemma.Tests
{
    /// <summary>Diagnostic only. Borrows norm layers/constants without editing the full model.</summary>
    public static class AudioNormIsolation
    {
        public static Model Create(Model source, JArray anchors, TensorShape shape)
        {
            if (source == null) throw new ArgumentNullException(nameof(source));
            if (anchors == null || anchors.Count != 1) throw new ArgumentException("Select exactly one norm anchor.", nameof(anchors));
            if (shape.rank < 1 || shape.HasZeroDims()) throw new ArgumentException("Norm input shape must be nonempty.", nameof(shape));
            var input = AudioCoefficientLayerMap.ResolveNormInputs(source, anchors).Single().Value;
            var output = AudioCoefficientLayerMap.Resolve(source, anchors).Single().Value;
            var constants = source.constants.ToDictionary(value => value.index);
            var neededConstants = new HashSet<int>(); var neededLayers = new HashSet<Layer>(); var visited = new HashSet<int>();
            var pending = new Stack<int>(); pending.Push(output);
            while (pending.Count > 0)
            {
                var index = pending.Pop();
                if (index < 0 || index == input || !visited.Add(index)) continue;
                if (constants.ContainsKey(index)) { neededConstants.Add(index); continue; }
                var producers = source.layers.Where(layer => layer.outputs.Contains(index)).ToArray();
                if (producers.Length != 1) throw new InvalidOperationException("Norm slice has an unbound or ambiguous dependency.");
                var producer = producers[0]; neededLayers.Add(producer);
                foreach (var dependency in producer.inputs) pending.Push(dependency);
            }
            if (neededLayers.Count != 6) throw new InvalidOperationException("Expected the pinned six-layer RMSNorm path.");
            if (neededConstants.Contains(0)) throw new InvalidOperationException("Norm constant collides with the required input index zero.");
            var layers = new List<Layer>();
            foreach (var layer in source.layers.Where(neededLayers.Contains))
            {
                if (input == 0 || !layer.inputs.Contains(input)) { layers.Add(layer); continue; }
                if (layer.GetType().Name != "Square" && layer.GetType().Name != "Mul")
                    throw new InvalidOperationException("Unexpected norm input consumer.");
                var copy = (Layer)Activator.CreateInstance(layer.GetType());
                copy.inputs = layer.inputs.Select(index => index == input ? 0 : index).ToArray();
                copy.outputs = (int[])layer.outputs.Clone(); layers.Add(copy);
            }
            return new Model { ProducerName = source.ProducerName,
                inputs = new List<Model.Input> { new Model.Input { name = "input", index = 0, dataType = DataType.Float, shape = new DynamicTensorShape(shape) } },
                outputs = new List<Model.Output> { new Model.Output { name = "output", index = output } },
                layers = layers,
                constants = source.constants.Where(value => neededConstants.Contains(value.index)).ToList() };
        }
    }
}
