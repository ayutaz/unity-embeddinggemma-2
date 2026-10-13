using System;
using System.Collections.Generic;
using System.Linq;
using Unity.InferenceEngine;

namespace EmbeddingGemma.Tests
{
    /// <summary>Diagnostic candidate only: restore Pow(x,-0.5) after import without modifying the borrowed model.</summary>
    public static class NativeInversePowerCandidate
    {
        public static Model Create(Model source)
        {
            if (source == null) throw new ArgumentNullException(nameof(source));
            if (!source.layers.Any(layer => layer.GetType().Name == "Rsqrt"))
                throw new ArgumentException("Candidate requires at least one imported Rsqrt.", nameof(source));
            var exponentIndex = source.constants.Select(value => value.index).Concat(source.inputs.Select(value => value.index))
                .Concat(source.layers.SelectMany(layer => layer.outputs)).Max() + 1;
            var result = new Model { inputs = new List<Model.Input>(source.inputs), outputs = new List<Model.Output>(source.outputs),
                constants = new List<Constant>(source.constants), layers = new List<Layer>(), ProducerName = source.ProducerName };
            result.constants.Add(new Constant(exponentIndex, new TensorShape(), new[] { -0.5f }));
            var type = typeof(Model).Assembly.GetType("Unity.InferenceEngine.Layers.Pow");
            if (type == null) throw new NotSupportedException("Pinned Sentis Pow layer is unavailable.");
            foreach (var layer in source.layers)
            {
                if (layer.GetType().Name != "Rsqrt") { result.layers.Add(layer); continue; }
                var power = (Layer)Activator.CreateInstance(type);
                power.inputs = new[] { layer.inputs[0], exponentIndex };
                power.outputs = (int[])layer.outputs.Clone();
                result.layers.Add(power);
            }
            return result;
        }
    }
}
