using System;
using System.Collections.Generic;
using System.Linq;
using Unity.InferenceEngine;

namespace EmbeddingGemma.Validation.Tests
{
    public static class ModelTensorSlice
    {
        public static Model Extract(Model source,int tensorIndex,string outputName,Model.Input[] boundaries)
        {
            if(source==null)throw new ArgumentNullException(nameof(source));
            if(string.IsNullOrEmpty(outputName))throw new ArgumentException("A diagnostic output name is required.",nameof(outputName));
            if(boundaries==null || boundaries.Select(value=>value.index).Distinct().Count()!=boundaries.Length
                || boundaries.Select(value=>value.name).Distinct().Count()!=boundaries.Length
                || boundaries.Any(value=>string.IsNullOrEmpty(value.name)||!value.shape.IsStatic()
                    ||!source.layers.Any(layer=>layer.outputs.Contains(value.index))))
                throw new ArgumentException("External boundaries must be unique fixed existing layer outputs.",nameof(boundaries));
            var external=new HashSet<int>(boundaries.Select(value=>value.index));
            var needed=new HashSet<int>{tensorIndex};var selected=new List<Layer>();
            for(var index=source.layers.Count-1;index>=0;index--)
            {
                var layer=source.layers[index];
                if(!layer.outputs.Any(value=>needed.Contains(value)&&!external.Contains(value)))continue;
                if(layer.outputs.Any(external.Contains))throw new ArgumentException("External boundary overlaps another required producer output.",nameof(boundaries));
                selected.Add(layer);needed.ExceptWith(layer.outputs);needed.UnionWith(layer.inputs.Where(value=>value>=0));
            }
            var constants=source.constants.Where(value=>needed.Contains(value.index)).ToList();
            var inputs=source.inputs.Where(value=>needed.Contains(value.index)).ToList();
            inputs.AddRange(boundaries.Where(value=>needed.Contains(value.index)));
            if(inputs.Select(value=>value.name).Distinct().Count()!=inputs.Count)throw new ArgumentException("Slice input names are ambiguous.",nameof(boundaries));
            needed.ExceptWith(constants.Select(value=>value.index));needed.ExceptWith(inputs.Select(value=>value.index));
            if(needed.Count!=0)throw new ArgumentException("Requested tensor has an unresolved producer.",nameof(tensorIndex));
            selected.Reverse();
            return new Model{ProducerName=source.ProducerName+"/diagnostic-tensor-slice",inputs=inputs,
                constants=constants,layers=selected,outputs=new List<Model.Output>{new Model.Output{name=outputName,index=tensorIndex}}};
        }
        public static Model Extract(Model source,int tensorIndex,string outputName)
            =>Extract(source,tensorIndex,outputName,Array.Empty<Model.Input>());
    }
}
