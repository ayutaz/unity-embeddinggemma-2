using System;
using System.Collections.Generic;
using System.Linq;
using Unity.InferenceEngine;

namespace EmbeddingGemma.Validation.Tests
{
    public static class ModelTensorSlice
    {
        public static Model Extract(Model source,int tensorIndex,string outputName,Model.Input[] boundaries)
            =>throw new NotImplementedException("Saved-model slicing across an external GPU boundary is missing.");
        public static Model Extract(Model source,int tensorIndex,string outputName)
        {
            if(source==null)throw new ArgumentNullException(nameof(source));
            if(string.IsNullOrEmpty(outputName))throw new ArgumentException("A diagnostic output name is required.",nameof(outputName));
            var needed=new HashSet<int>{tensorIndex};var selected=new List<Layer>();
            for(var index=source.layers.Count-1;index>=0;index--)
            {
                var layer=source.layers[index];
                if(!layer.outputs.Any(needed.Contains))continue;
                selected.Add(layer);needed.ExceptWith(layer.outputs);needed.UnionWith(layer.inputs.Where(value=>value>=0));
            }
            var constants=source.constants.Where(value=>needed.Contains(value.index)).ToList();
            var inputs=source.inputs.Where(value=>needed.Contains(value.index)).ToList();
            needed.ExceptWith(constants.Select(value=>value.index));needed.ExceptWith(inputs.Select(value=>value.index));
            if(needed.Count!=0)throw new ArgumentException("Requested tensor has an unresolved producer.",nameof(tensorIndex));
            selected.Reverse();
            return new Model{ProducerName=source.ProducerName+"/diagnostic-tensor-slice",inputs=inputs,
                constants=constants,layers=selected,outputs=new List<Model.Output>{new Model.Output{name=outputName,index=tensorIndex}}};
        }
    }
}
