using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Unity.InferenceEngine;

namespace EmbeddingGemma.Validation.Tests
{
    public static class ModelTensorSlice
    {
        public static Model Extract(Model source,Model.Output[] outputs,Model.Input[] boundaries)
            =>throw new NotImplementedException("Multiple live saved-model slice outputs are missing.");
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
            // Pinned Worker named-input lookup uses input.index as its list
            // position. Renumber every retained tensor to avoid collisions;
            // clone descriptors and layers, retaining coefficient backing bytes.
            var map=new Dictionary<int,int>();
            foreach(var input in inputs)map.Add(input.index,map.Count);
            foreach(var constant in constants)map.Add(constant.index,map.Count);
            foreach(var layer in selected)foreach(var index in layer.outputs)
                if(index>=0&&!map.ContainsKey(index))map.Add(index,map.Count);
            var clone=typeof(object).GetMethod("MemberwiseClone",BindingFlags.Instance|BindingFlags.NonPublic);
            var result=new Model{ProducerName=source.ProducerName+"/diagnostic-tensor-slice"};
            foreach(var input in inputs)result.inputs.Add(new Model.Input{name=input.name,index=map[input.index],dataType=input.dataType,shape=input.shape});
            foreach(var constant in constants)
            {
                var copy=(Constant)clone.Invoke(constant,null);copy.index=map[constant.index];result.constants.Add(copy);
            }
            foreach(var layer in selected)
            {
                var copy=(Layer)clone.Invoke(layer,null);
                copy.inputs=layer.inputs.Select(index=>index<0?index:map[index]).ToArray();
                copy.outputs=layer.outputs.Select(index=>index<0?index:map[index]).ToArray();result.layers.Add(copy);
            }
            result.outputs.Add(new Model.Output{name=outputName,index=map[tensorIndex]});return result;
        }
        public static Model Extract(Model source,int tensorIndex,string outputName)
            =>Extract(source,tensorIndex,outputName,Array.Empty<Model.Input>());
    }
}
