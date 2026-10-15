using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Unity.InferenceEngine;

namespace EmbeddingGemma.Validation.Tests
{
    public static class SavedProjectionPartition
    {
        public static (Model model,int[] outputs) MarkGelu(Model source)
        {
            if(source==null)throw new ArgumentNullException(nameof(source));
            var selected=source.layers.Where(layer=>layer.GetType().Name=="GeluFast").ToArray();
            if(selected.Length==0||selected.Any(layer=>layer.inputs.Length!=1||layer.outputs.Length!=1))throw new ArgumentException("Expected original unary tanh GELU layers.",nameof(source));
            var identity=typeof(Worker).Assembly.GetTypes().Single(type=>type.Name=="Identity"&&typeof(Layer).IsAssignableFrom(type));
            var replacements=selected.ToDictionary(layer=>layer,layer=>
            {
                var marker=(Layer)Activator.CreateInstance(identity);marker.inputs=(int[])layer.inputs.Clone();marker.outputs=(int[])layer.outputs.Clone();return marker;
            });
            return(new Model{ProducerName=source.ProducerName+"/external-gpu-gelu",inputs=new List<Model.Input>(source.inputs),outputs=new List<Model.Output>(source.outputs),
                constants=new List<Constant>(source.constants),layers=source.layers.Select(layer=>replacements.TryGetValue(layer,out var marker)?marker:layer).ToList()},selected.Select(layer=>layer.outputs[0]).ToArray());
        }
        public sealed class Plan
        {
            public Model[] Models;
            public string[][] Inputs;
            public string[] Outputs;
            public int[] Widths,Columns,Rows;
            public float[][] Weights;
        }
        public static Plan Create(Model source,int[] selectedOutputs)
        {
            if(source==null || selectedOutputs==null || selectedOutputs.Length==0
                || selectedOutputs.Distinct().Count()!=selectedOutputs.Length || source.inputs.Count!=1 || source.outputs.Count!=1
                || source.inputs[0].dataType!=DataType.Float || !source.inputs[0].shape.IsStatic())
                throw new ArgumentException("Expected one fixed float model and unique ordered projection outputs.");
            var selectedSet=new HashSet<int>(selectedOutputs);
            var selected=source.layers.Select((layer,index)=>(layer,index)).Where(row=>row.layer.outputs.Any(selectedSet.Contains)).ToArray();
            if(selected.Length!=selectedOutputs.Length || !selected.Select(row=>row.layer.outputs[0]).SequenceEqual(selectedOutputs))
                throw new ArgumentException("Projection selection is missing, ambiguous or out of order.",nameof(selectedOutputs));
            var constants=source.constants.ToDictionary(value=>value.index);
            var inference=typeof(Worker).Assembly.GetType("Unity.InferenceEngine.Compiler.Analyser.PartialInferenceAnalysis");
            if(inference==null)throw new NotSupportedException("Pinned static shape inference is unavailable.");
            var context=inference.GetMethod("InferModelPartialTensors").Invoke(null,new object[]{source});
            var getPartial=context.GetType().GetMethod("GetPartialTensor");
            DynamicTensorShape Shape(int index)
            {
                var partial=getPartial.Invoke(context,new object[]{index});
                var shape=(DynamicTensorShape)partial.GetType().GetProperty("shape").GetValue(partial);
                if(!shape.IsStatic())throw new NotSupportedException("Projection boundary shape is not proved static.");
                return shape;
            }
            DataType Type(int index)
            {
                var partial=getPartial.Invoke(context,new object[]{index});
                return (DataType)partial.GetType().GetProperty("dataType").GetValue(partial);
            }
            var inputNames=source.inputs.ToDictionary(value=>value.index,value=>value.name);
            string Name(int index)=>inputNames.TryGetValue(index,out var name)?name:"tensor_"+index;
            var producer=source.layers.SelectMany(layer=>layer.outputs.Where(index=>index>=0).Select(index=>(index,layer))).ToDictionary(row=>row.index,row=>row.layer);
            var constantValues=new HashSet<int>(constants.Keys);
            foreach(var layer in source.layers)
                if(layer.inputs.All(index=>index<0||constantValues.Contains(index)) || layer.GetType().Name=="Shape")
                    constantValues.UnionWith(layer.outputs.Where(index=>index>=0));
            var frontier=source.inputs.Select(value=>value.index).ToList();var available=new List<int>(frontier);var seen=new HashSet<int>(frontier);
            var parts=new List<Model>();var inputs=new List<string[]>();var outputs=new List<string>();
            var widths=new List<int>();var columns=new List<int>();var rows=new List<int>();var weights=new List<float[]>();
            var removed=new HashSet<int>();var cursor=0;
            Model Part(Model.Output[] wanted)
            {
                var boundarySet=new HashSet<int>(frontier);var needed=new HashSet<int>(wanted.Select(value=>value.index));
                for(var i=source.layers.Count-1;i>=0;i--)
                {
                    var layer=source.layers[i];
                    if(!layer.outputs.Any(index=>needed.Contains(index)&&!boundarySet.Contains(index)))continue;
                    if(layer.outputs.Any(removed.Contains))throw new InvalidOperationException("An externalized projection escaped its cut.");
                    needed.ExceptWith(layer.outputs);needed.UnionWith(layer.inputs.Where(index=>index>=0));
                }
                var boundaries=frontier.Where(index=>!inputNames.ContainsKey(index)).Select(index=>new Model.Input{
                    index=index,name=Name(index),dataType=Type(index),shape=Shape(index)}).ToArray();
                if(boundaries.Any(value=>value.dataType!=DataType.Float))throw new NotSupportedException("Only GPU float carriers are qualified.");
                return ModelTensorSlice.Extract(source,wanted,boundaries);
            }
            foreach(var row in selected)
            {
                var layer=row.layer;var kind=layer.GetType().Name;
                var marker=kind=="Identity";Constant weight=null;
                if(marker&&(layer.inputs.Length!=1||layer.outputs.Length!=1))throw new ArgumentException("GPU unary marker arity differs.");
                if(!marker&&((kind!="MatMul"&&kind!="MatMul2D") || layer.inputs.Length!=2 || layer.outputs.Length!=1
                    || constants.ContainsKey(layer.inputs[0]) || !constants.TryGetValue(layer.inputs[1],out weight)
                    || weight.shape.rank!=2 || weight.dataType!=DataType.Float || weight.lengthBytes!=weight.shape.length*4))
                    throw new ArgumentException("Selected output is not an audited static FP32 projection.",nameof(selectedOutputs));
                if((bool?)layer.GetType().GetField("transposeA")?.GetValue(layer)==true)throw new NotSupportedException("Transposed data requires separate qualification.");
                var transposed=(bool?)layer.GetType().GetField("transposeB")?.GetValue(layer)??false;
                var shape=Shape(layer.inputs[0]).ToTensorShape();var width=shape[shape.rank-1];var count=marker?width:weight.shape[transposed?0:1];
                if((marker?(shape.rank<2||shape.rank>4):(shape.rank!=2&&shape.rank!=3)) || shape.length==0 || Type(layer.inputs[0])!=DataType.Float || (!marker&&weight.shape[transposed?1:0]!=width))
                    throw new NotSupportedException("Fixed GPU matrix shape or type differs.");
                var outputShape=Shape(layer.outputs[0]).ToTensorShape();
                if(outputShape.rank!=shape.rank || outputShape.length!=shape.length/width*count || outputShape[outputShape.rank-1]!=count || (marker&&outputShape!=shape))
                    throw new NotSupportedException("Projection output shape differs from its coefficients.");
                for(var index=cursor;index<row.index;index++)foreach(var value in source.layers[index].outputs)
                    if(value>=0&&seen.Add(value))available.Add(value);
                var future=new HashSet<int>(source.layers.Skip(row.index).SelectMany(value=>value.inputs).Where(index=>index>=0));
                future.UnionWith(source.outputs.Select(value=>value.index));
                var pending=new Stack<int>(future);var expanded=new HashSet<int>();
                while(pending.Count>0)
                {
                    var value=pending.Pop();
                    if(!constantValues.Contains(value)||!expanded.Add(value)||!producer.TryGetValue(value,out var parent))continue;
                    foreach(var operand in parent.inputs.Where(index=>index>=0))if(future.Add(operand))pending.Push(operand);
                }
                var live=available.Where(value=>future.Contains(value)&&!constantValues.Contains(value)).ToList();
                if(!live.Contains(layer.inputs[0]))throw new InvalidOperationException("Projection data is not live at its boundary.");
                foreach(var value in live)if(Type(value)!=DataType.Float)throw new NotSupportedException("Only GPU float carriers are qualified.");
                parts.Add(Part(live.Select(index=>new Model.Output{index=index,name=Name(index)}).ToArray()));
                inputs.Add(new[]{Name(layer.inputs[0])});outputs.Add(Name(layer.outputs[0]));
                widths.Add(width);columns.Add(count);rows.Add(shape.length/width);
                if(marker)weights.Add(null);
                else
                {
                    var raw=(ArraySegment<byte>)typeof(Constant).GetField("array",BindingFlags.Instance|BindingFlags.NonPublic).GetValue(weight);
                    var values=new float[raw.Count/4];Buffer.BlockCopy(raw.Array,raw.Offset,values,0,raw.Count);
                    if(transposed){var ordered=new float[values.Length];for(var k=0;k<width;k++)for(var n=0;n<count;n++)ordered[k*count+n]=values[n*width+k];values=ordered;}
                    weights.Add(values);
                }
                frontier=live.Concat(layer.outputs).Distinct().ToList();removed.Add(layer.outputs[0]);
                if(seen.Add(layer.outputs[0]))available.Add(layer.outputs[0]);cursor=row.index+1;
            }
            parts.Add(Part(source.outputs.ToArray()));
            return new Plan{Models=parts.ToArray(),Inputs=inputs.ToArray(),Outputs=outputs.ToArray(),Widths=widths.ToArray(),Columns=columns.ToArray(),Rows=rows.ToArray(),Weights=weights.ToArray()};
        }
    }
}
