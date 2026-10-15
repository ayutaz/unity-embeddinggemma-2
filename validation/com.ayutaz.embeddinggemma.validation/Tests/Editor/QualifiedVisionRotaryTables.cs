using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using Unity.InferenceEngine;
using UnityEngine;

namespace EmbeddingGemma.Validation.Tests
{
    /// <summary>Validation-only bounded correction of derived position tables; never changes borrowed graph or learned tensors.</summary>
    public static class QualifiedVisionRotaryTables
    {
        static readonly FieldInfo ArrayField=typeof(Constant).GetField("array",BindingFlags.Instance|BindingFlags.NonPublic);
        static string Hash(Constant value)
        {
            var raw=(ArraySegment<byte>)ArrayField.GetValue(value);using var stream=new MemoryStream(raw.Array,raw.Offset,raw.Count,false);
            return PlayerFileHash.Compute(stream,out _);
        }
        public static Model Rebind(Model source,IReadOnlyDictionary<string,float[]> corrections)
        {
            if(source==null)throw new ArgumentNullException(nameof(source));
            if(corrections==null||corrections.Count!=4)throw new ArgumentException("Four audited derived position tables are required.",nameof(corrections));
            var targets=source.constants.Where(c=>c.dataType==DataType.Float&&c.shape.rank==4&&c.shape[0]==1&&c.shape[1]>0&&c.shape[2]==1&&c.shape[3]==32).ToArray();
            var groups=targets.GroupBy(Hash).ToArray();
            if(groups.Length!=4||!new HashSet<string>(groups.Select(g=>g.Key)).SetEquals(corrections.Keys)||targets.Select(c=>c.shape).Distinct().Count()!=1)
                throw new ArgumentException("Correction hashes/shapes do not identify the four source position tables.",nameof(corrections));
            var indices=new HashSet<int>(targets.Select(c=>c.index));
            if(source.layers.Where(l=>l.inputs.Any(indices.Contains)).Any(l=>l.GetType().Name!="Mul"))
                throw new ArgumentException("Position tables must only feed the original rotary multiplications.",nameof(source));
            foreach(var group in groups)
            {
                var constant=group.First();var values=corrections[group.Key];
                if(values==null||values.Length!=constant.shape.length)throw new ArgumentException("Position table length differs.",nameof(corrections));
                var raw=(ArraySegment<byte>)ArrayField.GetValue(constant);
                for(var i=0;i<values.Length;i++)
                {
                    var before=BitConverter.ToSingle(raw.Array,raw.Offset+4*i);var after=values[i];
                    if(float.IsNaN(after)||float.IsInfinity(after)||Math.Abs(after)>1f||Math.Abs((double)after-before)>1e-6)
                        throw new ArgumentException("Position correction must be finite, bounded and within 1e-6 of the audited derived table.",nameof(corrections));
                }
            }
            var next=source.constants.Select(c=>c.index).Concat(source.inputs.Select(i=>i.index)).Concat(source.layers.SelectMany(l=>l.outputs)).DefaultIfEmpty(-1).Max()+1;
            var map=new Dictionary<int,int>();var constants=new List<Constant>(source.constants);
            foreach(var group in groups)
            {
                var index=next++;constants.Add(new Constant(index,group.First().shape,(float[])corrections[group.Key].Clone()));
                foreach(var constant in group)map.Add(constant.index,index);
            }
            var clone=typeof(object).GetMethod("MemberwiseClone",BindingFlags.Instance|BindingFlags.NonPublic);var layers=new List<Layer>();
            foreach(var layer in source.layers)
            {
                if(!layer.inputs.Any(map.ContainsKey)){layers.Add(layer);continue;}
                var copy=(Layer)clone.Invoke(layer,null);copy.inputs=layer.inputs.Select(i=>map.TryGetValue(i,out var replacement)?replacement:i).ToArray();copy.outputs=(int[])layer.outputs.Clone();layers.Add(copy);
            }
            return new Model{ProducerName=source.ProducerName+"/qualified-derived-rotary",inputs=new List<Model.Input>(source.inputs),outputs=new List<Model.Output>(source.outputs),constants=constants,layers=layers};
        }
        public static Model RebindOriginalSquareForValidation(Model source)
        {
            var root=Path.GetFullPath(Path.Combine(Application.dataPath,"../../numerical-acceptance-state/r38716679261"));
            float[][] Halves(string file,string expected)
            {
                var path=Path.Combine(root,file);using(var stream=File.OpenRead(path))if(PlayerFileHash.Compute(stream,out _)!=expected)throw new InvalidDataException("Original native rotary capture hash differs.");
                var raw=File.ReadAllBytes(path);if(raw.Length!=2520*64*4)throw new InvalidDataException("Original rotary shape differs.");
                var result=new[]{new float[2520*32],new float[2520*32]};
                for(var row=0;row<2520;row++)for(var half=0;half<2;half++)Buffer.BlockCopy(raw,(row*64+half*32)*4,result[half],row*32*4,32*4);
                return result;
            }
            var cos=Halves("cat-fp32.bin","8e7ad14c37f14fb5b11e70b08701181c67ad24f4a874385eee5dd63655b39b75");
            var sin=Halves("cat_1-fp32.bin","1eeda3ae4a8b5bf7f09e29b6d6c763054d9b49ec6de551e09d54a080232a331e");
            return Rebind(source,new Dictionary<string,float[]>{
                ["622f85993d68e4c044e47b8e2052d63fc291c876b7dae56d5b2758ba36f76ffe"]=cos[0],
                ["cf3705ffb0b26cee4d3374d0f7c822b3ac5f9c4f8d53959dcba5727e1acfd5e1"]=sin[0],
                ["b4b3f08873b84818e3eef4ebcc4b8bd43a3718b6a91bd70b09b0c744ffb4f5ba"]=cos[1],
                ["a8e17435db4001e3bb11fa9e174ebafecb0005e902a27ee8e9e6fe525094caa1"]=sin[1]});
        }
    }
}
