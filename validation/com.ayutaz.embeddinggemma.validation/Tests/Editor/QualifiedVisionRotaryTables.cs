using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using Newtonsoft.Json.Linq;
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
        public static Model RebindFromNativeTables(Model source,float[][] tables)
        {
            if(source==null)throw new ArgumentNullException(nameof(source));
            if(tables==null||tables.Length!=4)throw new ArgumentException("Four native position tables are required.",nameof(tables));
            var groups=source.constants.Where(c=>c.dataType==DataType.Float&&c.shape.rank==4&&c.shape[0]==1&&c.shape[1]>0&&c.shape[2]==1&&c.shape[3]==32).GroupBy(Hash).ToArray();
            if(groups.Length!=4)throw new ArgumentException("Expected four source position table values.",nameof(source));
            var corrections=new Dictionary<string,float[]>();var used=new HashSet<int>();
            foreach(var group in groups)
            {
                var constant=group.First();var raw=(ArraySegment<byte>)ArrayField.GetValue(constant);var matches=new List<int>();
                for(var candidate=0;candidate<tables.Length;candidate++)
                {
                    var values=tables[candidate];if(values==null||values.Length!=constant.shape.length)continue;var matched=true;
                    for(var i=0;i<values.Length;i++)
                    {
                        var value=values[i];var before=BitConverter.ToSingle(raw.Array,raw.Offset+4*i);
                        if(float.IsNaN(value)||float.IsInfinity(value)||Math.Abs(value)>1f||!(Math.Abs((double)value-before)<=1e-6)){matched=false;break;}
                    }
                    if(matched)matches.Add(candidate);
                }
                if(matches.Count!=1||!used.Add(matches[0]))throw new ArgumentException("Native rotary correspondence is missing or ambiguous.",nameof(tables));
                corrections.Add(group.Key,tables[matches[0]]);
            }
            return Rebind(source,corrections);
        }
        public static Model RebindOriginalProfileForValidation(Model source,string profile)
        {
            if(profile=="square")return RebindOriginalSquareForValidation(source);
            string expected;
            if(profile=="landscape")expected="02e43d7327de4e1790eafdb0bec154c460aff8e87fd3e0d70b8969e3161747f3";
            else if(profile=="portrait")expected="eb2c3e4db71a9cffb67196d0e0abd8e886e4f7d1f49f5cd4f63a4629820fa85e";
            else throw new ArgumentException("Unknown original image profile.",nameof(profile));
            var root=Path.GetFullPath(Path.Combine(Application.dataPath,"../../numerical-acceptance-state/rot38729203226/"+profile));
            void Check(string path,string hash){using var stream=File.OpenRead(path);if(PlayerFileHash.Compute(stream,out _)!=hash)throw new InvalidDataException("Pinned original rotary file hash differs.");}
            var manifest=Path.Combine(root,"rotary-reference.json");Check(manifest,expected);var reference=JObject.Parse(File.ReadAllText(manifest));
            if(reference.Value<string>("profile")!=profile||reference.Value<string>("source")!="a5423c400846d6429ed980895503da928bc69e85"||!reference.Value<bool>("independent_of_image_input")||reference.Value<bool>("patch_executed")||reference.Value<bool>("full_encoder_executed"))
                throw new InvalidDataException("Original rotary capture provenance differs.");
            var tables=new List<float[]>();
            foreach(var name in new[]{"cos","sin"})
            {
                var row=reference["captures"][name];if(row.Value<string>("file")!=name+"-fp32.bin")throw new InvalidDataException("Unexpected rotary capture filename.");
                var path=Path.Combine(root,row.Value<string>("file"));Check(path,row.Value<string>("sha256"));var raw=File.ReadAllBytes(path);
                if(raw.Length!=2520*64*4)throw new InvalidDataException("Original rotary shape differs.");
                for(var half=0;half<2;half++)
                {
                    var values=new float[2520*32];for(var i=0;i<2520;i++)Buffer.BlockCopy(raw,(i*64+half*32)*4,values,i*32*4,32*4);tables.Add(values);
                }
            }
            return RebindFromNativeTables(source,tables.ToArray());
        }
    }
}
