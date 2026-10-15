using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using EmbeddingGemma.Tests;
using NUnit.Framework;
using Unity.InferenceEngine;

namespace EmbeddingGemma.Validation.Tests
{
    /// <summary>Fixed vision RMS diagnostic operation; owns Workers/division, borrows the audited template.</summary>
    public sealed class QualifiedVisionRmsOperation : IGpuTensorOperation
    {
        GpuProjectionPipeline pipeline;
        GpuScalarDivide divide;
        public BackendType Backend=>BackendType.GPUCompute;
        public static Model InlineTailRms(Model source,Model template)
        {
            if(source==null||template==null)throw new ArgumentNullException();
            Assert.That(template.inputs.Count,Is.EqualTo(1));Assert.That(template.outputs.Count,Is.EqualTo(1));
            var square=source.layers.Single(layer=>layer.GetType().Name=="Square");
            var analysis=typeof(Worker).Assembly.GetType("Unity.InferenceEngine.Compiler.Analyser.PartialInferenceAnalysis");
            if(analysis==null)throw new NotSupportedException("Pinned tail shape analysis is unavailable.");
            var context=analysis.GetMethod("InferModelPartialTensors").Invoke(null,new object[]{source});
            var partial=context.GetType().GetMethod("GetPartialTensor").Invoke(context,new object[]{square.inputs[0]});
            var inferred=(DynamicTensorShape)partial.GetType().GetProperty("shape").GetValue(partial);
            if(!inferred.IsStatic()||!template.inputs[0].shape.IsStatic())
                throw new ArgumentException("Tail RMS replacement requires a proven fixed intermediate shape.");
            var target=inferred.ToTensorShape();
            if(target.rank!=2||target[1]!=768||template.inputs[0].shape.ToTensorShape()!=target)
                throw new ArgumentException("Tail RMS template must match the source profile's actual width-768 intermediate.",nameof(template));
            Layer Next(Layer current,string kind)=>source.layers.Single(layer=>layer.GetType().Name==kind&&layer.inputs.Contains(current.outputs[0]));
            var mean=Next(square,"ReduceMean");var scalar=Next(mean,"ScalarMad");var inverse=Next(scalar,"Rsqrt");var final=Next(inverse,"Mul");
            Assert.That(final.inputs.Contains(square.inputs[0]),Is.True);
            Assert.That((float)scalar.GetType().GetField("sFloat").GetValue(scalar),Is.EqualTo(1f));
            Assert.That((float)scalar.GetType().GetField("bFloat").GetValue(scalar),Is.EqualTo(1e-6f));
            var chain=new HashSet<Layer>{square,mean,scalar,inverse,final};
            foreach(var layer in chain.Where(layer=>!ReferenceEquals(layer,final)))
            {
                Assert.That(source.outputs.Any(output=>layer.outputs.Contains(output.index)),Is.False);
                Assert.That(source.layers.Where(consumer=>consumer.inputs.Intersect(layer.outputs).Any()).All(chain.Contains),Is.True);
            }
            var next=source.inputs.Select(value=>value.index).Concat(source.constants.Select(value=>value.index)).Concat(source.layers.SelectMany(layer=>layer.outputs.Concat(layer.inputs))).Max()+1;
            var map=new Dictionary<int,int>{{template.inputs[0].index,square.inputs[0]},{template.outputs[0].index,final.outputs[0]}};
            var clone=typeof(object).GetMethod("MemberwiseClone",BindingFlags.Instance|BindingFlags.NonPublic);
            var constants=new List<Constant>(source.constants);
            foreach(var constant in template.constants)
            {
                var copy=(Constant)clone.Invoke(constant,null);copy.index=next++;map.Add(constant.index,copy.index);constants.Add(copy);
            }
            foreach(var layer in template.layers)foreach(var output in layer.outputs)if(output>=0&&!map.ContainsKey(output))map.Add(output,next++);
            var replacements=new List<Layer>();
            foreach(var layer in template.layers)
            {
                var copy=(Layer)clone.Invoke(layer,null);copy.inputs=layer.inputs.Select(index=>index<0?index:map[index]).ToArray();copy.outputs=layer.outputs.Select(index=>index<0?index:map[index]).ToArray();replacements.Add(copy);
            }
            var result=new Model{ProducerName=source.ProducerName+"/inline-unscaled-tail-rms",inputs=new List<Model.Input>(source.inputs),outputs=new List<Model.Output>(source.outputs),constants=constants};
            foreach(var layer in source.layers)
                if(ReferenceEquals(layer,square))result.layers.AddRange(replacements);
                else if(!chain.Contains(layer))result.layers.Add(layer);
            return result;
        }
        public static Model InlineLayerRms(Model source,Model hiddenTemplate,Model scaledHeadTemplate,Model unscaledHeadTemplate,int layerCount)
        {
            if(source==null||hiddenTemplate==null||scaledHeadTemplate==null||unscaledHeadTemplate==null||layerCount<1||layerCount>16)
                throw new ArgumentException("Expected audited fixed vision RMS templates and one to sixteen layers.");
            var squares=source.layers.Where(layer=>layer.GetType().Name=="Square").Take(layerCount*7).ToArray();
            if(squares.Length!=layerCount*7)throw new ArgumentException("Original vision layer RMS count differs.");
            var constants=source.constants.ToDictionary(value=>value.index);var added=new List<Constant>();
            var removed=new HashSet<Layer>();var insertions=new Dictionary<Layer,List<Layer>>();
            var next=source.inputs.Select(value=>value.index).Concat(source.constants.Select(value=>value.index)).Concat(source.layers.SelectMany(layer=>layer.outputs.Concat(layer.inputs))).Max()+1;
            var clone=typeof(object).GetMethod("MemberwiseClone",BindingFlags.Instance|BindingFlags.NonPublic);
            Layer Next(Layer current,string kind)=>source.layers.Single(layer=>layer.GetType().Name==kind&&layer.inputs.Contains(current.outputs[0]));
            for(var ordinal=0;ordinal<squares.Length;ordinal++)
            {
                var square=squares[ordinal];var mean=Next(square,"ReduceMean");var scalar=Next(mean,"ScalarMad");var inverse=Next(scalar,"Rsqrt");var multiply=Next(inverse,"Mul");
                Assert.That(multiply.inputs.Contains(square.inputs[0]),Is.True);
                Assert.That((float)scalar.GetType().GetField("sFloat").GetValue(scalar),Is.EqualTo(1f));
                Assert.That((float)scalar.GetType().GetField("bFloat").GetValue(scalar),Is.EqualTo(1e-6f));
                var scaled=source.layers.SingleOrDefault(layer=>layer.GetType().Name=="Mul"&&layer.inputs.Contains(multiply.outputs[0])
                    &&layer.inputs.Any(index=>constants.TryGetValue(index,out var constant)&&constant.dataType==DataType.Float&&constant.shape.rank==1));
                var scale=scaled==null?null:constants[scaled.inputs.Single(constants.ContainsKey)];
                var position=ordinal%7;var head=position>=1&&position<=3;var unscaled=position==3;var width=head?64:768;
                Assert.That(scale==null,Is.EqualTo(unscaled));if(scale!=null)Assert.That(scale.shape,Is.EqualTo(new TensorShape(width)));
                var final=scaled??multiply;var chain=new HashSet<Layer>{square,mean,scalar,inverse,multiply};if(scaled!=null)chain.Add(scaled);
                foreach(var layer in chain.Where(layer=>!ReferenceEquals(layer,final)))
                    Assert.That(source.layers.Where(consumer=>consumer.inputs.Intersect(layer.outputs).Any()).All(chain.Contains),Is.True,"An internal RMS value escapes its replacement.");
                removed.UnionWith(chain);
                var template=head?(unscaled?unscaledHeadTemplate:scaledHeadTemplate):hiddenTemplate;
                Assert.That(template.inputs.Count,Is.EqualTo(1));Assert.That(template.outputs.Count,Is.EqualTo(1));
                Assert.That(template.inputs[0].shape.ToTensorShape(),Is.EqualTo(head?new TensorShape(1,2520,12,64):new TensorShape(1,2520,768)));
                var map=new Dictionary<int,int>{{template.inputs[0].index,square.inputs[0]},{template.outputs[0].index,final.outputs[0]}};
                var templateScale=unscaled?null:template.constants.Single(value=>value.dataType==DataType.Float&&value.shape==new TensorShape(width));
                foreach(var constant in template.constants)
                {
                    if(ReferenceEquals(constant,templateScale)){map.Add(constant.index,scale.index);continue;}
                    var copy=(Constant)clone.Invoke(constant,null);copy.index=next++;map.Add(constant.index,copy.index);added.Add(copy);
                }
                foreach(var layer in template.layers)foreach(var output in layer.outputs)if(output>=0&&!map.ContainsKey(output))map.Add(output,next++);
                var layers=new List<Layer>();
                foreach(var layer in template.layers)
                {
                    var copy=(Layer)clone.Invoke(layer,null);copy.inputs=layer.inputs.Select(index=>index<0?index:map[index]).ToArray();copy.outputs=layer.outputs.Select(index=>index<0?index:map[index]).ToArray();layers.Add(copy);
                }
                insertions.Add(square,layers);
            }
            var result=new Model{ProducerName=source.ProducerName+"/inline-qualified-vision-rms",inputs=new List<Model.Input>(source.inputs),outputs=new List<Model.Output>(source.outputs),constants=source.constants.Concat(added).ToList()};
            foreach(var layer in source.layers)
                if(insertions.TryGetValue(layer,out var replacement))result.layers.AddRange(replacement);
                else if(!removed.Contains(layer))result.layers.Add(layer);
            return result;
        }
        public static Model BorrowScale(Model template,Constant originalScale)
        {
            if(template==null||originalScale==null||originalScale.dataType!=DataType.Float||originalScale.shape!=new TensorShape(768)||originalScale.lengthBytes!=768*4)
                throw new ArgumentException("Expected an original finite width-768 FP32 scale.");
            var field=typeof(Constant).GetField("array",BindingFlags.Instance|BindingFlags.NonPublic);
            var bytes=(ArraySegment<byte>)field.GetValue(originalScale);
            if(bytes.Array==null||bytes.Count!=768*4)throw new ArgumentException("Scale backing bytes differ.");
            for(var i=0;i<768;i++)
            {
                var value=BitConverter.ToSingle(bytes.Array,bytes.Offset+i*4);
                if(float.IsNaN(value)||float.IsInfinity(value))throw new ArgumentException("Scale must be finite.");
            }
            var target=template.constants.Single(value=>value.dataType==DataType.Float&&value.shape==new TensorShape(768));
            var replacement=(Constant)typeof(object).GetMethod("MemberwiseClone",BindingFlags.Instance|BindingFlags.NonPublic).Invoke(originalScale,null);
            replacement.index=target.index;
            return new Model{ProducerName=template.ProducerName+"/borrow-original-scale",inputs=new List<Model.Input>(template.inputs),outputs=new List<Model.Output>(template.outputs),
                layers=new List<Layer>(template.layers),constants=template.constants.Select(value=>ReferenceEquals(value,target)?replacement:value).ToList()};
        }
        public static (Model model,int[] outputs,Constant[] scales) MarkAll(Model source)
        {
            if(source==null)throw new ArgumentNullException(nameof(source));
            var constants=source.constants.ToDictionary(value=>value.index);
            var squares=source.layers.Where(layer=>layer.GetType().Name=="Square").ToArray();
            if(squares.Length!=7)throw new ArgumentException("Expected exactly the seven original vision layer RMS operations.",nameof(source));
            Layer Next(Layer current,string kind)=>source.layers.Single(layer=>layer.GetType().Name==kind&&layer.inputs.Contains(current.outputs[0]));
            var changes=new Dictionary<Layer,Layer>();var outputs=new List<int>();var scales=new List<Constant>();
            var identity=typeof(Worker).Assembly.GetTypes().Single(type=>type.Name=="Identity"&&typeof(Layer).IsAssignableFrom(type));
            foreach(var square in squares)
            {
                var mean=Next(square,"ReduceMean");var scalar=Next(mean,"ScalarMad");var inverse=Next(scalar,"Rsqrt");var multiply=Next(inverse,"Mul");
                Assert.That(multiply.inputs.Contains(square.inputs[0]),Is.True);
                Assert.That((float)scalar.GetType().GetField("sFloat").GetValue(scalar),Is.EqualTo(1f));
                Assert.That((float)scalar.GetType().GetField("bFloat").GetValue(scalar),Is.EqualTo(1e-6f));
                var scaled=source.layers.SingleOrDefault(layer=>layer.GetType().Name=="Mul"&&layer.inputs.Contains(multiply.outputs[0])
                    &&layer.inputs.Any(index=>constants.TryGetValue(index,out var constant)&&constant.dataType==DataType.Float&&constant.shape.rank==1));
                var scale=scaled==null?null:constants[scaled.inputs.Single(constants.ContainsKey)];
                if(scale!=null)Assert.That(scale.shape==new TensorShape(64)||scale.shape==new TensorShape(768),Is.True);
                var final=scaled??multiply;var chain=new HashSet<Layer>{square,mean,scalar,inverse,multiply};if(scaled!=null)chain.Add(scaled);
                foreach(var layer in chain.Where(layer=>!ReferenceEquals(layer,final)))
                    Assert.That(source.layers.Where(consumer=>consumer.inputs.Intersect(layer.outputs).Any()).All(chain.Contains),Is.True,"An internal RMS value escapes its cut.");
                var marker=(Layer)Activator.CreateInstance(identity);marker.inputs=new[]{square.inputs[0]};marker.outputs=(int[])final.outputs.Clone();
                changes.Add(final,marker);outputs.Add(marker.outputs[0]);scales.Add(scale);
            }
            return(new Model{ProducerName=source.ProducerName+"/qualified-seven-rms-markers",inputs=new List<Model.Input>(source.inputs),outputs=new List<Model.Output>(source.outputs),
                constants=new List<Constant>(source.constants),layers=source.layers.Select(layer=>changes.TryGetValue(layer,out var marker)?marker:layer).ToList()},outputs.ToArray(),scales.ToArray());
        }
        // Diagnostic graph surgery only: validate the exact six-node original RMS,
        // preserve every descriptor/coefficient, and expose its unchanged tensor ID.
        public static (Model model,int output) MarkFirst(Model source,int projectionOutput)
        {
            var constants=source.constants.ToDictionary(value=>value.index);
            var producer=source.layers.SelectMany(layer=>layer.outputs.Where(index=>index>=0).Select(index=>(index,layer))).ToDictionary(row=>row.index,row=>row.layer);
            var reshape=producer[producer[projectionOutput].inputs[0]];Assert.That(reshape.GetType().Name,Is.EqualTo("Reshape"));
            var final=producer[reshape.inputs[0]];Assert.That(final.GetType().Name,Is.EqualTo("Mul"));
            var scale=constants[final.inputs.Single(constants.ContainsKey)];
            Assert.That(scale.shape,Is.EqualTo(new TensorShape(768)));Assert.That(scale.dataType,Is.EqualTo(DataType.Float));
            var raw=(ArraySegment<byte>)typeof(Constant).GetField("array",BindingFlags.Instance|BindingFlags.NonPublic).GetValue(scale);
            using(var bytes=new MemoryStream(raw.Array,raw.Offset,raw.Count,false))
                Assert.That(PlayerFileHash.Compute(bytes,out _),Is.EqualTo("acc544061ccfe25009301bdfc15bab61468511ef6f27962b70c54efb65da9cb7"));
            var multiply=producer[final.inputs.Single(index=>!constants.ContainsKey(index))];Assert.That(multiply.GetType().Name,Is.EqualTo("Mul"));
            var inverse=producer[multiply.inputs.Single(index=>producer.TryGetValue(index,out var layer)&&layer.GetType().Name=="Rsqrt")];
            var scalar=producer[inverse.inputs[0]];Assert.That(scalar.GetType().Name,Is.EqualTo("ScalarMad"));
            var mean=producer[scalar.inputs[0]];Assert.That(mean.GetType().Name,Is.EqualTo("ReduceMean"));
            var square=producer[mean.inputs[0]];Assert.That(square.GetType().Name,Is.EqualTo("Square"));
            var input=square.inputs[0];Assert.That(multiply.inputs.Single(index=>index!=inverse.outputs[0]),Is.EqualTo(input));
            Assert.That((float)scalar.GetType().GetField("sFloat").GetValue(scalar),Is.EqualTo(1f));
            Assert.That((float)scalar.GetType().GetField("bFloat").GetValue(scalar),Is.EqualTo(1e-6f));
            var chain=new HashSet<Layer>{final,multiply,inverse,scalar,mean,square};Assert.That(chain.Count,Is.EqualTo(6));
            foreach(var layer in chain.Where(layer=>!ReferenceEquals(layer,final)))
                Assert.That(source.layers.Where(consumer=>consumer.inputs.Intersect(layer.outputs).Any()).All(chain.Contains),Is.True,"An internal RMS value escapes its cut.");
            var identity=typeof(Worker).Assembly.GetTypes().Single(type=>type.Name=="Identity"&&typeof(Layer).IsAssignableFrom(type));
            var marker=(Layer)Activator.CreateInstance(identity);marker.inputs=new[]{input};marker.outputs=(int[])final.outputs.Clone();
            return(new Model{ProducerName=source.ProducerName+"/qualified-first-rms-marker",inputs=new List<Model.Input>(source.inputs),outputs=new List<Model.Output>(source.outputs),
                constants=new List<Constant>(source.constants),layers=source.layers.Select(layer=>ReferenceEquals(layer,final)?marker:layer).ToList()},marker.outputs[0]);
        }
        public QualifiedVisionRmsOperation(Model model)
        {
            if(model==null)throw new ArgumentNullException(nameof(model));
            if(model.inputs.Count!=1||!model.inputs[0].shape.IsStatic())
                throw new ArgumentException("Expected a fixed qualified vision RMS template.",nameof(model));
            if(model.inputs[0].shape.ToTensorShape()==new TensorShape(1,2520,12,64))
            {
                pipeline=new GpuProjectionPipeline(new[]{model},Array.Empty<string>(),Array.Empty<string>(),Array.Empty<IGpuTensorOperation>());return;
            }
            if(model.inputs[0].shape.ToTensorShape()!=new TensorShape(1,2520,768))
                throw new ArgumentException("Expected the qualified fixed first vision RMS template.",nameof(model));
            var scalar=model.layers.Single(layer=>layer.GetType().Name=="ScalarMad"
                &&(float)layer.GetType().GetField("sFloat").GetValue(layer)==1f/768f
                &&(float)layer.GetType().GetField("bFloat").GetValue(layer)==1e-6f);
            var before=ModelTensorSlice.Extract(model,scalar.inputs[0],"total");
            var corrected=(Layer)typeof(object).GetMethod("MemberwiseClone",BindingFlags.Instance|BindingFlags.NonPublic).Invoke(scalar,null);
            corrected.GetType().GetField("sFloat").SetValue(corrected,1f);
            var changed=new Model{ProducerName=model.ProducerName+"/explicit-mean-division",inputs=new List<Model.Input>(model.inputs),
                outputs=new List<Model.Output>(model.outputs),constants=new List<Constant>(model.constants),
                layers=model.layers.Select(layer=>ReferenceEquals(layer,scalar)?corrected:layer).ToList()};
            var after=ModelTensorSlice.Extract(changed,changed.outputs[0].index,"normalized",new[]{new Model.Input{
                name="mean",index=scalar.inputs[0],dataType=DataType.Float,shape=new DynamicTensorShape(1,2520,1)}});
            try
            {
                divide=new GpuScalarDivide(768f);
                pipeline=new GpuProjectionPipeline(new[]{before,after},new[]{"total"},new[]{"mean"},new IGpuTensorOperation[]{divide});
            }
            catch{Dispose();throw;}
        }
        public Tensor<float> Execute(Tensor<float> input)
        {
            if(pipeline==null)throw new ObjectDisposedException(nameof(QualifiedVisionRmsOperation));
            return pipeline.Execute(input);
        }
        public void Dispose(){pipeline?.Dispose();pipeline=null;divide?.Dispose();divide=null;}
    }
}
