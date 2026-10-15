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
