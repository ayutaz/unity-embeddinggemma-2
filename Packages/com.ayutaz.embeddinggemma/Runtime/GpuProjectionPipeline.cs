using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Unity.InferenceEngine;

namespace EmbeddingGemma
{
    /// <summary>
    /// Experimental main-thread composition of fixed GPU segments and borrowed
    /// GPU projections. Owns Workers and per-call copies; never reads intermediate data.
    /// </summary>
    public sealed class GpuProjectionPipeline : IDisposable
    {
        readonly Model[] models;
        readonly string[][] projectionInputs;
        readonly string[] projectionOutputs;
        readonly IGpuTensorOperation[] projections;
        readonly HashSet<string>[] liveAfterStage;
        readonly Action[] releaseGpuCache;
        Worker[] workers;

        public GpuProjectionPipeline(Model[] models,string[] projectionInputs,string[] projectionOutputs,GpuBlockedMatmul[] projections)
            : this(models,projectionInputs,projectionOutputs,(IGpuTensorOperation[])projections) { }

        public GpuProjectionPipeline(Model[] models,string[] projectionInputs,string[] projectionOutputs,IGpuTensorOperation[] projections)
            : this(models,projectionInputs?.Select(name=>new[]{name}).ToArray(),projectionOutputs,projections) { }

        public GpuProjectionPipeline(Model[] models,string[][] projectionInputs,string[] projectionOutputs,IGpuTensorOperation[] projections)
            : this(models,projectionInputs,projectionOutputs,projections,false) { }

        public GpuProjectionPipeline(Model[] models,string[][] projectionInputs,string[] projectionOutputs,IGpuTensorOperation[] projections,bool releaseUnusedGpuCache)
        {
            if(models==null || projections==null || projectionInputs==null || projectionOutputs==null
                || models.Length!=projections.Length+1 || projectionInputs.Length!=projections.Length
                || projectionOutputs.Length!=projections.Length || models.Any(m=>m==null)
                || projections.Any(p=>p==null || p.Backend!=BackendType.GPUCompute) || projectionOutputs.Any(string.IsNullOrEmpty))
                throw new ArgumentException("Expected ordered segments and matching GPU projection boundaries.");
            for(var index=0;index<projections.Length;index++)
                if(projectionInputs[index]==null || projectionInputs[index].Length!=(projections[index] is IGpuBinaryTensorOperation?2:1) || projectionInputs[index].Any(string.IsNullOrEmpty))
                    throw new ArgumentException("GPU operation operand names differ from its arity.");
            if(models[0].inputs.Count!=1 || models[0].inputs[0].dataType!=DataType.Float
                || !models[0].inputs[0].shape.IsStatic() || models[models.Length-1].outputs.Count!=1)
                throw new ArgumentException("Expected one fixed float input and one final output.");
            this.models=(Model[])models.Clone();this.projectionInputs=projectionInputs.Select(names=>(string[])names.Clone()).ToArray();
            this.projectionOutputs=(string[])projectionOutputs.Clone();this.projections=(IGpuTensorOperation[])projections.Clone();
            // Compute once: residual branches can consume a value many stages later.
            // Keep the final output live until it is transferred to the caller.
            liveAfterStage=new HashSet<string>[models.Length];
            var live=new HashSet<string>{models[models.Length-1].outputs[0].name};
            for(var stage=models.Length-1;stage>=0;stage--)
            {
                liveAfterStage[stage]=new HashSet<string>(live);
                live.UnionWith(models[stage].inputs.Select(boundary=>boundary.name));
                if(stage<projections.Length)live.UnionWith(this.projectionInputs[stage]);
            }
            workers=new Worker[models.Length];
            if(releaseUnusedGpuCache)releaseGpuCache=new Action[models.Length];
            try
            {
                for(var i=0;i<models.Length;i++)
                {
                    workers[i]=new Worker(models[i],BackendType.GPUCompute);
                    var cpu=(HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallback",BindingFlags.NonPublic|BindingFlags.Instance)?.GetValue(workers[i]);
                    var flush=(HashSet<int>)typeof(Worker).GetField("m_LayerCPUFallbackShouldFlushGPU",BindingFlags.NonPublic|BindingFlags.Instance)?.GetValue(workers[i]);
                    if(cpu==null || flush==null)throw new NotSupportedException("Pinned GPU fallback audit is unavailable.");
                    var data= models[i].layers.SelectMany(l=>l.outputs).Concat(models[i].inputs.Select(input=>input.index));
                    if(cpu.Intersect(data).Any() || flush.Count!=0)
                        throw new NotSupportedException("GPU composition cannot use CPU computation or intermediate readback.");
                    if(releaseGpuCache!=null)releaseGpuCache[i]=CacheRelease(workers[i],models[i]);
                }
            }
            catch { Dispose();throw; }
        }

        static Action CacheRelease(Worker worker,Model model)
        {
            const BindingFlags fields=BindingFlags.Instance|BindingFlags.NonPublic;
            var storage=typeof(Worker).GetField("m_Storage",fields)?.GetValue(worker);
            var pool=storage?.GetType().GetField("m_TensorPool",fields)?.GetValue(storage);
            var cacheField=pool?.GetType().GetField("m_computeMemoryPool",fields);
            var freeField=cacheField?.FieldType.GetField("freeBuffers",fields);
            var valuesField=storage?.GetType().GetField("m_InUseTensorsPool",fields);
            if(cacheField==null||freeField==null||valuesField==null||!typeof(IDisposable).IsAssignableFrom(cacheField.FieldType))
                throw new NotSupportedException("Pinned unused GPU cache inspection is unavailable.");
            var protectedIndices=new HashSet<int>(model.inputs.Select(value=>value.index).Concat(model.constants.Select(value=>value.index)));
            var copiedOutputs=model.outputs.Select(value=>value.index).Where(index=>!protectedIndices.Contains(index)).Distinct().ToArray();
            return ()=>
            {
                if(!(valuesField.GetValue(storage) is IDictionary<int,Tensor> values))throw new NotSupportedException("Pinned live tensor storage differs.");
                var protectedTensors=new HashSet<Tensor>(protectedIndices.Where(values.ContainsKey).Select(index=>values[index]).Where(value=>value!=null));
                var retired=new HashSet<Tensor>();
                foreach(var index in copiedOutputs)
                {
                    if(!values.TryGetValue(index,out var value))continue;
                    values.Remove(index);
                    if(value!=null&&!protectedTensors.Contains(value)&&retired.Add(value))value.Dispose();
                }
                var cache=cacheField.GetValue(pool);
                if(!(freeField.GetValue(cache) is System.Collections.IDictionary free))throw new NotSupportedException("Pinned free GPU cache differs.");
                if(free.Count==0)return;
                var replacement=Activator.CreateInstance(cacheField.FieldType,true);
                try{((IDisposable)cache).Dispose();cacheField.SetValue(pool,replacement);}
                catch{((IDisposable)replacement).Dispose();throw;}
            };
        }

        public Tensor<float> Execute(Tensor<float> input)
        {
            if(workers==null)throw new ObjectDisposedException(nameof(GpuProjectionPipeline));
            if(input==null)throw new ArgumentNullException(nameof(input));
            if(input.dataOnBackend==null)throw new ObjectDisposedException(nameof(input));
            if(input.backendType!=BackendType.GPUCompute || input.shape!=models[0].inputs[0].shape.ToTensorShape())
                throw new ArgumentException("Input must match the fixed shape and already be on GPUCompute.");
            var values=new Dictionary<string,Tensor<float>>{{models[0].inputs[0].name,input}};
            var owned=new HashSet<Tensor<float>>();
            try
            {
                for(var stage=0;stage<workers.Length;stage++)
                {
                    foreach(var boundary in models[stage].inputs)
                    {
                        if(!values.TryGetValue(boundary.name,out var value))throw new InvalidOperationException("GPU boundary input is missing: "+boundary.name);
                        workers[stage].SetInput(boundary.name,value);
                    }
                    workers[stage].Schedule();
                    foreach(var boundary in models[stage].outputs)
                    {
                        Tensor copy=null;
                        workers[stage].CopyOutput(boundary.name,ref copy);
                        var value=copy as Tensor<float>;
                        if(value==null){copy?.Dispose();throw new InvalidOperationException("Expected a float GPU boundary.");}
                        owned.Add(value);
                        if(value.backendType!=BackendType.GPUCompute)throw new InvalidOperationException("Intermediate output left GPUCompute.");
                        values[boundary.name]=value;
                    }
                    releaseGpuCache?[stage]?.Invoke();
                    if(stage<projections.Length)
                    {
                        var operands=new Tensor<float>[projectionInputs[stage].Length];
                        for(var index=0;index<operands.Length;index++)
                            if(!values.TryGetValue(projectionInputs[stage][index],out operands[index]))throw new InvalidOperationException("GPU operation input is missing.");
                        var projected=projections[stage] is IGpuBinaryTensorOperation binary?binary.Execute(operands[0],operands[1]):projections[stage].Execute(operands[0]);
                        if(projected==null || operands.Any(value=>ReferenceEquals(projected,value)))throw new InvalidOperationException("GPU operation must return an independent owned tensor.");
                        owned.Add(projected);
                        if(projected.backendType!=BackendType.GPUCompute)throw new InvalidOperationException("GPU operation output left GPUCompute.");
                        values[projectionOutputs[stage]]=projected;
                    }
                    foreach(var name in values.Keys.Where(name=>!liveAfterStage[stage].Contains(name)).ToArray())values.Remove(name);
                    var retained=new HashSet<Tensor<float>>(values.Values);
                    foreach(var value in owned.Where(value=>!retained.Contains(value)).ToArray())
                    {
                        value.Dispose();owned.Remove(value);
                    }
                }
                var output=values[models[models.Length-1].outputs[0].name];owned.Remove(output);return output;
            }
            finally { foreach(var value in owned)value.Dispose(); }
        }

        public void Dispose()
        {
            if(workers==null)return;
            foreach(var worker in workers)worker?.Dispose();workers=null;
        }
    }
}
