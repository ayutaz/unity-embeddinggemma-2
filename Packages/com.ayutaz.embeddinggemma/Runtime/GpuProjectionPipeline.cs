using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.Rendering;

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
        readonly Action[] releaseCompletedBackendCache;
        readonly int ownerThread=Thread.CurrentThread.ManagedThreadId;
        Worker[] workers;
        bool executing,disposeRequested;

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
            if(releaseUnusedGpuCache){releaseGpuCache=new Action[models.Length];releaseCompletedBackendCache=new Action[models.Length];}
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
                    if(releaseGpuCache!=null)
                    {
                        releaseGpuCache[i]=CacheRelease(workers[i],models[i]);releaseCompletedBackendCache[i]=CompletedBackendCacheRelease(workers[i]);
                    }
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
            var backend=typeof(Worker).GetField("m_Backend",fields)?.GetValue(worker);
            var backendType=typeof(Worker).Assembly.GetType("Unity.InferenceEngine.GPUComputeBackend");
            var scratchField=backendType?.GetField("m_MemoryPool",fields);
            var freeField=cacheField?.FieldType.GetField("freeBuffers",fields);
            var valuesField=storage?.GetType().GetField("m_InUseTensorsPool",fields);
            if(cacheField==null||freeField==null||valuesField==null||scratchField==null||!backendType.IsInstanceOfType(backend)||scratchField.FieldType!=cacheField.FieldType||!typeof(IDisposable).IsAssignableFrom(cacheField.FieldType))
                throw new NotSupportedException("Pinned unused GPU cache inspection is unavailable.");
            void ClearPool(object container,FieldInfo field)
            {
                var cache=field.GetValue(container);
                if(!(freeField.GetValue(cache) is System.Collections.IDictionary free))throw new NotSupportedException("Pinned free GPU cache differs.");
                if(free.Count==0)return;
                var replacement=Activator.CreateInstance(field.FieldType,true);
                try{((IDisposable)cache).Dispose();field.SetValue(container,replacement);}
                catch{((IDisposable)replacement).Dispose();throw;}
            }
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
                ClearPool(pool,cacheField);ClearPool(backend,scratchField);
            };
        }

        static Action CompletedBackendCacheRelease(Worker worker)
        {
            const BindingFlags fields=BindingFlags.Instance|BindingFlags.NonPublic;
            var backend=typeof(Worker).GetField("m_Backend",fields)?.GetValue(worker);
            var type=typeof(Worker).Assembly.GetType("Unity.InferenceEngine.GfxDeviceBackend");
            if(type==null||!type.IsInstanceOfType(backend))return null;
            var context=type.GetField("m_GfxContext",fields);
            if(context==null||!typeof(IDisposable).IsAssignableFrom(context.FieldType))throw new NotSupportedException("Pinned compiled GPU operator context is unavailable.");
            return ()=>
            {
                var previous=(IDisposable)context.GetValue(backend);var replacement=(IDisposable)Activator.CreateInstance(context.FieldType);
                try{previous.Dispose();context.SetValue(backend,replacement);}
                catch{replacement.Dispose();throw;}
            };
        }

        public Tensor<float> Execute(Tensor<float> input)
        {
            using var call=BeginCall(input);
            for(var stage=0;stage<models.Length;stage++)call.RunStage(stage);
            return call.TakeResult();
        }

        /// <summary>
        /// Executes on Unity's main thread, yielding between GPU-complete stages to
        /// bound work in flight. Uses graphics fences, never intermediate tensor readback.
        /// With cache release enabled, also retires completed ML-device compiled operators.
        /// Borrows input and operations; transfers the final tensor only on success.
        /// Cancellation or Dispose aborts at a stage boundary and releases per-call tensors.
        /// Await completion before reusing the pipeline or disposing borrowed resources.
        /// </summary>
        public async Task<Tensor<float>> ExecuteAsync(Tensor<float> input,CancellationToken cancellationToken=default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if(Thread.CurrentThread.ManagedThreadId!=ownerThread||SynchronizationContext.Current==null)
                throw new InvalidOperationException("GPU asynchronous execution requires Unity's main thread and synchronization context.");
            if(!SystemInfo.supportsGraphicsFence)throw new NotSupportedException("Staged GPU execution requires graphics fences.");
            using var call=BeginCall(input);
            for(var stage=0;stage<models.Length;stage++)
            {
                cancellationToken.ThrowIfCancellationRequested();call.RunStage(stage);
                var fence=Graphics.CreateGraphicsFence(GraphicsFenceType.AsyncQueueSynchronisation,SynchronisationStageFlags.AllGPUOperations);
                do
                {
                    await Task.Yield();
                }while(!fence.passed);
                releaseCompletedBackendCache?[stage]?.Invoke();
                cancellationToken.ThrowIfCancellationRequested();ThrowIfDisposed();
            }
            return call.TakeResult();
        }

        void ThrowIfDisposed()
        {
            if(workers==null||disposeRequested)throw new ObjectDisposedException(nameof(GpuProjectionPipeline));
        }
        Call BeginCall(Tensor<float> input)
        {
            ThrowIfDisposed();
            if(executing)throw new InvalidOperationException("Only one GPU pipeline execution may be active.");
            if(Thread.CurrentThread.ManagedThreadId!=ownerThread)throw new InvalidOperationException("GPU execution requires the creating main thread.");
            if(input==null)throw new ArgumentNullException(nameof(input));
            if(input.dataOnBackend==null)throw new ObjectDisposedException(nameof(input));
            if(input.backendType!=BackendType.GPUCompute || input.shape!=models[0].inputs[0].shape.ToTensorShape())
                throw new ArgumentException("Input must match the fixed shape and already be on GPUCompute.");
            var call=new Call(this,input);executing=true;return call;
        }
        void EndCall()
        {
            executing=false;if(disposeRequested)DisposeWorkers();
        }
        sealed class Call : IDisposable
        {
            readonly GpuProjectionPipeline owner;
            readonly Dictionary<string,Tensor<float>> values;
            readonly HashSet<Tensor<float>> owned=new HashSet<Tensor<float>>();
            bool disposed;
            public Call(GpuProjectionPipeline owner,Tensor<float> input)
            {
                this.owner=owner;values=new Dictionary<string,Tensor<float>>{{owner.models[0].inputs[0].name,input}};
            }
            public void RunStage(int stage)
            {
                owner.ThrowIfDisposed();
                var model=owner.models[stage];var worker=owner.workers[stage];
                foreach(var boundary in model.inputs)
                {
                    if(!values.TryGetValue(boundary.name,out var value))throw new InvalidOperationException("GPU boundary input is missing: "+boundary.name);
                    worker.SetInput(boundary.name,value);
                }
                worker.Schedule();
                foreach(var boundary in model.outputs)
                {
                    Tensor copy=null;worker.CopyOutput(boundary.name,ref copy);
                    var value=copy as Tensor<float>;
                    if(value==null){copy?.Dispose();throw new InvalidOperationException("Expected a float GPU boundary.");}
                    owned.Add(value);
                    if(value.backendType!=BackendType.GPUCompute)throw new InvalidOperationException("Intermediate output left GPUCompute.");
                    values[boundary.name]=value;
                }
                owner.releaseGpuCache?[stage]?.Invoke();
                if(stage<owner.projections.Length)
                {
                    var operands=new Tensor<float>[owner.projectionInputs[stage].Length];
                    for(var index=0;index<operands.Length;index++)
                        if(!values.TryGetValue(owner.projectionInputs[stage][index],out operands[index]))throw new InvalidOperationException("GPU operation input is missing.");
                    var operation=owner.projections[stage];
                    var projected=operation is IGpuBinaryTensorOperation binary?binary.Execute(operands[0],operands[1]):operation.Execute(operands[0]);
                    if(projected==null || operands.Any(value=>ReferenceEquals(projected,value)))throw new InvalidOperationException("GPU operation must return an independent owned tensor.");
                    owned.Add(projected);
                    if(projected.backendType!=BackendType.GPUCompute)throw new InvalidOperationException("GPU operation output left GPUCompute.");
                    values[owner.projectionOutputs[stage]]=projected;
                }
                foreach(var name in values.Keys.Where(name=>!owner.liveAfterStage[stage].Contains(name)).ToArray())values.Remove(name);
                var retained=new HashSet<Tensor<float>>(values.Values);
                foreach(var value in owned.Where(value=>!retained.Contains(value)).ToArray())
                {
                    value.Dispose();owned.Remove(value);
                }
            }
            public Tensor<float> TakeResult()
            {
                owner.ThrowIfDisposed();var output=values[owner.models[owner.models.Length-1].outputs[0].name];owned.Remove(output);return output;
            }
            public void Dispose()
            {
                if(disposed)return;disposed=true;
                try{foreach(var value in owned)value.Dispose();}
                finally{owned.Clear();owner.EndCall();}
            }
        }

        public void Dispose()
        {
            if(workers==null)return;disposeRequested=true;
            if(!executing)DisposeWorkers();
        }
        void DisposeWorkers()
        {
            foreach(var worker in workers)worker?.Dispose();workers=null;
        }
    }
}
