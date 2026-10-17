using System;
using System.Diagnostics;
using Newtonsoft.Json.Linq;
using Unity.InferenceEngine;
using UnityEngine.Profiling;

namespace EmbeddingGemma.Validation.Tests
{
    public static class GpuOperationTrace
    {
        /// <summary>Borrow the operation and tensors. Never download values or treat CPU submission as GPU completion.</summary>
        public static IGpuTensorOperation Wrap(IGpuTensorOperation operation, int index, Action<JObject> observe, bool gpuTiming = false)
        {
            if (operation == null) throw new ArgumentNullException(nameof(operation));
            if (observe == null) throw new ArgumentNullException(nameof(observe));
            if (index < 0) throw new ArgumentOutOfRangeException(nameof(index));
            return operation is IGpuBinaryTensorOperation binary
                ? new BinaryTrace(binary, index, observe, gpuTiming)
                : new UnaryTrace(operation, index, observe, gpuTiming);
        }
        class UnaryTrace : IGpuTensorOperation
        {
            readonly IGpuTensorOperation operation;
            readonly int index;
            readonly Action<JObject> observe;
            readonly CustomSampler sampler;
            public BackendType Backend => operation.Backend;
            public UnaryTrace(IGpuTensorOperation operation, int index, Action<JObject> observe, bool timing)
            {
                this.operation = operation; this.index = index; this.observe = observe;
                if (timing) sampler = CustomSampler.Create("EmbeddingGemma.Trace.Operation." + index, true);
            }
            JObject Row(string name, Tensor<float>[] inputs)
            {
                var shapes = new JArray(); var backends = new JArray();
                foreach (var input in inputs)
                {
                    shapes.Add(input == null ? null : input.shape.ToString());
                    backends.Add(input == null ? null : input.backendType.ToString());
                }
                return new JObject { ["utc"] = DateTime.UtcNow.ToString("O"), ["event"] = name,
                    ["operation_index"] = index, ["operation_type"] = operation.GetType().FullName,
                    ["input_shapes"] = shapes, ["input_backends"] = backends,
                    ["gpu_marker"] = sampler?.name, ["gpu_completion_verified"] = false };
            }
            protected Tensor<float> Invoke(Func<Tensor<float>> action, params Tensor<float>[] inputs)
            {
                observe(Row("begin", inputs));
                var watch = new Stopwatch(); Tensor<float> output;
                sampler?.Begin(); watch.Start();
                try { output = action(); }
                catch (Exception error)
                {
                    watch.Stop(); sampler?.End();
                    var failed = Row("exception", inputs); failed["error_type"] = error.GetType().FullName; observe(failed);
                    throw;
                }
                watch.Stop(); sampler?.End();
                var row = Row("submitted", inputs);
                row["cpu_wall_ns"] = (long)(watch.ElapsedTicks * (1000000000.0 / Stopwatch.Frequency));
                row["output_shape"] = output?.shape.ToString(); row["output_backend"] = output?.backendType.ToString();
                observe(row); return output;
            }
            public Tensor<float> Execute(Tensor<float> input) => Invoke(() => operation.Execute(input), input);
            // The session remains the owner of the underlying operation and tensors.
            public void Dispose() { }
        }
        sealed class BinaryTrace : UnaryTrace, IGpuBinaryTensorOperation
        {
            readonly IGpuBinaryTensorOperation operation;
            public BinaryTrace(IGpuBinaryTensorOperation operation, int index, Action<JObject> observe, bool timing)
                : base(operation, index, observe, timing) { this.operation = operation; }
            public Tensor<float> Execute(Tensor<float> left, Tensor<float> right)
                => Invoke(() => operation.Execute(left, right), left, right);
        }
    }
}
