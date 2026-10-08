using System;
using System.Linq;
using Unity.InferenceEngine;
using UnityEngine;

namespace EmbeddingGemma
{
    /// <summary>
    /// Synchronous, main-thread text inference for the pinned batch-1, length-128 fp32 export.
    /// Owns its Worker and per-call input tensors; the caller owns the supplied Model.
    /// The exported graph includes projection, mean pooling and L2 normalization.
    /// </summary>
    public sealed class TextEmbedder : IDisposable
    {
        public const int SequenceLength = 128;
        public const int EmbeddingDimension = 768;
        readonly TextTokenizer tokenizer;
        Worker worker;
        public BackendType Backend { get; }

        public TextEmbedder(Model model, string tokenizerJson, BackendType backend)
        {
            if (model == null) throw new ArgumentNullException(nameof(model));
            if (tokenizerJson == null) throw new ArgumentNullException(nameof(tokenizerJson));
            if (backend != BackendType.CPU && backend != BackendType.GPUCompute)
                throw new ArgumentOutOfRangeException(nameof(backend), "Use CPU or GPUCompute explicitly.");
            if (backend == BackendType.GPUCompute && !SystemInfo.supportsComputeShaders)
                throw new NotSupportedException("This device cannot run GPUCompute.");
            var names = new[] { "input_ids", "attention_mask" };
            if (model.inputs.Count != 2 || model.outputs.Count != 1 ||
                names.Any(name => !model.inputs.Any(input => input.name == name && input.dataType == DataType.Int &&
                    input.shape.IsStatic() && input.shape.ToTensorShape() == new TensorShape(1, SequenceLength))))
                throw new ArgumentException("Expected two int32 [1,128] inputs and one embedding output.", nameof(model));
            tokenizer = new TextTokenizer(tokenizerJson);
            worker = new Worker(model, backend);
            Backend = backend;
            if (worker.backendType != backend)
            {
                Dispose();
                throw new NotSupportedException("The requested backend was not created.");
            }
        }

        public float[] EmbedQuery(string text) => Embed(text, TextRole.Query);
        public float[] EmbedDocument(string text, string title = null) => Embed(text, TextRole.Document, title);
        public float[] EmbedRaw(string text) => Embed(text, TextRole.Raw);

        public float[] Embed(string text, TextRole role, string title = null)
        {
            if (worker == null) throw new ObjectDisposedException(nameof(TextEmbedder));
            var encoded = tokenizer.Encode(TextPrompts.Format(text, role, title));
            var ids = encoded.GetIds().ToArray();
            var mask = encoded.GetAttentionMask().ToArray();
            if (ids.Length != SequenceLength || mask.Length != SequenceLength)
                throw new InvalidOperationException("Tokenizer must pad/truncate every input to 128 tokens.");
            using var inputIds = new Tensor<int>(new TensorShape(1, SequenceLength), ids);
            using var attentionMask = new Tensor<int>(new TensorShape(1, SequenceLength), mask);
            worker.SetInput("input_ids", inputIds);
            worker.SetInput("attention_mask", attentionMask);
            worker.Schedule();
            var output = worker.PeekOutput() as Tensor<float>;
            if (output == null || output.shape != new TensorShape(1, EmbeddingDimension))
                throw new InvalidOperationException("Expected a float32 [1,768] embedding.");
            // Copy only the final vector to CPU; the Worker continues to own the output tensor.
            var result = output.DownloadToArray();
            if (result.Any(value => float.IsNaN(value) || float.IsInfinity(value)))
                throw new InvalidOperationException("Embedding contains non-finite values.");
            var normSquared = result.Sum(value => (double)value * value);
            if (Math.Abs(normSquared - 1.0) > 0.002)
                throw new InvalidOperationException("Exported embedding is not L2 normalized.");
            return result;
        }

        public void Dispose()
        {
            worker?.Dispose();
            worker = null;
        }
    }
}
