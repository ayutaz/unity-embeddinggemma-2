using System;
using System.Threading;
using System.Threading.Tasks;
using Unity.InferenceEngine;
using UnityEngine;

namespace EmbeddingGemma
{
    /// <summary>
    /// Fixed-profile image encoder using qualified GPU segments. Owns Workers and
    /// returned results; borrows models, operations and preprocessing tensors.
    /// Dispose operations separately after pending calls complete. Large graphs
    /// should use EncodeAsync to retire work at GPU fences without tensor readback.
    /// </summary>
    public sealed class PartitionedImageFeatureEncoder : IAsyncImageFeatureEncoder, IDisposable
    {
        GpuProjectionPipeline pipeline;
        public BackendType Backend => BackendType.GPUCompute;
        public ImageEncoderProfile Profile { get; }

        public PartitionedImageFeatureEncoder(Model[] models,string[][] operationInputs,
            string[] operationOutputs,IGpuTensorOperation[] operations,ImageEncoderProfile profile)
        {
            if(profile==null)throw new ArgumentNullException(nameof(profile));
            if(!SystemInfo.supportsComputeShaders)throw new NotSupportedException("This device cannot run GPUCompute.");
            if(models==null||models.Length==0||models[0]==null||models[0].inputs.Count!=1
                ||models[0].inputs[0].name!="pixels"||models[0].inputs[0].dataType!=DataType.Float
                ||!models[0].inputs[0].shape.IsStatic()||models[0].inputs[0].shape.ToTensorShape()!=new TensorShape(1,2520,768))
                throw new ArgumentException("Expected fixed float [1,2520,768] input named pixels.",nameof(models));
            Profile=profile;
            pipeline=new GpuProjectionPipeline(models,operationInputs,operationOutputs,operations,true);
        }

        Tensor<float> Validate(ImagePreprocessingResult input)
        {
            if(pipeline==null)throw new ObjectDisposedException(nameof(PartitionedImageFeatureEncoder));
            if(input==null)throw new ArgumentNullException(nameof(input));
            var pixels=input.Pixels;
            if(pixels.dataOnBackend==null)throw new ObjectDisposedException(nameof(input));
            if(input.ResizedWidth!=Profile.ResizedWidth||input.ResizedHeight!=Profile.ResizedHeight
                ||input.SoftTokenCount!=Profile.FeatureCount||pixels.shape!=new TensorShape(1,2520,768)||pixels.backendType!=Backend)
                throw new ArgumentException("Preprocessing geometry, feature count, shape and device must match the fixed GPU image profile.",nameof(input));
            Profile.ValidatePositions(input.Positions);return pixels;
        }

        ImageEncodingResult TakeResult(Tensor<float> output)
        {
            if(output.shape!=new TensorShape(Profile.FeatureCount,ImageEncoderProfile.FeatureWidth)||output.backendType!=Backend)
            {
                output.Dispose();throw new InvalidOperationException("Expected projected float soft tokens on GPUCompute.");
            }
            return new ImageEncodingResult(output,Profile.ResizedWidth,Profile.ResizedHeight);
        }

        public ImageEncodingResult Encode(ImagePreprocessingResult input)
        {
            var pixels=Validate(input);return TakeResult(pipeline.Execute(pixels));
        }

        public async Task<ImageEncodingResult> EncodeAsync(ImagePreprocessingResult input,CancellationToken cancellationToken=default)
        {
            var pixels=Validate(input);return TakeResult(await pipeline.ExecuteAsync(pixels,cancellationToken));
        }

        public void Dispose(){pipeline?.Dispose();pipeline=null;}
    }
}
