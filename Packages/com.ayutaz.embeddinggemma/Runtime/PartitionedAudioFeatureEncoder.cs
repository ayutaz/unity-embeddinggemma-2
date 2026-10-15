using System;
using System.Threading;
using System.Threading.Tasks;
using Unity.InferenceEngine;
using UnityEngine;

namespace EmbeddingGemma
{
    /// <summary>
    /// Fixed-profile audio encoder backed by audited GPU segments. Owns its
    /// pipeline Workers and returned results; borrows models, operations and
    /// preprocessing tensors. Dispose the borrowed operations separately.
    /// The supplied segments must have been audited for exactly this profile.
    /// </summary>
    public sealed class PartitionedAudioFeatureEncoder : IAsyncAudioFeatureEncoder, IDisposable
    {
        GpuProjectionPipeline pipeline;
        public BackendType Backend => BackendType.GPUCompute;
        public AudioEncoderProfile Profile { get; }

        public PartitionedAudioFeatureEncoder(Model[] models,string[][] operationInputs,
            string[] operationOutputs,IGpuTensorOperation[] operations,AudioEncoderProfile profile)
        {
            if(profile==null)throw new ArgumentNullException(nameof(profile));
            if(!SystemInfo.supportsComputeShaders)throw new NotSupportedException("This device cannot run GPUCompute.");
            if(models==null || models.Length==0 || models[0]==null || models[0].inputs.Count!=1
                || models[0].inputs[0].name!="input" || models[0].inputs[0].dataType!=DataType.Float
                || !models[0].inputs[0].shape.IsStatic()
                || models[0].inputs[0].shape.ToTensorShape()!=new TensorShape(1,AudioEncoderProfile.FrameCount,AudioEncoderProfile.MelBinCount))
                throw new ArgumentException("Expected fixed float [1,1119,128] input named input.",nameof(models));
            Profile=profile;
            pipeline=new GpuProjectionPipeline(models,operationInputs,operationOutputs,operations,true);
        }

        Tensor<float> Validate(AudioPreprocessingResult input)
        {
            if(pipeline==null)throw new ObjectDisposedException(nameof(PartitionedAudioFeatureEncoder));
            if(input==null)throw new ArgumentNullException(nameof(input));
            var features=input.Features;var mask=input.FrameMask;
            if(features.dataOnBackend==null || mask.dataOnBackend==null)throw new ObjectDisposedException(nameof(input));
            if(input.ValidFrameCount!=Profile.ValidFrameCount
                || features.shape!=new TensorShape(1,AudioEncoderProfile.FrameCount,AudioEncoderProfile.MelBinCount)
                || mask.shape!=new TensorShape(1,AudioEncoderProfile.FrameCount)
                || features.backendType!=Backend || mask.backendType!=Backend)
                throw new ArgumentException("Preprocessing prefix, tensor shapes and devices must match the fixed GPU encoder profile.",nameof(input));
            return features;
        }

        AudioEncodingResult TakeResult(Tensor<float> output)
        {
            if(output.shape!=new TensorShape(Profile.FeatureCount,AudioEncoderProfile.FeatureWidth) || output.backendType!=Backend)
            {
                output.Dispose();throw new InvalidOperationException("Expected projected float soft tokens on GPUCompute.");
            }
            return new AudioEncodingResult(output,Profile.ValidFrameCount);
        }

        public AudioEncodingResult Encode(AudioPreprocessingResult input)
        {
            var features=Validate(input);return TakeResult(pipeline.Execute(features));
        }

        /// <summary>Retires completed GPU segment caches at fences without intermediate readback.</summary>
        public async Task<AudioEncodingResult> EncodeAsync(AudioPreprocessingResult input,CancellationToken cancellationToken=default)
        {
            var features=Validate(input);return TakeResult(await pipeline.ExecuteAsync(features,cancellationToken));
        }

        public void Dispose(){pipeline?.Dispose();pipeline=null;}
    }
}
