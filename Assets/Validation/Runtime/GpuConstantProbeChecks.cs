using System;
using System.Linq;
using Newtonsoft.Json.Linq;
using Unity.InferenceEngine;

namespace EmbeddingGemma.Validation
{
    /// <summary>Weights-free executable check of GPU storage inspection under actual Player stripping.</summary>
    public static class GpuConstantProbeChecks
    {
        public static JObject Evaluate(BackendType backend)
        {
            const int count=2097153;
            var values=Enumerable.Range(0,count).Select(index=>0.25f+((uint)index*1664525u+1013904223u)%65536/65536f).ToArray();
            var graph=new FunctionalGraph();var input=graph.AddInput<float>(new TensorShape(1,count),"input");
            var model=graph.Compile(input+Functional.Constant(new TensorShape(1,count),values));
            if(!model.constants.Any(value=>value.lengthBytes>GpuConstantWorker.DefaultChunkBytes))
                throw new InvalidOperationException("Probe must include a default-budget crossing constant.");
            var result=new JObject {["success"]=false,["requested_backend"]=backend.ToString(),["output_elements"]=count};
            using(var holder=new GpuConstantWorker(model,backend))
            using(var tensor=new Tensor<float>(new TensorShape(1,count),new float[count]))
            {
                var worker=holder.Worker;worker.Schedule(tensor);var output=worker.PeekOutput() as Tensor<float>;
                if(output==null||output.backendType!=backend)throw new InvalidOperationException("Unexpected probe output backend.");
                var actual=output.DownloadToArray();var mismatches=0;
                for(var index=0;index<count;index++)
                    if(BitConverter.SingleToInt32Bits(actual[index])!=BitConverter.SingleToInt32Bits(values[index]))mismatches++;
                result["actual_backend"]=output.backendType.ToString();result["mismatches"]=mismatches;
                result["uploaded_constants"]=holder.UploadedConstantCount;result["upload_calls"]=holder.UploadCallCount;
                result["maximum_upload_bytes"]=holder.MaximumUploadBytes;
                if(mismatches!=0)throw new InvalidOperationException("Probe output changed constant bits.");
            }
            // The owner is disposed; the borrowed Model must still be usable by an independent Worker.
            using(var baseline=new Worker(model,BackendType.CPU))
            using(var tensor=new Tensor<float>(new TensorShape(1,count),new float[count]))
            {
                baseline.Schedule(tensor);var actual=((Tensor<float>)baseline.PeekOutput()).DownloadToArray();
                result["borrowed_model_reusable"]=actual.Length==count&&actual[0]==values[0]&&actual[count-1]==values[count-1];
            }
            if(result.Value<bool>("borrowed_model_reusable")!=true)throw new InvalidOperationException("Borrowed Model was invalidated.");
            result["success"]=true;return result;
        }
    }
}
