using System;
using System.Linq;
using System.Reflection;
using System.Runtime.ExceptionServices;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using Unity.InferenceEngine;

namespace EmbeddingGemma.Tests
{
    public sealed class GpuConstantProbeTests
    {
        [TestCase(BackendType.CPU)] [TestCase(BackendType.GPUCompute)]
        public void ExecutableProbeChecksRealUploadAndAllOutputValues(BackendType backend)
        {
            var type=AppDomain.CurrentDomain.GetAssemblies().Select(assembly=>assembly.GetType("EmbeddingGemma.Validation.GpuConstantProbeChecks")).FirstOrDefault(value=>value!=null);
            Assert.That(type,Is.Not.Null,"IL2CPP executable probe is missing.");
            JObject result;
            try {result=(JObject)type.GetMethod("Evaluate").Invoke(null,new object[] {backend});}
            catch(TargetInvocationException error){ExceptionDispatchInfo.Capture(error.InnerException).Throw();throw;}
            Assert.That(result.Value<bool>("success"),Is.True);Assert.That(result.Value<string>("actual_backend"),Is.EqualTo(backend.ToString()));
            Assert.That(result.Value<int>("output_elements"),Is.EqualTo(2097153));Assert.That(result.Value<int>("mismatches"),Is.Zero);
            Assert.That(result.Value<int>("uploaded_constants"),Is.EqualTo(backend==BackendType.CPU?0:1));
            Assert.That(result.Value<int>("upload_calls"),Is.EqualTo(backend==BackendType.CPU?0:2));
            Assert.That(result.Value<int>("maximum_upload_bytes"),Is.EqualTo(backend==BackendType.CPU?0:8388608));
            Assert.That(result.Value<bool>("borrowed_model_reusable"),Is.True);
        }
    }
}
