using System;
using System.IO;
using System.Linq;
using System.Reflection;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEditor;
using UnityEngine;

namespace EmbeddingGemma.Validation.Tests
{
    [InitializeOnLoad]
    static class OptionalNativeAllocationTracing
    {
        static OptionalNativeAllocationTracing()
        {
            if(Environment.GetEnvironmentVariable("EG_TRACE_NATIVE_ALLOCATIONS")!="1")return;
            var mode=NativeAllocationDiagnosticsTests.ModeProperty();
            mode.SetValue(null,Enum.Parse(mode.PropertyType,"EnabledWithStackTrace"));
            Debug.Log("EG_NATIVE_ALLOCATION_TRACE_MODE="+mode.GetValue(null));
        }
    }

    // Diagnostic-only: compare an Editor/Test Runner baseline against small
    // production GPU ownership contracts. Neither proves real-model numerics.
    public sealed class NativeAllocationDiagnosticsTests
    {
        public static PropertyInfo ModeProperty()
        {
            var type=AppDomain.CurrentDomain.GetAssemblies().Select(assembly=>assembly.GetType("Unity.Collections.NativeLeakDetection")).FirstOrDefault(value=>value!=null);
            Assert.That(type,Is.Not.Null,"Pinned native leak detection API must be available.");
            var mode=type.GetProperty("Mode",BindingFlags.Public|BindingFlags.Static);Assert.That(mode,Is.Not.Null);return mode;
        }
        static void Run(bool gpu)
        {
            Assert.That(Environment.GetEnvironmentVariable("EG_TRACE_NATIVE_ALLOCATIONS"),Is.EqualTo("1"));
            Assert.That(ModeProperty().GetValue(null).ToString(),Is.EqualTo("EnabledWithStackTrace"),"The diagnostic must enable allocation traces before any Worker creation.");
            if(gpu)
            {
                var contracts=new EmbeddingGemma.Tests.PartitionedAudioFeatureEncoderTests();
                contracts.GpuAdapterOwnsWorkersAndResultsWhileBorrowingInputsAndOperations();
                contracts.WrongPrefixDeviceNullAndDisposedInputAreRejectedWithoutPoisoningReuse();
                contracts.WrongScheduledOutputShapeIsRejectedAndCanBeRetried();
                new EmbeddingGemma.Tests.AudioEmbedderTests().PipelineReturnsOwnedNormalized768VectorAndReleasesIntermediateCopies(BackendType.GPUCompute);
            }
            var source=Environment.GetEnvironmentVariable("EG_ACCEPTANCE_SOURCE");
            var path=Path.GetFullPath(Path.Combine(Application.dataPath,"../artifacts/native-allocation-trace-"+source+(gpu?"-tiny-gpu":"-baseline")+".json"));
            Assert.That(File.Exists(path),Is.False);
            File.WriteAllText(path,new JObject{["source"]=source,["success"]=true,["scope"]=gpu?"tiny_gpu_ownership_contracts_with_allocation_tracing":"editor_test_runner_baseline_no_inference",
                ["allocation_trace_mode"]=ModeProperty().GetValue(null).ToString(),["production_contracts_exercised"]=gpu?4:0,
                ["actual_model_executed"]=false,["leak_free_shutdown_proven"]=false}.ToString());
        }
        [Test] public void NoInferenceNativeAllocationBaseline()=>Run(false);
        [Test] public void TinyGpuOwnershipWithNativeAllocationTrace()=>Run(true);
    }
}
