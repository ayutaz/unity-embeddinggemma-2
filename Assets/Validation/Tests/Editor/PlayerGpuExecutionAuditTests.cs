using System;
using System.Reflection;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using Unity.InferenceEngine;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class PlayerGpuExecutionAuditTests
    {
        static JObject EvaluateModel(Model model, string backend = "GPUCompute", int[] cpu = null, int[] flush = null)
        {
            var method = Required().GetMethod("EvaluateModel");
            Assert.That(method, Is.Not.Null, "Audit graph routing before distinguishing an empty alias from missing computation.");
            return (JObject)method.Invoke(null, new object[] { model, backend, cpu ?? Array.Empty<int>(), flush ?? Array.Empty<int>() });
        }
        static Model Alias()
        {
            var model = new Model();
            model.inputs.Add(new Model.Input { index = 0, name = "residual" });
            model.inputs.Add(new Model.Input { index = 1, name = "projected" });
            model.outputs.Add(new Model.Output { index = 1, name = "output" });
            return model;
        }
        [Test] public void EmptySegmentMustProveOutputAliasesAnInputWithoutCpuWork()
        {
            var result = EvaluateModel(Alias());
            Assert.That((bool)result["success"], Is.True);
            Assert.That((string)result["execution_kind"], Is.EqualTo("input_alias"));
            Assert.That((int)result["computed_layer_output_count"], Is.Zero);
            Assert.That((int)result["input_alias_output_indices"][0], Is.EqualTo(1));
            Assert.That((bool)result["tensor_values_downloaded_by_auditor"], Is.False);
        }
        [TestCase("no_outputs")] [TestCase("unknown_output")] [TestCase("duplicate_input")]
        [TestCase("duplicate_output")] [TestCase("cpu_worker")] [TestCase("cpu_alias")] [TestCase("flush")]
        public void EmptyGraphIsNotAnAliasWithoutCompleteRoutingAndGpuOwnership(string problem)
        {
            var model = Alias();
            if (problem == "no_outputs") model.outputs.Clear();
            if (problem == "unknown_output") model.outputs[0] = new Model.Output { index = 5, name = "missing" };
            if (problem == "duplicate_input") model.inputs.Add(model.inputs[0]);
            if (problem == "duplicate_output") model.outputs.Add(model.outputs[0]);
            var error = Assert.Throws<TargetInvocationException>(() => EvaluateModel(model,
                problem == "cpu_worker" ? "CPU" : "GPUCompute", problem == "cpu_alias" ? new[] { 1 } : null,
                problem == "flush" ? new[] { 1 } : null));
            Assert.That(error.InnerException, Is.InstanceOf<InvalidOperationException>());
        }
        static Type Required()
        {
            var type = typeof(PlayerBuildInfo).Assembly.GetType("EmbeddingGemma.Validation.PlayerGpuExecutionAudit");
            Assert.That(type, Is.Not.Null, "Real Player acceptance must inspect the live Worker's CPU assignments.");
            return type;
        }
        static JObject Evaluate(string backend, int[] computed, int[] cpu, int[] flush)
            => (JObject)Required().GetMethod("Evaluate").Invoke(null, new object[] { backend, computed, cpu, flush });
        [Test]
        public void ComputationAndIntermediateFlushAreAuditedSeparatelyFromCpuInputs()
        {
            var result = Evaluate("GPUCompute", new[] { 2, 3 }, new[] { 0 }, Array.Empty<int>());
            Assert.That((bool)result["success"], Is.True);
            Assert.That((int)result["computed_layer_output_count"], Is.EqualTo(2));
            Assert.That(result["computation_cpu_fallback_indices"].HasValues, Is.False);
            Assert.That(result["intermediate_flush_indices"].HasValues, Is.False);
        }
        [TestCase("cpu_worker")] [TestCase("computation_fallback")] [TestCase("flush")] [TestCase("no_graph")]
        public void IncompleteGpuExecutionCannotBeReportedAsSuccess(string problem)
        {
            var error = Assert.Throws<TargetInvocationException>(() => Evaluate(problem == "cpu_worker" ? "CPU" : "GPUCompute",
                problem == "no_graph" ? Array.Empty<int>() : new[] { 2, 3 },
                problem == "computation_fallback" ? new[] { 3 } : Array.Empty<int>(),
                problem == "flush" ? new[] { 3 } : Array.Empty<int>()));
            Assert.That(error.InnerException, Is.InstanceOf<InvalidOperationException>());
        }
    }
}
