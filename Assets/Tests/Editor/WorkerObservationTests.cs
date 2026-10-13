using System;
using System.Collections;
using System.Linq;
using System.Reflection;
using NUnit.Framework;
using Unity.InferenceEngine;

namespace EmbeddingGemma.Tests
{
    public sealed class WorkerObservationTests
    {
        [TestCase(BackendType.CPU, false)] [TestCase(BackendType.GPUCompute, false)]
        [TestCase(BackendType.CPU, true)] [TestCase(BackendType.GPUCompute, true)]
        public void LayerObservationPreservesNormalFinalBitsAndCopiesLiveOutputs(BackendType backend, bool cpuShapeDependency)
        {
            var type = AppDomain.CurrentDomain.GetAssemblies().Select(assembly => assembly.GetType("EmbeddingGemma.Tests.WorkerObservation"))
                .FirstOrDefault(value => value != null);
            Assert.That(type, Is.Not.Null, "Unchanged-model layer observer is missing.");
            var shape = new TensorShape(1, 2);
            var graph = new FunctionalGraph(); var x = graph.AddInput<float>(shape, "input");
            var model = graph.Compile(x + Functional.Constant(shape, new[] { 1f, 1f }));
            if (cpuShapeDependency)
            {
                var shapeIndex = model.constants.Select(value => value.index).Concat(model.layers.SelectMany(value => value.outputs)).Max() + 1;
                var shapeType = typeof(Model).Assembly.GetType("Unity.InferenceEngine.Layers.Shape");
                var shapeLayer = (Layer)Activator.CreateInstance(shapeType, new object[] { 0, 2 });
                shapeLayer.inputs = new[] { model.inputs[0].index }; shapeLayer.outputs = new[] { shapeIndex };
                var reshapeType = typeof(Model).Assembly.GetType("Unity.InferenceEngine.Layers.Reshape");
                var reshape = (Layer)Activator.CreateInstance(reshapeType, new object[] { false });
                reshape.inputs = new[] { model.outputs[0].index, shapeIndex }; reshape.outputs = new[] { shapeIndex + 1 };
                model.layers.Insert(0, shapeLayer); model.layers.Add(reshape);
                model.outputs[0] = new Model.Output { name = "output", index = shapeIndex + 1 };
            }
            var layers = model.layers.Select(layer => layer.GetType().Name + ":" + string.Join(",", layer.outputs)).ToArray();
            using var input = new Tensor<float>(shape, new[] { 1f, 2f });
            if (backend == BackendType.GPUCompute) ComputeTensorData.Pin(input);
            var result = type.GetMethod("Observe").Invoke(null, new object[] { model, backend, input, shape, 4 });
            var resultType = result.GetType();
            Assert.That((float[])resultType.GetField("Baseline").GetValue(result), Is.EqualTo(new[] { 2f, 3f }));
            Assert.That((float[])resultType.GetField("Final").GetValue(result), Is.EqualTo(new[] { 2f, 3f }));
            var snapshots = (IList)resultType.GetField("Snapshots").GetValue(result);
            Assert.That(snapshots.Count, Is.GreaterThanOrEqualTo(1));
            var snapshot = snapshots[0]; var snapshotType = snapshot.GetType();
            Assert.That((float[])snapshotType.GetField("Values").GetValue(snapshot), Is.EqualTo(new[] { 2f, 3f }));
            Assert.That(snapshotType.GetField("Backend").GetValue(snapshot), Is.EqualTo(backend));
            Assert.That(model.layers.Select(layer => layer.GetType().Name + ":" + string.Join(",", layer.outputs)), Is.EqualTo(layers));
            Assert.That(model.outputs.Count, Is.EqualTo(1), "Do not add debug outputs or change optimization.");
        }
    }
}
