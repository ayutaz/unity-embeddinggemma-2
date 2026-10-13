using System;
using System.IO;
using System.Linq;
using System.Xml.Linq;
using NUnit.Framework;
using UnityEditor.Build;

namespace EmbeddingGemma.Tests
{
    public sealed class GpuConstantLinkerPreservationTests
    {
        [Test]
        public void PackageSuppliesPrivateSentisMembersThroughLinkerCallback()
        {
            var type = AppDomain.CurrentDomain.GetAssemblies()
                .Select(assembly => assembly.GetType("EmbeddingGemma.Editor.GpuConstantLinkerPreservation"))
                .FirstOrDefault(value => value != null);
            Assert.That(type, Is.Not.Null, "Package must register an explicit Unity linker callback; package link.xml alone is ignored.");
            Assert.That(typeof(IUnityLinkerProcessor).IsAssignableFrom(type), Is.True);
            var processor = (IUnityLinkerProcessor)Activator.CreateInstance(type);
            var path = processor.GenerateAdditionalLinkXmlFile(null, null);
            var package = UnityEditor.PackageManager.PackageInfo.FindForAssetPath("Packages/com.ayutaz.embeddinggemma/Runtime/GpuConstantWorker.cs");
            Assert.That(Path.GetFullPath(path), Is.EqualTo(Path.GetFullPath(Path.Combine(package.resolvedPath, "Runtime/link.xml"))));
            var assembly = XDocument.Load(path).Root.Elements("assembly").Single();
            Assert.That((string)assembly.Attribute("fullname"), Is.EqualTo("Unity.InferenceEngine"));
            var worker = assembly.Elements("type").Single(value => (string)value.Attribute("fullname") == "Unity.InferenceEngine.Worker");
            Assert.That(worker.Elements("field").Select(value => (string)value.Attribute("name")),
                Is.EquivalentTo(new[] { "m_Storage", "m_LayerCPUFallback" }));
            var storage = assembly.Elements("type").Single(value => (string)value.Attribute("fullname") == "Unity.InferenceEngine.ModelStorage");
            Assert.That(storage.Elements("method").Select(value => (string)value.Attribute("name")), Is.EqualTo(new[] { "GetTensor" }));
            Assert.That(assembly.DescendantsAndSelf().Attributes("preserve").Any(), Is.False, "Keep explicit members without broadly preserving all Sentis code.");
        }
    }
}
