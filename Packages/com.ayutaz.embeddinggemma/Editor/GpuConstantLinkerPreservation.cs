using System;
using System.IO;
using UnityEditor;
using UnityEditor.Build;
using UnityEditor.Build.Reporting;
using UnityEditor.UnityLinker;

namespace EmbeddingGemma.Editor
{
    /// <summary>Registers package-local preservation metadata, which Unity does not discover automatically.</summary>
    public sealed class GpuConstantLinkerPreservation : IUnityLinkerProcessor
    {
        public int callbackOrder => 0;

        public string GenerateAdditionalLinkXmlFile(BuildReport report, UnityLinkerBuildPipelineData data)
        {
            var package = UnityEditor.PackageManager.PackageInfo.FindForAssetPath("Packages/com.ayutaz.embeddinggemma/Runtime/GpuConstantWorker.cs");
            if (package == null) throw new BuildFailedException("EmbeddingGemma Runtime package is unresolved.");
            var path = Path.Combine(package.resolvedPath, "Runtime/link.xml");
            if (!File.Exists(path)) throw new BuildFailedException("EmbeddingGemma GPU constant preservation metadata is missing.");
            return path;
        }
    }
}
