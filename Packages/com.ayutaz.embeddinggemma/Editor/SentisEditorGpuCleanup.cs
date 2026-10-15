using System;
using System.Linq;
using System.Reflection;
using UnityEditor;
using UnityEngine;

namespace EmbeddingGemma.Editor
{
    /// <summary>
    /// Sentis 2.6.1 registers its shared compute-buffer cleanup from a runtime
    /// initialization callback. EditMode inference can create those resources
    /// without entering PlayMode, so register the same cleanup for Editor exit
    /// and assembly reload. Never release shared resources during inference.
    /// </summary>
    [InitializeOnLoad]
    static class SentisEditorGpuCleanup
    {
        static SentisEditorGpuCleanup()
        {
            EditorApplication.quitting+=CleanupForEditorShutdown;
            AssemblyReloadEvents.beforeAssemblyReload+=CleanupForEditorShutdown;
        }

        internal static void CleanupForEditorShutdown()
        {
            var assembly=AppDomain.CurrentDomain.GetAssemblies().FirstOrDefault(value=>value.GetName().Name=="Unity.InferenceEngine");
            if(assembly==null)return;
            var type=assembly.GetType("Unity.InferenceEngine.ComputeTensorDataReaper");
            var cleanup=type?.GetMethod("CleanupStaticResources",BindingFlags.Static|BindingFlags.NonPublic);
            if(cleanup==null)
            {
                Debug.LogError("EmbeddingGemma Editor GPU cleanup requires the pinned Sentis 2.6.1 shared-resource lifecycle.");
                return;
            }
            // Pinned upstream implementation is idempotent and releases only
            // its own dummy ComputeBuffer and async callback CommandBuffer.
            cleanup.Invoke(null,null);
        }
    }
}
