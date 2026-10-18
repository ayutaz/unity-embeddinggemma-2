using System;
using UnityEngine;
using UnityEngine.Rendering;

namespace EmbeddingGemma.Validation
{
    /// <summary>Runtime admission only. Passing this check is never evidence of GPU execution.</summary>
    public static class PlayerMixedRuntimePolicy
    {
        public static void Validate(PlayerBuildInfo build, RuntimePlatform platform, GraphicsDeviceType graphics,
            bool editor, bool development, bool compute, string unity)
        {
            if (editor || development || !compute) throw new NotSupportedException("An actual Release Player and compute GPU are required.");
            if (build == null) throw new ArgumentException("Pinned build provenance is required.");
            build.Validate();
            var target = platform == RuntimePlatform.WindowsPlayer ? "StandaloneWindows64" : platform == RuntimePlatform.Android ? "Android"
                : platform == RuntimePlatform.IPhonePlayer ? "iOS" : null;
            if (target == null || build.target != target || build.scriptingBackend != "IL2CPP" || build.stripping != "High" || build.unityVersion != unity
                || platform == RuntimePlatform.Android && graphics != GraphicsDeviceType.Vulkan
                || platform == RuntimePlatform.IPhonePlayer && graphics != GraphicsDeviceType.Metal)
                throw new NotSupportedException("Matching Release IL2CPP High build and supported device graphics context are required.");
        }
    }
}
