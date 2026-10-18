using System;
using System.Reflection;
using System.Runtime.ExceptionServices;

namespace EmbeddingGemma.Validation.Tests
{
    /// <summary>Validation only: invoke the pinned Editor database's explicit rebuild.</summary>
    public static class EditorIndexRebuild
    {
        public static void Begin(object database)
        {
            if (database == null) throw new ArgumentNullException(nameof(database));
            var method = database.GetType().GetMethod("Build", BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic,
                null, Type.EmptyTypes, null);
            if (method == null || method.ReturnType != typeof(void) || method.ContainsGenericParameters)
                throw new NotSupportedException("Pinned Editor Search database Build() contract is unavailable.");
            try { method.Invoke(database, null); }
            catch (TargetInvocationException error) when (error.InnerException != null)
            { ExceptionDispatchInfo.Capture(error.InnerException).Throw(); throw; }
        }
    }
}
