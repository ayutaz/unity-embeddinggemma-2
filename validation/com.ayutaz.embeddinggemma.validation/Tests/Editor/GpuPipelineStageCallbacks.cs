using System;
namespace EmbeddingGemma.Validation.Tests
{
    /// <summary>
    /// Validation-only marker boundaries around borrowed pipeline callbacks.
    /// Model submission ends before unused-cache cleanup; the next model starts
    /// after the preceding GPU fence and completed-backend cleanup. No tensor
    /// access, scheduling, cache policy or resource ownership is changed here.
    /// </summary>
    public sealed class GpuPipelineStageCallbacks : IDisposable
    {
        readonly Action[] before, completed, originalBefore, originalCompleted, wrappedBefore, wrappedCompleted;
        readonly Action<int> begin;
        readonly Action<int, bool> end;
        int openStage = -1, nextStage;
        bool active, disposed;

        public GpuPipelineStageCallbacks(Action[] before, Action[] completed, Action<int> begin, Action<int, bool> end)
        {
            if (before == null || completed == null || begin == null || end == null) throw new ArgumentNullException();
            if (before.Length == 0 || before.Length != completed.Length) throw new ArgumentException("Matching nonempty stage arrays required.");
            for (var i = 0; i < before.Length; i++)
                if (before[i] == null || completed[i] == null) throw new ArgumentNullException("stage", "Pinned cache callbacks must be present.");
            this.before = before; this.completed = completed; this.begin = begin; this.end = end;
            originalBefore = (Action[])before.Clone(); originalCompleted = (Action[])completed.Clone();
            wrappedBefore = new Action[before.Length]; wrappedCompleted = new Action[before.Length];
            for (var i = 0; i < before.Length; i++)
            {
                var stage = i;
                wrappedBefore[i] = () =>
                {
                    RequireActive();
                    if (openStage != stage || nextStage != stage) throw new InvalidOperationException("Unexpected model stage order.");
                    openStage = -1; end(stage, false);
                    originalBefore[stage]();
                };
                wrappedCompleted[i] = () =>
                {
                    RequireActive();
                    if (openStage != -1 || nextStage != stage) throw new InvalidOperationException("Model marker must close before completed cleanup.");
                    originalCompleted[stage]();
                    nextStage++;
                    if (nextStage < before.Length) Start(nextStage);
                };
                before[i] = wrappedBefore[i]; completed[i] = wrappedCompleted[i];
            }
        }
        void RequireActive()
        {
            if (disposed) throw new ObjectDisposedException(nameof(GpuPipelineStageCallbacks));
            if (!active) throw new InvalidOperationException("Begin a marker call before executing the borrowed pipeline.");
        }
        void Start(int stage) { begin(stage); openStage = stage; }
        public void BeginCall()
        {
            if (disposed) throw new ObjectDisposedException(nameof(GpuPipelineStageCallbacks));
            if (active) throw new InvalidOperationException("Marker calls must be serial.");
            active = true; nextStage = 0;
            try { Start(0); }
            catch { active = false; throw; }
        }
        public void EndCall()
        {
            if (!active) return;
            active = false;
            if (openStage < 0) return;
            var stage = openStage; openStage = -1; end(stage, true);
        }
        public void Dispose()
        {
            if (disposed) return;
            try { EndCall(); }
            finally
            {
                disposed = true;
                for (var i = 0; i < before.Length; i++)
                {
                    if (ReferenceEquals(before[i], wrappedBefore[i])) before[i] = originalBefore[i];
                    if (ReferenceEquals(completed[i], wrappedCompleted[i])) completed[i] = originalCompleted[i];
                }
            }
        }
    }
}
