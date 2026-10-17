using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Reflection;
using System.Runtime.ExceptionServices;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json.Linq;

namespace EmbeddingGemma.Validation.Tests
{
    /// <summary>Editor-only decorators; borrow all stage inputs/results and never read tensor values.</summary>
    public static class MediaStageTiming
    {
        public static T Wrap<T>(T stage, string name, Action<JObject> observe, bool observeSubmissions = false) where T : class
        {
            if (stage == null) throw new ArgumentNullException(nameof(stage));
            if (!typeof(T).IsInterface) throw new ArgumentException("A stage interface is required.");
            if (string.IsNullOrWhiteSpace(name)) throw new ArgumentException("A stage name is required.");
            if (observe == null) throw new ArgumentNullException(nameof(observe));
            var result = DispatchProxy.Create<T, TimingProxy>();
            var proxy = (TimingProxy)(object)result;
            proxy.Stage = stage; proxy.Name = name; proxy.Observe = observe; proxy.ObserveSubmissions = observeSubmissions;
            return result;
        }
        public static IDisposable Attach(object facade, Action<JObject> observe, bool observeSubmissions = false)
        {
            if (facade is GpuMediaSession session)
                facade = Field(session, "image").GetValue(session) ?? Field(session, "audio").GetValue(session);
            if (!(facade is ImageEmbedder) && !(facade is AudioEmbedder))
                throw new ArgumentException("An image/audio facade or open media session is required.");
            var scope = new Attached(facade);
            try
            {
                var pre = Field(facade, "preprocessor").GetValue(facade);
                var enc = Field(facade, "encoder").GetValue(facade);
                var back = Field(facade, "backbone").GetValue(facade);
                if (pre is TimingProxy || enc is TimingProxy || back is TimingProxy)
                    throw new InvalidOperationException("Detach existing stage timing before attaching again.");
                if (facade is ImageEmbedder)
                {
                    scope.Set("preprocessor", Wrap((IImagePreprocessor)pre, "preprocessing", observe, observeSubmissions));
                    scope.Set("encoder", enc is IAsyncImageFeatureEncoder asyncEncoder
                        ? (object)Wrap(asyncEncoder, "encoder", observe, observeSubmissions) : Wrap((IImageFeatureEncoder)enc, "encoder", observe, observeSubmissions));
                }
                else
                {
                    scope.Set("preprocessor", pre is IAsyncAudioPreprocessor asyncPre
                        ? (object)Wrap(asyncPre, "preprocessing", observe, observeSubmissions) : Wrap((IAudioPreprocessor)pre, "preprocessing", observe, observeSubmissions));
                    scope.Set("encoder", enc is IAsyncAudioFeatureEncoder asyncEncoder
                        ? (object)Wrap(asyncEncoder, "encoder", observe, observeSubmissions) : Wrap((IAudioFeatureEncoder)enc, "encoder", observe, observeSubmissions));
                }
                scope.Set("backbone", Wrap((IMultimodalBackbone)back, "backbone", observe, observeSubmissions));
                return scope;
            }
            catch { scope.Dispose(); throw; }
        }
        static FieldInfo Field(object target, string name) => target.GetType().GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)
            ?? throw new NotSupportedException("Pinned stage field is unavailable: " + name);
        sealed class Attached : IDisposable
        {
            readonly object facade;
            readonly List<(FieldInfo field, object original, object replacement)> bindings = new List<(FieldInfo, object, object)>();
            public Attached(object facade) { this.facade = facade; }
            public void Set(string name, object value)
            {
                var field = Field(facade, name); var original = field.GetValue(facade);
                field.SetValue(facade, value); bindings.Add((field, original, value));
            }
            public void Dispose()
            {
                foreach (var binding in bindings)
                    if (ReferenceEquals(binding.field.GetValue(facade), binding.replacement))
                        binding.field.SetValue(facade, binding.original);
                // A closed facade has cleared its borrowed fields; never resurrect it.
                bindings.Clear();
            }
        }
        public class TimingProxy : DispatchProxy
        {
            internal object Stage; internal string Name; internal Action<JObject> Observe; internal bool ObserveSubmissions;
            static long nextId;
            object Original(MethodInfo method, object[] args)
            {
                try { return method.Invoke(Stage, args); }
                catch (TargetInvocationException error) when (error.InnerException != null)
                { ExceptionDispatchInfo.Capture(error.InnerException).Throw(); throw; }
            }
            protected override object Invoke(MethodInfo method, object[] args)
            {
                if (method.IsSpecialName) return Original(method, args);
                var call = new Call(Name, method.Name, Observe, Interlocked.Increment(ref nextId), typeof(Task).IsAssignableFrom(method.ReturnType));
                try
                {
                    var result = ObserveSubmissions
                        ? SynchronousSubmissionTrace.Invoke(() => Original(method, args),
                            () => call.Submission("submitting", null), error => call.Submission("submitted", error))
                        : Original(method, args);
                    if (result is Task task)
                    {
                        if (method.ReturnType == typeof(Task)) return AwaitVoid(task, call);
                        if (!method.ReturnType.IsGenericType || method.ReturnType.GetGenericTypeDefinition() != typeof(Task<>))
                            throw new NotSupportedException("Expected Task or Task<T> stage return type.");
                        return typeof(TimingProxy).GetMethod(nameof(Await), BindingFlags.Instance | BindingFlags.NonPublic)
                            .MakeGenericMethod(method.ReturnType.GetGenericArguments()[0]).Invoke(this, new[] { result, call });
                    }
                    call.End(null); return result;
                }
                catch (Exception error) { call.End(error); throw; }
            }
            async Task<T> Await<T>(Task<T> task, Call call)
            {
                try { var result = await task; call.End(null); return result; }
                catch (Exception error) { call.End(error); throw; }
            }
            static async Task AwaitVoid(Task task, Call call)
            {
                try { await task; call.End(null); }
                catch (Exception error) { call.End(error); throw; }
            }
            sealed class Call
            {
                readonly Action<JObject> observe; readonly JObject header;
                readonly Process process = Process.GetCurrentProcess();
                readonly TimeSpan cpu; readonly Stopwatch watch; bool ended;
                public Call(string stage, string method, Action<JObject> observe, long id, bool asynchronous)
                {
                    this.observe = observe;
                    header = new JObject { ["stage"] = stage, ["method"] = method, ["call_id"] = id,
                        ["asynchronous"] = asynchronous, ["event"] = "begin", ["utc"] = DateTime.UtcNow.ToString("O"),
                        ["cpu_scope"] = "whole process during stage interval, includes other Editor threads",
                        ["wall_scope"] = "call through completion, including GPU waits and async frame scheduling",
                        ["host_peak_scope"] = "process lifetime, not this stage peak", ["gpu_elapsed_ns"] = null };
                    observe((JObject)header.DeepClone()); process.Refresh(); cpu = process.TotalProcessorTime; watch = Stopwatch.StartNew();
                }
                public void End(Exception error)
                {
                    if (ended) return; ended = true; watch.Stop();
                    try
                    {
                        process.Refresh(); var elapsedCpu = process.TotalProcessorTime - cpu;
                        GpuExecutionProfiler.ReadProcessMemory(process, out var current, out var peak);
                        var row = (JObject)header.DeepClone();
                        row["event"] = error == null ? "completed" : error is OperationCanceledException ? "cancelled" : "failed";
                        row["utc"] = DateTime.UtcNow.ToString("O"); row["wall_seconds"] = watch.Elapsed.TotalSeconds;
                        row["process_cpu_seconds"] = elapsedCpu.TotalSeconds;
                        row["host_working_set_bytes"] = current; row["host_lifetime_peak_bytes"] = peak;
                        row["unity_allocated_bytes"] = UnityEngine.Profiling.Profiler.GetTotalAllocatedMemoryLong();
                        row["error_type"] = error?.GetType().FullName;
                        observe(row);
                    }
                    catch when (error != null) { /* A measurement failure must not replace the original stage failure. */ }
                    finally { process.Dispose(); }
                }
                public void Submission(string eventName, Exception error)
                {
                    var row = (JObject)header.DeepClone(); row["event"] = eventName;
                    row["utc"] = DateTime.UtcNow.ToString("O"); row["error_type"] = error?.GetType().FullName;
                    row["submission_scope"] = "Synchronous invocation through return, including a pending Task; not async completion or measured GPU time.";
                    observe(row);
                }
            }
        }
    }
}
