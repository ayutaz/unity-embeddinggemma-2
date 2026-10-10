using System;
using System.Collections;
using System.Diagnostics;
using System.IO;
using System.Linq;
using UnityEngine.Networking;

namespace EmbeddingGemma.Validation
{
    [Serializable] public sealed class PlayerBundleStage
    {
        public string source, directory, error, transport;
        public bool transferCompleted;
        public int transferredFiles;
        public double milliseconds;
    }
    public static class PlayerBundleStager
    {
        static readonly string[] Names = { "bundle.json", "model-fp32.sentis", "model-float16.sentis", "tokenizer.json", "reference.json", "search-reference.json" };
        public static IEnumerator Stage(string source, string destination, PlayerBundleStage result,
            Func<string, string, IEnumerator> transfer = null)
        {
            if (result == null) throw new ArgumentNullException(nameof(result));
            var timer = Stopwatch.StartNew();
            result.source = source; result.error = null; result.transferCompleted = false; result.transferredFiles = 0;
            result.transport = transfer == null ? "UnityWebRequest.DownloadHandlerFile" : "injected_transport";
            try
            {
                try
                {
                    if (string.IsNullOrWhiteSpace(source)) throw new ArgumentException("Bundle source is required.");
                    if (!source.Contains("://"))
                    {
                        result.directory = Path.GetFullPath(source); result.transport = "filesystem"; result.transferCompleted = true;
                    }
                    else
                    {
                        // This harness extracts local StreamingAssets; it does not download models from the Internet.
                        if (!(source.StartsWith("file://", StringComparison.Ordinal) ||
                            source.StartsWith("jar:file://", StringComparison.Ordinal) && source.Contains("!/")))
                            throw new NotSupportedException("Unsupported StreamingAssets source: " + source);
                        result.directory = Path.GetFullPath(destination);
                        if (Directory.Exists(result.directory) && Directory.EnumerateFileSystemEntries(result.directory).Any())
                            throw new IOException("Staging directory must be empty; previous runs are never overwritten.");
                        Directory.CreateDirectory(result.directory);
                    }
                }
                catch (Exception exception) { result.error = exception.GetType().Name + ": " + exception.Message; }
                if (result.error != null || result.transferCompleted) yield break;
                foreach (var name in Names)
                {
                    var final = Path.Combine(result.directory, name); var partial = final + ".part";
                    IEnumerator routine = null;
                    try
                    {
                        try { routine = (transfer ?? Download)(source.TrimEnd('/') + "/" + name, partial); }
                        catch (Exception exception) { result.error = exception.GetType().Name + ": " + name + ": " + exception.Message; }
                        if (result.error != null) yield break;
                        while (true)
                        {
                            var moved = false; object current = null;
                            try { moved = routine.MoveNext(); if (moved) current = routine.Current; }
                            catch (Exception exception) { result.error = exception.GetType().Name + ": " + name + ": " + exception.Message; }
                            if (result.error != null) yield break;
                            if (!moved) break;
                            yield return current;
                        }
                        // A transfer must complete before publishing the file under its fixed name.
                        try { File.Move(partial, final); result.transferredFiles++; }
                        catch (Exception exception) { result.error = exception.GetType().Name + ": " + name + ": " + exception.Message; }
                        if (result.error != null) yield break;
                    }
                    finally
                    {
                        try { (routine as IDisposable)?.Dispose(); }
                        catch (Exception exception) { result.error ??= "Transfer disposal: " + exception.Message; }
                        try { if (File.Exists(partial)) File.Delete(partial); }
                        catch (Exception exception) { result.error ??= "Partial-file cleanup: " + exception.Message; }
                    }
                    if (result.error != null) yield break;
                }
                // The caller must still perform the pinned receipt and full SHA-256 audit.
                result.transferCompleted = true;
            }
            finally { timer.Stop(); result.milliseconds = timer.Elapsed.TotalMilliseconds; }
        }
        static IEnumerator Download(string source, string output)
        {
            using var request = new UnityWebRequest(source, UnityWebRequest.kHttpVerbGET);
            request.downloadHandler = new DownloadHandlerFile(output) { removeFileOnAbort = true };
            request.timeout = 300;
            var operation = request.SendWebRequest();
            // Explicitly wait for completion in both Player coroutines and EditMode Test Runner.
            while (!operation.isDone) yield return null;
            if (request.result != UnityWebRequest.Result.Success) throw new IOException(request.error ?? "StreamingAssets transfer failed.");
        }
    }
}
