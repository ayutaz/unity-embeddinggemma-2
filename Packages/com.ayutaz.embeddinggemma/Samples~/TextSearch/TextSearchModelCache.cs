using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using Newtonsoft.Json.Linq;
using UnityEngine;
using UnityEngine.Networking;

namespace EmbeddingGemma.Samples
{
    public sealed class TextSearchModelStage : IDisposable
    {
        public string ModelPath, TokenizerPath, Error, CacheRejectedError;
        public bool Success, CacheReused;
        public int TransferredFiles;
        public double Milliseconds, AuditMilliseconds;
        public Dictionary<string, string> HashBackends;
        internal FileStream Lease;
        public void Dispose() { Lease?.Dispose(); Lease = null; }
    }
    public static class TextSearchModelCache
    {
        const string Owner = "EmbeddingGemma sample model cache v1\n";
        static readonly string[] Models = { "model-fp32.sentis", "model-float16.sentis" };
        static readonly string[] Names = { "model-fp32.sentis", "model-float16.sentis", "tokenizer.json" };
        static readonly string[] Directories = { "active", "pending", "previous", "incoming" };
        static readonly string[] Files = new[] { "preparation.json" }.Concat(Names).ToArray();
        public static string Combine(string directory, string name)
            => directory.Contains("://") ? directory.TrimEnd('/') + "/" + name : Path.Combine(directory, name);
        public static IEnumerator Resolve(string model, string tokenizer, string root, TextSearchModelStage result,
            Func<string, string, IEnumerator> transfer = null)
        {
            if (result == null) throw new ArgumentNullException(nameof(result));
            if (result.Lease != null) throw new InvalidOperationException("Release the previous stage before reusing it.");
            result.Success = false; result.CacheReused = false; result.Error = null; result.CacheRejectedError = null;
            result.TransferredFiles = 0; result.ModelPath = null; result.TokenizerPath = null; result.HashBackends = null;
            result.AuditMilliseconds = 0; var clock = Stopwatch.StartNew();
            // Own every nested enumerator so explicit cancellation disposes requests and locks.
            var stack = new Stack<IEnumerator>(); stack.Push(Core(model, tokenizer, root, result, transfer));
            try
            {
                while (stack.Count > 0)
                {
                    var moved = false; object current = null;
                    try
                    {
                        moved = stack.Peek().MoveNext();
                        if (moved) current = stack.Peek().Current;
                        else (stack.Pop() as IDisposable)?.Dispose();
                    }
                    catch (Exception exception) { result.Error = exception.GetType().Name + ": " + exception.Message; break; }
                    if (!moved) continue;
                    if (current is IEnumerator child) stack.Push(child); else yield return current;
                }
            }
            finally
            {
                while (stack.Count > 0)
                    try { (stack.Pop() as IDisposable)?.Dispose(); }
                    catch (Exception exception) { result.Error ??= "Cache disposal: " + exception.Message; }
                if (!result.Success) result.Error ??= "Preparation cancelled.";
                if (result.Error != null) { result.Success = false; result.Dispose(); }
                clock.Stop(); result.Milliseconds = clock.Elapsed.TotalMilliseconds;
            }
        }
        static IEnumerator Core(string model, string tokenizer, string root, TextSearchModelStage result, Func<string, string, IEnumerator> transfer)
        {
            var slash = model?.LastIndexOf('/') ?? -1;
            var name = slash < 0 ? "" : model.Substring(slash + 1);
            var source = slash < 0 ? "" : model.Substring(0, slash);
            if (!Models.Contains(name) || tokenizer != source + "/tokenizer.json" || !LocalSource(source))
                throw new NotSupportedException("Unsupported model source; use sibling prepared files in local file or APK URLs.");
            root = Path.GetFullPath(root);
            if (root.Length > Path.GetPathRoot(root).Length) root = root.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
            Prepare(root);
            FileStream lease = new(Path.Combine(root, ".lock"), FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None);
            try
            {
                ValidateRoot(root); Recover(root); Delete(root, "incoming"); Delete(root, "pending");
                var incoming = Path.Combine(root, "incoming"); Directory.CreateDirectory(incoming);
                var manifestPath = Path.Combine(incoming, "preparation.json");
                yield return Transfer(source + "/preparation.json", manifestPath, result, transfer);
                var receipt = ReadManifest(manifestPath);
                var active = Path.Combine(root, "active"); Dictionary<string, string> hashes = null;
                if (Directory.Exists(active) && File.Exists(Path.Combine(active, name)))
                {
                    try
                    {
                        if (JToken.DeepEquals(receipt, ReadManifest(Path.Combine(active, "preparation.json"))))
                        { hashes = Audit(active, name, receipt, result); result.CacheReused = true; }
                    }
                    catch (Exception exception) { result.CacheRejectedError = exception.GetType().Name + ": " + exception.Message; }
                }
                if (hashes == null)
                {
                    var pending = Path.Combine(root, "pending"); Directory.CreateDirectory(pending);
                    File.Move(manifestPath, Path.Combine(pending, "preparation.json"));
                    foreach (var file in new[] { name, "tokenizer.json" }) yield return Transfer(source + "/" + file, Path.Combine(pending, file), result, transfer);
                    hashes = Audit(pending, name, receipt, result);
                    if (Directory.Exists(active)) Directory.Move(active, Path.Combine(root, "previous"));
                    try { Directory.Move(pending, active); }
                    catch { if (!Directory.Exists(active) && Directory.Exists(Path.Combine(root, "previous"))) Directory.Move(Path.Combine(root, "previous"), active); throw; }
                    Delete(root, "previous");
                }
                Delete(root, "incoming");
                result.ModelPath = Path.Combine(active, name); result.TokenizerPath = Path.Combine(active, "tokenizer.json");
                result.HashBackends = hashes; result.Lease = lease; lease = null; result.Success = true;
            }
            finally { try { Delete(root, "incoming"); } finally { lease?.Dispose(); } }
        }
        static bool LocalSource(string source)
        {
            if (source.StartsWith("jar:", StringComparison.Ordinal))
            {
                var boundary = source.IndexOf("!/", StringComparison.Ordinal);
                if (boundary < 0) return false;
                source = source.Substring(4, boundary - 4);
            }
            return Uri.TryCreate(source, UriKind.Absolute, out var uri) && uri.IsFile && !uri.IsUnc &&
                (string.IsNullOrEmpty(uri.Host) || uri.IsLoopback);
        }
        static JObject ReadManifest(string path)
        {
            if (new FileInfo(path).Length > 65536) throw new InvalidDataException("Invalid preparation: manifest is too large.");
            var data = JObject.Parse(File.ReadAllText(path), new Newtonsoft.Json.Linq.JsonLoadSettings { DuplicatePropertyNameHandling = DuplicatePropertyNameHandling.Error });
            bool Hex(JToken value, int length) => value?.Type == JTokenType.String && Regex.IsMatch((string)value, "\\A[a-f0-9]{" + length + "}\\z");
            var files = data["files"] as JArray;
            if (data["success"]?.Type != JTokenType.Boolean || !(bool)data["success"] || (string)data["unityVersion"] != Application.unityVersion ||
                !Hex(data["sourceCommit"], 40) || !Hex(data["searchSourceCommit"], 40) || !Hex(data["modelSha256"], 64) ||
                files == null || files.Count != 3 || !files.Select(file => (string)file["name"]).SequenceEqual(Names) ||
                files.Any(file => file["bytes"]?.Type != JTokenType.Integer || (long)file["bytes"] <= 0 || !Hex(file["sha256"], 64)))
                throw new InvalidDataException("Invalid preparation: successful pinned source and exactly three fixed file descriptors are required.");
            return data;
        }
        static Dictionary<string, string> Audit(string directory, string model, JObject receipt, TextSearchModelStage result)
        {
            var clock = Stopwatch.StartNew();
            try
            {
                var expectedNames = new[] { "preparation.json", model, "tokenizer.json" };
                if (!Directory.GetFiles(directory).Select(Path.GetFileName).OrderBy(name => name, StringComparer.Ordinal)
                    .SequenceEqual(expectedNames.OrderBy(name => name, StringComparer.Ordinal))) throw new IOException("Unexpected cache selection.");
                var backends = new Dictionary<string, string>();
                foreach (var name in new[] { model, "tokenizer.json" })
                {
                    var entry = receipt["files"].Single(file => (string)file["name"] == name);
                    using var stream = File.OpenRead(Path.Combine(directory, name));
                    if (stream.Length != (long)entry["bytes"]) throw new InvalidDataException("Prepared file length differs: " + name);
                    var hash = TextSearchStreamHash.Compute(stream, out var backend);
                    if (hash != (string)entry["sha256"]) throw new InvalidDataException("Prepared file SHA-256 differs: " + name);
                    backends[name] = backend;
                }
                return backends;
            }
            finally { clock.Stop(); result.AuditMilliseconds += clock.Elapsed.TotalMilliseconds; }
        }
        static IEnumerator Transfer(string source, string final, TextSearchModelStage result, Func<string, string, IEnumerator> transfer)
        {
            var partial = final + ".part";
            try { yield return (transfer ?? Download)(source, partial); File.Move(partial, final); result.TransferredFiles++; }
            finally { if (File.Exists(partial)) File.Delete(partial); }
        }
        static IEnumerator Download(string source, string output)
        {
            using var request = new UnityWebRequest(source, UnityWebRequest.kHttpVerbGET);
            request.downloadHandler = new DownloadHandlerFile(output) { removeFileOnAbort = true }; request.timeout = 300;
            var operation = request.SendWebRequest(); while (!operation.isDone) yield return null;
            if (request.result != UnityWebRequest.Result.Success) throw new IOException(request.error ?? "Local model transfer failed.");
        }
        static void RejectLink(string path) { if ((File.GetAttributes(path) & FileAttributes.ReparsePoint) != 0) throw new IOException("Unexpected cache link."); }
        static void Prepare(string root)
        {
            if (Directory.Exists(root))
            {
                RejectLink(root);
                if (Directory.EnumerateFileSystemEntries(root).Any() && (!File.Exists(Path.Combine(root, ".owner")) || File.ReadAllText(Path.Combine(root, ".owner")) != Owner))
                    throw new IOException("Model cache directory is not owned by this sample.");
            }
            else Directory.CreateDirectory(root);
            var path = Path.Combine(root, ".owner");
            if (!File.Exists(path)) { using var file = new FileStream(path, FileMode.CreateNew, FileAccess.Write, FileShare.None); var bytes = Encoding.UTF8.GetBytes(Owner); file.Write(bytes, 0, bytes.Length); }
        }
        static void ValidateRoot(string root)
        {
            foreach (var entry in Directory.GetFileSystemEntries(root))
            {
                RejectLink(entry); var name = Path.GetFileName(entry);
                if (Directory.Exists(entry)) { if (!Directories.Contains(name)) throw new IOException("Unexpected cache directory."); ValidateDirectory(entry); }
                else if (name != ".owner" && name != ".lock") throw new IOException("Unexpected cache file.");
            }
        }
        static void ValidateDirectory(string directory)
        {
            RejectLink(directory);
            foreach (var entry in Directory.GetFileSystemEntries(directory))
            {
                RejectLink(entry); var name = Path.GetFileName(entry);
                if (Directory.Exists(entry) || !Files.Any(file => name == file || name == file + ".part")) throw new IOException("Unexpected cache contents: " + name);
            }
        }
        static void Delete(string root, string name)
        {
            if (!Directories.Contains(name)) throw new ArgumentException("Only fixed owned directories can be removed.");
            var directory = Path.GetFullPath(Path.Combine(root, name));
            if (Path.GetDirectoryName(directory) != root) throw new IOException("Cache cleanup escaped the root.");
            if (!Directory.Exists(directory)) return;
            ValidateDirectory(directory); foreach (var file in Directory.GetFiles(directory)) File.Delete(file); Directory.Delete(directory, false);
        }
        static void Recover(string root)
        {
            var previous = Path.Combine(root, "previous"); var active = Path.Combine(root, "active");
            if (!Directory.Exists(previous)) return;
            if (Directory.Exists(active))
            {
                var valid = false;
                try { var model = Models.Single(name => File.Exists(Path.Combine(active, name))); Audit(active, model, ReadManifest(Path.Combine(active, "preparation.json")), new TextSearchModelStage()); valid = true; }
                catch (Exception) { }
                if (valid) { Delete(root, "previous"); return; }
                Delete(root, "active");
            }
            Directory.Move(previous, active);
        }
    }
}
