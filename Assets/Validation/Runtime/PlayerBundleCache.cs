using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using Newtonsoft.Json.Linq;

namespace EmbeddingGemma.Validation
{
    public static class PlayerBundleCache
    {
        const string Owner = "EmbeddingGemma validation bundle cache v1\n";
        static readonly string[] Directories = { "active", "pending", "previous", "incoming" };
        static readonly string[] Files = new[] { "bundle.json" }.Concat(PlayerBundleLoader.FileNames).ToArray();
        public static IEnumerator Resolve(string source, string root, PlayerBundleStage result, Func<string, string, IEnumerator> transfer = null)
        {
            PlayerBundleAudit audit = null;
            try { yield return ResolveAudited(source, root, result, value => audit = value, transfer); }
            finally { audit?.Dispose(); }
        }
        internal static IEnumerator ResolveAudited(string source, string root, PlayerBundleStage result, Action<PlayerBundleAudit> complete,
            Func<string, string, IEnumerator> transfer = null)
        {
            var timer = Stopwatch.StartNew();
            result.source = source; result.directory = null; result.error = null; result.cacheRejectedError = null;
            result.transferCompleted = false; result.cacheReused = false; result.auditPassed = false; result.transferredFiles = 0; result.auditMilliseconds = 0;
            result.transport = transfer == null ? "UnityWebRequest.DownloadHandlerFile" : "injected_transport";
            var stack = new Stack<IEnumerator>(); stack.Push(Core(source, root, result, complete, transfer));
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
                    catch (Exception exception) { result.error ??= exception.GetType().Name + ": " + exception.Message; break; }
                    if (!moved) continue;
                    if (current is IEnumerator child) stack.Push(child);
                    else yield return current;
                }
            }
            finally
            {
                while (stack.Count > 0)
                    try { (stack.Pop() as IDisposable)?.Dispose(); }
                    catch (Exception exception) { result.error ??= "Cache disposal: " + exception.Message; }
                timer.Stop(); result.milliseconds = timer.Elapsed.TotalMilliseconds;
                if (result.error != null) result.auditPassed = false;
            }
        }
        static IEnumerator Core(string source, string root, PlayerBundleStage result, Action<PlayerBundleAudit> complete, Func<string, string, IEnumerator> transfer)
        {
            PlayerBundleStager.RequireLocalSource(source);
            root = Path.GetFullPath(root); Prepare(root);
            FileStream lease = new(Path.Combine(root, ".lock"), FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None);
            try
            {
                ValidateRoot(root); Recover(root);
                DeleteOwnedDirectory(root, "incoming"); DeleteOwnedDirectory(root, "pending");
                var incoming = Path.Combine(root, "incoming"); Directory.CreateDirectory(incoming);
                var manifest = Path.Combine(incoming, "bundle.json");
                yield return PlayerBundleStager.TransferFile(source.TrimEnd('/') + "/bundle.json", manifest, result, transfer);
                if (result.error != null) yield break;
                var receipt = JObject.Parse(File.ReadAllText(manifest)); PlayerBundleLoader.ValidateReceipt(receipt);
                var active = Path.Combine(root, "active"); PlayerBundleAudit audit = null;
                if (Directory.Exists(active))
                {
                    var clock = Stopwatch.StartNew();
                    try
                    {
                        if (JToken.DeepEquals(receipt, JObject.Parse(File.ReadAllText(Path.Combine(active, "bundle.json")))))
                        { audit = PlayerBundleLoader.Audit(active); result.cacheReused = true; }
                    }
                    catch (Exception exception) { result.cacheRejectedError = exception.GetType().Name + ": " + exception.Message; }
                    finally { clock.Stop(); result.auditMilliseconds += clock.Elapsed.TotalMilliseconds; }
                }
                if (audit == null)
                {
                    var pending = Path.Combine(root, "pending"); Directory.CreateDirectory(pending);
                    File.Move(manifest, Path.Combine(pending, "bundle.json"));
                    foreach (var name in PlayerBundleLoader.FileNames)
                    {
                        yield return PlayerBundleStager.TransferFile(source.TrimEnd('/') + "/" + name, Path.Combine(pending, name), result, transfer);
                        if (result.error != null) yield break;
                    }
                    result.transferCompleted = true;
                    var clock = Stopwatch.StartNew();
                    try { audit = PlayerBundleLoader.Audit(pending); }
                    finally { clock.Stop(); result.auditMilliseconds += clock.Elapsed.TotalMilliseconds; }
                    Promote(root); audit.Bundle.Directory = active;
                }
                DeleteOwnedDirectory(root, "incoming");
                result.directory = active; result.transferCompleted = true; result.auditPassed = true;
                audit.Lease = lease; lease = null;
                try { complete(audit); }
                catch { audit.Dispose(); throw; }
            }
            finally
            {
                // Only this manifest directory is cleaned here. Incomplete pending is bounded
                // and is discarded under the lock on the next attempt.
                try { DeleteOwnedDirectory(root, "incoming"); }
                finally { lease?.Dispose(); }
            }
        }
        static void Prepare(string root)
        {
            if (Directory.Exists(root))
            {
                RejectLink(root);
                if (Directory.EnumerateFileSystemEntries(root).Any() &&
                    (!File.Exists(Path.Combine(root, ".owner")) || File.ReadAllText(Path.Combine(root, ".owner")) != Owner))
                    throw new IOException("Cache directory is not owned by this validation harness.");
            }
            else Directory.CreateDirectory(root);
            var owner = Path.Combine(root, ".owner");
            if (!File.Exists(owner))
            {
                using var file = new FileStream(owner, FileMode.CreateNew, FileAccess.Write, FileShare.None);
                var bytes = Encoding.UTF8.GetBytes(Owner); file.Write(bytes, 0, bytes.Length);
            }
        }
        static void RejectLink(string path)
        {
            if ((File.GetAttributes(path) & FileAttributes.ReparsePoint) != 0) throw new IOException("Cache links are not owned storage: " + path);
        }
        static void ValidateRoot(string root)
        {
            RejectLink(root);
            foreach (var entry in Directory.GetFileSystemEntries(root))
            {
                RejectLink(entry); var name = Path.GetFileName(entry);
                if (Directory.Exists(entry))
                {
                    if (!Directories.Contains(name)) throw new IOException("Unexpected cache directory: " + name);
                    ValidateBundleDirectory(entry);
                }
                else if (name != ".owner" && name != ".lock") throw new IOException("Unexpected cache file: " + name);
            }
        }
        static void ValidateBundleDirectory(string path)
        {
            RejectLink(path);
            foreach (var entry in Directory.GetFileSystemEntries(path))
            {
                RejectLink(entry); var name = Path.GetFileName(entry);
                if (Directory.Exists(entry) || !Files.Any(file => name == file || name == file + ".part"))
                    throw new IOException("Unexpected file in owned cache: " + name);
            }
        }
        static void DeleteOwnedDirectory(string root, string name)
        {
            if (!Directories.Contains(name)) throw new ArgumentException("Only fixed cache directories may be cleaned.");
            var path = Path.GetFullPath(Path.Combine(root, name));
            if (Path.GetDirectoryName(path) != root) throw new IOException("Cache cleanup escaped its root.");
            if (!Directory.Exists(path)) return;
            ValidateBundleDirectory(path);
            foreach (var file in Directory.GetFiles(path)) File.Delete(file);
            Directory.Delete(path, false); // Never recurse into user directories or follow links.
        }
        static void Recover(string root)
        {
            var previous = Path.Combine(root, "previous"); var active = Path.Combine(root, "active");
            if (!Directory.Exists(previous)) return;
            if (Directory.Exists(active))
            {
                var valid = false;
                try { using var audit = PlayerBundleLoader.Audit(active); valid = true; }
                catch (Exception) { /* A crashed publication can be repaired from the retained previous copy. */ }
                if (valid) { DeleteOwnedDirectory(root, "previous"); return; }
                DeleteOwnedDirectory(root, "active");
            }
            Directory.Move(previous, active);
        }
        static void Promote(string root)
        {
            var active = Path.Combine(root, "active"); var previous = Path.Combine(root, "previous");
            if (Directory.Exists(active)) Directory.Move(active, previous);
            try { Directory.Move(Path.Combine(root, "pending"), active); }
            catch
            {
                if (!Directory.Exists(active) && Directory.Exists(previous)) Directory.Move(previous, active);
                throw;
            }
            DeleteOwnedDirectory(root, "previous");
        }
    }
}
