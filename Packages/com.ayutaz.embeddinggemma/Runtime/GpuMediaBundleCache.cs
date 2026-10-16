using System;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json.Linq;

namespace EmbeddingGemma
{
    /// <summary>Keep this lease until the module session has closed, including pending GPU calls.</summary>
    public sealed class GpuMediaBundleLease : IDisposable
    {
        internal FileStream Lock;
        public GpuMediaBundle Bundle { get; internal set; }
        public string ModuleName { get; internal set; }
        /// <summary>Includes the small descriptor read on every acquisition.</summary>
        public int TransferredFiles { get; internal set; }
        public long TransferredBytes { get; internal set; }
        public int ReusedObjects { get; internal set; }
        public void Dispose() { Lock?.Dispose(); Lock = null; }
    }

    /// <summary>
    /// Audited local prepared-bundle cache. Content-addressed objects are shared
    /// across modules and producer revisions. An exclusive lease prevents repair
    /// while another session owns this cache. Never downloads or converts weights.
    /// </summary>
    public static class GpuMediaBundleCache
    {
        const string Owner = "EmbeddingGemma prepared GPU media cache v1\n";
        static readonly Regex ObjectName = new Regex("\\A[a-f0-9]{64}\\.(sentis|f32|json)\\z");
        static readonly Regex PendingName = new Regex("\\A[a-f0-9]{32}\\.part\\z");

        public static async Task<GpuMediaBundleLease> AcquireAsync(string sourceDirectory, string moduleName, string cacheDirectory,
            CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var source = Path.GetFullPath(sourceDirectory); var root = Path.GetFullPath(cacheDirectory);
            if (string.Equals(source.TrimEnd(Path.DirectorySeparatorChar), root.TrimEnd(Path.DirectorySeparatorChar), StringComparison.OrdinalIgnoreCase))
                throw new IOException("Source bundle and cache must be separate directories.");
            Prepare(root);
            var lease = new GpuMediaBundleLease { ModuleName = moduleName };
            string pending = null;
            try
            {
                lease.Lock = new FileStream(Path.Combine(root, ".lock"), FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None);
                Validate(root); var objects = Path.Combine(root, "objects"); var incoming = Path.Combine(root, "incoming");
                Directory.CreateDirectory(objects); Directory.CreateDirectory(incoming);
                // After a process crash only validated, sample-owned partials can remain.
                foreach (var orphan in Directory.GetFiles(incoming)) File.Delete(orphan);
                await Task.Yield(); cancellationToken.ThrowIfCancellationRequested();
                var manifest = Path.Combine(source, "bundle.json");
                if (new FileInfo(manifest).Length > 4 * 1024 * 1024) throw new InvalidDataException("Media descriptor is too large.");
                var json = File.ReadAllText(manifest); var bundle = GpuMediaBundle.Parse(json, root); var data = JObject.Parse(json);
                lease.TransferredFiles = 1; lease.TransferredBytes = new FileInfo(manifest).Length;
                foreach (var name in bundle.FilesForModule(moduleName))
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    var filename = name.StartsWith("objects/", StringComparison.Ordinal) ? name.Substring(8) : "";
                    var metadata = data["files"][name]; var sha = (string)metadata["sha256"]; var length = (long)metadata["bytes"];
                    if (!ObjectName.IsMatch(filename) || filename.Substring(0, 64) != sha)
                        throw new InvalidDataException("Cache modules must use content-addressed prepared objects.");
                    var destination = Path.Combine(objects, filename);
                    if (File.Exists(destination)) RejectLink(destination);
                    if (Matches(destination, length, sha)) { lease.ReusedObjects++; await Task.Yield(); continue; }
                    pending = Path.Combine(incoming, Guid.NewGuid().ToString("N") + ".part");
                    await Copy(Path.Combine(source, name), pending, cancellationToken);
                    if (!Matches(pending, length, sha)) throw new InvalidDataException("Transferred prepared object failed its size or SHA256 audit.");
                    cancellationToken.ThrowIfCancellationRequested();
                    if (File.Exists(destination)) File.Delete(destination);
                    File.Move(pending, destination); pending = null; lease.TransferredFiles++; lease.TransferredBytes += length;
                }
                cancellationToken.ThrowIfCancellationRequested(); lease.Bundle = bundle; return lease;
            }
            catch { lease.Dispose(); throw; }
            finally { if (pending != null && File.Exists(pending)) File.Delete(pending); }
        }
        static bool Matches(string path, long length, string sha)
        {
            if (!File.Exists(path)) return false;
            using var stream = File.OpenRead(path);
            return stream.Length == length && BundleStreamHash.Compute(stream, out _) == sha;
        }
        static async Task Copy(string source, string destination, CancellationToken token)
        {
            using var input = File.OpenRead(source);
            using var output = new FileStream(destination, FileMode.CreateNew, FileAccess.Write, FileShare.None);
            var buffer = new byte[4 * 1024 * 1024]; int count;
            while ((count = input.Read(buffer, 0, buffer.Length)) != 0)
            {
                token.ThrowIfCancellationRequested(); output.Write(buffer, 0, count); await Task.Yield();
            }
        }
        static void RejectLink(string path)
        { if ((File.GetAttributes(path) & FileAttributes.ReparsePoint) != 0) throw new IOException("Prepared cache cannot contain filesystem links."); }
        static void Prepare(string root)
        {
            if (Directory.Exists(root))
            {
                RejectLink(root);
                if (Directory.EnumerateFileSystemEntries(root).Any() &&
                    (!File.Exists(Path.Combine(root, ".owner")) || File.ReadAllText(Path.Combine(root, ".owner")) != Owner))
                    throw new IOException("Cache directory is not owned by this sample.");
            }
            else Directory.CreateDirectory(root);
            var marker = Path.Combine(root, ".owner");
            if (!File.Exists(marker))
            { using var stream = new FileStream(marker, FileMode.CreateNew, FileAccess.Write, FileShare.None); var bytes = Encoding.UTF8.GetBytes(Owner); stream.Write(bytes, 0, bytes.Length); }
        }
        static void Validate(string root)
        {
            foreach (var entry in Directory.GetFileSystemEntries(root))
            {
                RejectLink(entry); var name = Path.GetFileName(entry);
                if (!Directory.Exists(entry))
                { if (name != ".owner" && name != ".lock") throw new IOException("Unexpected cache file."); continue; }
                if (name != "objects" && name != "incoming") throw new IOException("Unexpected cache directory.");
                foreach (var file in Directory.GetFileSystemEntries(entry))
                {
                    RejectLink(file);
                    if (Directory.Exists(file) || !(name == "objects" ? ObjectName : PendingName).IsMatch(Path.GetFileName(file)))
                        throw new IOException("Unexpected cache contents.");
                }
            }
        }
    }
}
