using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using UnityEngine.Networking;

namespace EmbeddingGemma.Validation
{
    /// <summary>Materialize pinned local fixtures. Original numerical and GPU gates remain with their existing loaders.</summary>
    public static class PlayerFixtureStager
    {
        sealed class Entry { public string Name, Sha; public long Bytes; }
        static string Sha(byte[] bytes)
        { using var hash = SHA256.Create(); return BitConverter.ToString(hash.ComputeHash(bytes)).Replace("-", "").ToLowerInvariant(); }
        static Entry[] Manifest(byte[] bytes, string digest)
        {
            if (bytes == null || bytes.Length == 0 || bytes.Length > 256 * 1024 ||
                !Regex.IsMatch(digest ?? "", "\\A[a-f0-9]{64}\\z") || Sha(bytes) != digest)
                throw new ArgumentException("An explicit bounded fixture manifest and its exact digest are required.");
            JObject manifest;
            try { manifest = JObject.Parse(new UTF8Encoding(false, true).GetString(bytes), new JsonLoadSettings { DuplicatePropertyNameHandling = DuplicatePropertyNameHandling.Error }); }
            catch (Exception error) when (error is JsonException || error is DecoderFallbackException)
            { throw new ArgumentException("Fixture manifest encoding or structure is invalid."); }
            if (manifest.Count != 2 || manifest["schema_version"]?.Type != JTokenType.Integer || (int?)manifest["schema_version"] != 1 ||
                !(manifest["files"] is JArray files) || files.Count == 0 || files.Count > 512)
                throw new ArgumentException("A complete versioned fixture inventory is required.");
            var entries = new List<Entry>(); var names = new HashSet<string>(StringComparer.OrdinalIgnoreCase); long total = 0;
            foreach (var value in files)
            {
                if (!(value is JObject row) || row.Count != 3 || row["path"]?.Type != JTokenType.String || row["bytes"]?.Type != JTokenType.Integer || row["sha256"]?.Type != JTokenType.String)
                    throw new ArgumentException("Every fixture requires one relative path, size and digest.");
                var name = (string)row["path"]; var size = (long)row["bytes"]; var sha = (string)row["sha256"];
                if (!Regex.IsMatch(name, "\\A[a-zA-Z0-9][a-zA-Z0-9._/-]*\\z") || name.Split('/').Any(part => part.Length == 0 || part == "." || part == "..") ||
                    !new[] { ".json", ".bin", ".png" }.Contains(Path.GetExtension(name).ToLowerInvariant()) || !names.Add(name) ||
                    size <= 0 || size > 64L * 1024 * 1024 || !Regex.IsMatch(sha, "\\A[a-f0-9]{64}\\z"))
                    throw new ArgumentException("Fixture paths, sizes and hashes must be safe, bounded and unique on every platform.");
                total += size; entries.Add(new Entry { Name = name, Bytes = size, Sha = sha });
            }
            if (total > 512L * 1024 * 1024 || names.Any(name => names.Any(other => other.StartsWith(name + "/", StringComparison.OrdinalIgnoreCase))))
                throw new ArgumentException("Fixture inventory exceeds its allowance or has conflicting file and directory names.");
            return entries.ToArray();
        }
        static void LocalSource(string source)
        {
            var local = source;
            if (local != null && local.StartsWith("jar:file://", StringComparison.Ordinal))
            {
                var end = local.IndexOf("!/", StringComparison.Ordinal);
                if (end < 0) throw new NotSupportedException("A local StreamingAssets source is required.");
                local = local.Substring(4, end - 4);
            }
            if (string.IsNullOrWhiteSpace(source) || source.IndexOfAny(new[] { '?', '#' }) >= 0 ||
                !Uri.TryCreate(local, UriKind.Absolute, out var uri) || !uri.IsFile || uri.UserInfo.Length != 0 ||
                uri.Host.Length != 0 && !string.Equals(uri.Host, "localhost", StringComparison.OrdinalIgnoreCase))
                throw new NotSupportedException("A local StreamingAssets source is required; network sources are not supported.");
        }
        public static IEnumerator Stage(string source, string destination, byte[] manifest, string manifestSha256, JObject report,
            Func<string, string, IEnumerator> transfer = null)
        {
            if (report == null || report.HasValues) throw new ArgumentException("A new empty stage report is required.");
            var entries = Manifest(manifest, manifestSha256); LocalSource(source);
            if (string.IsNullOrWhiteSpace(destination)) throw new ArgumentException("A new owned staging directory is required.");
            destination = Path.GetFullPath(destination);
            if (Directory.Exists(destination) || File.Exists(destination)) throw new IOException("A previous destination cannot be reused or overwritten.");
            report["scope"] = "local_streaming_fixture_materialization"; report["manifest_sha256"] = manifestSha256;
            report["success"] = false; report["phase"] = "staging"; report["original_fixture_validated"] = false; report["gpu_verified"] = false;
            report["transport"] = transfer == null ? "UnityWebRequest.DownloadHandlerFile" : "injected_transport";
            report["verified_files"] = 0; report["verified_bytes"] = 0; report["files"] = new JArray();
            bool Attempt(Action action)
            {
                try { action(); return true; }
                catch (Exception error) { report["error_type"] = error.GetType().Name; return false; }
            }
            IEnumerator routine = null; string partial = null; var complete = false;
            try
            {
                if (!Attempt(() => Directory.CreateDirectory(destination))) yield break;
                foreach (var entry in entries)
                {
                    var final = Path.Combine(destination, entry.Name.Replace('/', Path.DirectorySeparatorChar)); partial = final + ".part";
                    if (!Attempt(() => Directory.CreateDirectory(Path.GetDirectoryName(final)))) yield break;
                    if (!Attempt(() => { routine = (transfer ?? Download)(source.TrimEnd('/') + "/" + entry.Name, partial); if (routine == null) throw new InvalidOperationException("Transfer iterator is missing."); })) yield break;
                    while (true)
                    {
                        var moved = false; object current = null;
                        if (!Attempt(() => { moved = routine.MoveNext(); if (moved) current = routine.Current; })) yield break;
                        if (!moved) break;
                        yield return current;
                    }
                    var owned = routine; routine = null;
                    if (!Attempt(() => (owned as IDisposable)?.Dispose())) yield break;
                    string actual = null, backend = null;
                    if (!Attempt(() =>
                    {
                        using var stream = File.OpenRead(partial);
                        if (stream.Length != entry.Bytes) throw new InvalidDataException("Transferred fixture size differs.");
                        actual = PlayerFileHash.Compute(stream, out backend);
                        if (actual != entry.Sha) throw new InvalidDataException("Transferred fixture digest differs.");
                    })) yield break;
                    if (!Attempt(() => File.Move(partial, final))) yield break;
                    partial = null;
                    ((JArray)report["files"]).Add(new JObject { ["path"] = entry.Name, ["bytes"] = entry.Bytes, ["sha256"] = actual, ["hash_backend"] = backend });
                    report["verified_files"] = (int)report["verified_files"] + 1; report["verified_bytes"] = (long)report["verified_bytes"] + entry.Bytes;
                }
                complete = true;
            }
            finally
            {
                if (routine != null) Attempt(() => (routine as IDisposable)?.Dispose());
                if (partial != null) Attempt(() => { if (File.Exists(partial)) File.Delete(partial); });
                report["success"] = complete && report["error_type"] == null;
                report["phase"] = (bool)report["success"] ? "complete" : report["error_type"] == null ? "cancelled" : "failed";
                report["original_fixture_validated"] = false; report["gpu_verified"] = false;
            }
        }
        static IEnumerator Download(string source, string output)
        {
            using var request = new UnityWebRequest(source, UnityWebRequest.kHttpVerbGET);
            request.downloadHandler = new DownloadHandlerFile(output) { removeFileOnAbort = true }; request.timeout = 300;
            var operation = request.SendWebRequest();
            while (!operation.isDone) yield return null;
            if (request.result != UnityWebRequest.Result.Success) throw new IOException("Local fixture transfer did not complete.");
        }
    }
}
