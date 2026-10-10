using System;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Security.Cryptography;

namespace EmbeddingGemma.Samples
{
    /// <summary>Editor-only complete SHA-256; accelerated hashing must preserve the digest.</summary>
    public static class TextSearchFileHash
    {
        public static string Compute(string path, Func<string, string> acceleratedOutput = null)
        {
            path = Path.GetFullPath(path);
            // Open first: missing files must fail, and Windows writers cannot replace these bytes
            // while the native reader hashes them. Every fallback also hashes the whole stream.
            using var stream = File.OpenRead(path);
            if (acceleratedOutput != null || stream.Length >= 1024 * 1024)
            {
                try
                {
                    var digest = ParseNativeOutput((acceleratedOutput ?? ReadNativeOutput)(path));
                    if (digest != null) return digest;
                }
                catch (IOException) { }
                catch (UnauthorizedAccessException) { }
                catch (Win32Exception) { }
                catch (InvalidOperationException) { }
                catch (NotSupportedException) { }
            }
            using var hash = SHA256.Create();
            return BitConverter.ToString(hash.ComputeHash(stream)).Replace("-", "").ToLowerInvariant();
        }

        public static string ParseNativeOutput(string output)
        {
            if (output == null) return null;
            string digest = null;
            foreach (var line in output.Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries))
            {
                var candidate = line.Trim();
                if (candidate.Length != 64) continue;
                var valid = true;
                foreach (var character in candidate)
                    if (!Uri.IsHexDigit(character)) { valid = false; break; }
                if (!valid) continue;
                if (digest != null) return null;
                digest = candidate.ToLowerInvariant();
            }
            return digest;
        }

        static string ReadNativeOutput(string path)
        {
#if UNITY_EDITOR_WIN
            using var process = new Process
            {
                StartInfo = new ProcessStartInfo
                {
                    FileName = Path.Combine(Environment.SystemDirectory, "certutil.exe"),
                    Arguments = "-hashfile \"" + path + "\" SHA256",
                    UseShellExecute = false,
                    CreateNoWindow = true,
                    WindowStyle = ProcessWindowStyle.Hidden,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true
                }
            };
            if (!process.Start()) return null;
            // Drain both pipes before waiting to avoid blocking on a full output buffer.
            var output = process.StandardOutput.ReadToEndAsync();
            var error = process.StandardError.ReadToEndAsync();
            if (!process.WaitForExit(60000))
            {
                process.Kill();
                process.WaitForExit();
                return null;
            }
            var text = output.GetAwaiter().GetResult();
            _ = error.GetAwaiter().GetResult();
            return process.ExitCode == 0 ? text : null;
#else
            return null;
#endif
        }
    }
}
