using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Security.Cryptography;

namespace EmbeddingGemma.Samples
{
    internal static class TextSearchStreamHash
    {
        public static string Compute(Stream stream, out string backend, Func<Stream, string> accelerated = null)
        {
            if (stream == null || !stream.CanRead || !stream.CanSeek) throw new ArgumentException("A readable seekable audited stream is required.");
            backend = "dotnet_sha256";
            stream.Position = 0;
            try
            {
                var digest = (accelerated ?? Native)(stream);
                if (digest != null && digest.Length == 64)
                {
                    var valid = true; foreach (var character in digest) if (!Uri.IsHexDigit(character)) { valid = false; break; }
                    if (valid) { backend = accelerated == null ? "windows_cng" : "injected_accelerated"; return digest.ToLowerInvariant(); }
                }
            }
            catch (DllNotFoundException) { }
            catch (EntryPointNotFoundException) { }
            catch (BadImageFormatException) { }
            catch (NotSupportedException) { }
            catch (CryptographicException) { }
            // A failed native implementation may have consumed part of the stream.
            // Retry the complete contents using the same caller-owned read-only handle.
            stream.Position = 0;
            using var hash = SHA256.Create();
            return Hex(hash.ComputeHash(stream));
        }
        static string Hex(byte[] bytes) => BitConverter.ToString(bytes).Replace("-", "").ToLowerInvariant();
        static string Native(Stream stream)
        {
#if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN
            IntPtr algorithm = IntPtr.Zero, hash = IntPtr.Zero;
            try
            {
                Check(BCryptOpenAlgorithmProvider(out algorithm, "SHA256", null, 0));
                // Windows 7+ can own the hash-object allocation and release it with DestroyHash.
                Check(BCryptCreateHash(algorithm, out hash, IntPtr.Zero, 0, IntPtr.Zero, 0, 0));
                var buffer = new byte[1024 * 1024]; int read;
                while ((read = stream.Read(buffer, 0, buffer.Length)) > 0) Check(BCryptHashData(hash, buffer, (uint)read, 0));
                var digest = new byte[32]; Check(BCryptFinishHash(hash, digest, (uint)digest.Length, 0));
                return Hex(digest);
            }
            finally
            {
                if (hash != IntPtr.Zero) BCryptDestroyHash(hash);
                if (algorithm != IntPtr.Zero) BCryptCloseAlgorithmProvider(algorithm, 0);
            }
#else
            return null;
#endif
        }
#if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN
        static void Check(int status) { if (status != 0) throw new CryptographicException("Windows CNG SHA-256 status 0x" + status.ToString("X8")); }
        [DllImport("bcrypt.dll", CharSet = CharSet.Unicode)] static extern int BCryptOpenAlgorithmProvider(out IntPtr algorithm, string name, string implementation, uint flags);
        [DllImport("bcrypt.dll")] static extern int BCryptCreateHash(IntPtr algorithm, out IntPtr hash, IntPtr hashObject, uint objectBytes, IntPtr secret, uint secretBytes, uint flags);
        [DllImport("bcrypt.dll")] static extern int BCryptHashData(IntPtr hash, [In] byte[] buffer, uint bytes, uint flags);
        [DllImport("bcrypt.dll")] static extern int BCryptFinishHash(IntPtr hash, [Out] byte[] digest, uint bytes, uint flags);
        [DllImport("bcrypt.dll")] static extern int BCryptDestroyHash(IntPtr hash);
        [DllImport("bcrypt.dll")] static extern int BCryptCloseAlgorithmProvider(IntPtr algorithm, uint flags);
#endif
    }
}
