using System;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace EmbeddingGemma.Validation
{
    /// <summary>Explicit device-local run settings. Parsing performs no filesystem writes or downloads.</summary>
    public sealed class PlayerDeviceMixedConfiguration
    {
        public const string DirectoryName = "EmbeddingGemmaDeviceValidation";
        public string Root { get; private set; }
        public string RunId { get; private set; }
        public string CacheId { get; private set; }
        public string Precision { get; private set; }
        public string Phase { get; private set; }
        public string DescriptorSha256 { get; private set; }
        public string FixtureManifestSha256 { get; private set; }
        public string FixtureManifest => Path.Combine(Root, "fixture-manifest.json");
        public string StartupReport => Path.Combine(Root, "reports", RunId + "-device.json");
        public static string FixtureSource(string streamingAssetsPath)
        {
            if (string.IsNullOrWhiteSpace(streamingAssetsPath) || streamingAssetsPath.IndexOfAny(new[] { '?', '#', '\r', '\n' }) >= 0)
                throw new ArgumentException("An unambiguous local StreamingAssets root is required.");
            var source = streamingAssetsPath.TrimEnd('/'); var local = source;
            var jar = source.StartsWith("jar:file://", StringComparison.Ordinal);
            if (jar)
            {
                var boundary = source.IndexOf("!/", StringComparison.Ordinal);
                if (boundary < 0 || boundary + 2 == source.Length) throw new NotSupportedException("A local APK StreamingAssets root is required.");
                local = source.Substring(4, boundary - 4);
            }
            else if (!source.Contains("://"))
            {
                if (!Path.IsPathRooted(source)) throw new NotSupportedException("A local absolute StreamingAssets root is required.");
                local = Path.GetFullPath(source);
            }
            if (!Uri.TryCreate(local, UriKind.Absolute, out var uri) || !uri.IsFile || uri.UserInfo.Length != 0
                || uri.Host.Length != 0 && !string.Equals(uri.Host, "localhost", StringComparison.OrdinalIgnoreCase))
                throw new NotSupportedException("StreamingAssets transfer cannot use a remote source.");
            return (jar ? source : uri.AbsoluteUri.TrimEnd('/')) + "/" + DirectoryName + "/fixtures";
        }
        public static PlayerDeviceMixedConfiguration Parse(string json, string persistentDataPath)
        {
            if (string.IsNullOrWhiteSpace(persistentDataPath) || string.IsNullOrWhiteSpace(json) || json.Length > 256 * 1024)
                throw new ArgumentException("Bounded explicit device settings and a persistent data root are required.");
            JObject data;
            try { data = JObject.Parse(json, new JsonLoadSettings { DuplicatePropertyNameHandling = DuplicatePropertyNameHandling.Error }); }
            catch (JsonException) { throw new ArgumentException("Device run settings are invalid."); }
            var keys = new[] { "schema_version", "precision", "phase", "run_id", "cache_id", "bundle_descriptor_sha256", "fixture_manifest_sha256" };
            if (data.Count != keys.Length || keys.Any(key => data[key] == null) || data["schema_version"].Type != JTokenType.Integer || (long)data["schema_version"] != 1
                || keys.Skip(1).Any(key => data[key].Type != JTokenType.String))
                throw new ArgumentException("Every versioned device setting must be explicit and correctly typed.");
            var precision = (string)data["precision"]; var phase = (string)data["phase"];
            if (precision != "fp32" && precision != "float16" || phase != "cold" && phase != "warm"
                || !Regex.IsMatch((string)data["run_id"], "\\A[a-f0-9]{32}\\z") || !Regex.IsMatch((string)data["cache_id"], "\\A[a-f0-9]{32}\\z")
                || !Regex.IsMatch((string)data["fixture_manifest_sha256"], "\\A[a-f0-9]{64}\\z"))
                throw new ArgumentException("Known precision, process phase, canonical IDs and an exact fixture manifest pin are required.");
            var pin = precision == "fp32" ? PlayerMixedConfiguration.Fp32Descriptor : PlayerMixedConfiguration.Float16Descriptor;
            if ((string)data["bundle_descriptor_sha256"] != pin) throw new ArgumentException("The reviewed model descriptor pin differs.");
            return new PlayerDeviceMixedConfiguration { Root = Path.Combine(Path.GetFullPath(persistentDataPath), DirectoryName), RunId = (string)data["run_id"],
                CacheId = (string)data["cache_id"], Precision = precision, Phase = phase, DescriptorSha256 = pin, FixtureManifestSha256 = (string)data["fixture_manifest_sha256"] };
        }
        public string[] MixedArguments() => new[] {
            "--embeddinggemma-mixed-bundle", Path.Combine(Root, "bundles", Precision),
            "--embeddinggemma-mixed-fixtures", Path.Combine(Root, "runs", RunId, "fixtures"),
            "--embeddinggemma-mixed-cache", Path.Combine(Root, "cache", CacheId),
            "--embeddinggemma-mixed-output", Path.Combine(Root, "reports", RunId + ".json"),
            "--embeddinggemma-mixed-run-id", RunId, "--embeddinggemma-mixed-phase", Phase,
            "--embeddinggemma-mixed-precision", Precision, "--embeddinggemma-mixed-descriptor", DescriptorSha256 };
    }
}
