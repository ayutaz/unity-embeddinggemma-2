using System;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace EmbeddingGemma.Validation
{
    /// <summary>Checks cold/warm identity links; JSON flags alone never certify GPU execution.</summary>
    public static class PlayerMixedCachePolicy
    {
        public const string QualificationName = "mixed-cold-complete.json";
        static bool Hex(JToken value, int length) => value?.Type == JTokenType.String && Regex.IsMatch((string)value, "\\A[0-9a-f]{" + length + "}\\z");
        static bool True(JToken value) => value?.Type == JTokenType.Boolean && (bool)value;
        public static JObject Validate(PlayerMixedConfiguration config, PlayerBuildInfo build)
        {
            if (config == null || build == null || !Regex.IsMatch(build.codeCommit ?? "", "\\A[0-9a-f]{40}\\z"))
                throw new ArgumentException("A fixed build and explicit mixed-suite configuration are required.");
            var exists = Directory.Exists(config.Cache) || File.Exists(config.Cache);
            var result = new JObject { ["phase"] = config.Phase, ["cache_existed"] = exists,
                ["cache_transfers_must_be_zero"] = config.Phase == "warm", ["gpu_execution_certified"] = false };
            if (config.Phase == "cold")
            {
                if (exists) throw new InvalidOperationException("Cold execution requires a new model-cache directory.");
                return result;
            }
            if (!Directory.Exists(config.Cache)) throw new InvalidOperationException("Warm execution requires the completed cold cache.");
            var file = Path.Combine(config.Cache, QualificationName); JObject value; string digest;
            using (var stream = File.OpenRead(file))
            {
                if (stream.Length <= 0 || stream.Length > 1024 * 1024) throw new InvalidOperationException("Cold qualification exceeds its bounded allowance.");
                digest = PlayerFileHash.Compute(stream, out _); stream.Position = 0;
                try { using var reader = new StreamReader(stream); value = JObject.Parse(reader.ReadToEnd()); }
                catch (JsonException error) { throw new InvalidOperationException("Cold qualification metadata is invalid.", error); }
            }
            var counts = new[] { ("base_media_vectors", 36), ("warm_media_vectors", 45), ("mixed_documents", 18), ("mixed_queries", 4), ("mixed_rank_and_score_pairs", 72) };
            if ((int?)value["schema_version"] != 1 || !True(value["success"]) || !True(value["gpu_verified"])
                || (string)value["phase"] != "cold" || (string)value["source_commit"] != build.codeCommit
                || (string)value["precision"] != config.Precision || (string)value["bundle_descriptor_sha256"] != config.DescriptorSha256
                || !Hex(value["run_id"], 32) || (string)value["run_id"] == config.RunId || !Hex(value["cold_report_sha256"], 64)
                || counts.Any(p => value[p.Item1]?.Type != JTokenType.Integer || (long)value[p.Item1] != p.Item2))
                throw new InvalidOperationException("Warm execution requires complete matching cold evidence and a distinct run ID.");
            result["cold_run_id"] = value["run_id"].DeepClone(); result["cold_report_sha256"] = value["cold_report_sha256"].DeepClone();
            result["cold_qualification_sha256"] = digest;
            result["independent_process_receipt_verification_required"] = true; return result;
        }
    }
}
