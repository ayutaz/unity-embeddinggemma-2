using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;

namespace EmbeddingGemma.Validation
{
    public sealed class PlayerMediaConfiguration
    {
        public string Bundle, Fixtures, Module, Precision, DescriptorSha256, Output, RunId;
        public static PlayerMediaConfiguration Parse(string[] args)
        {
            const string prefix = "--embeddinggemma-media-";
            var required = new[] { "bundle", "fixtures", "module", "precision", "descriptor", "output", "run-id" };
            var values = new Dictionary<string, string>(StringComparer.Ordinal);
            args ??= Array.Empty<string>();
            for (var index = 0; index < args.Length; index++)
            {
                if (!args[index].StartsWith(prefix, StringComparison.Ordinal)) continue;
                var key = args[index].Substring(prefix.Length);
                if (!required.Contains(key) || values.ContainsKey(key) || index + 1 >= args.Length ||
                    string.IsNullOrWhiteSpace(args[index + 1]) || args[index + 1].StartsWith("-", StringComparison.Ordinal))
                    throw new ArgumentException("Invalid or duplicate media validation argument.");
                values.Add(key, args[++index]);
            }
            if (required.Any(key => !values.ContainsKey(key)) ||
                !new[] { "image-square", "image-landscape", "image-portrait", "audio-1119", "audio-839" }.Contains(values["module"]) ||
                values["precision"] != "fp32" && values["precision"] != "float16" ||
                !Regex.IsMatch(values["descriptor"], "\\A[a-f0-9]{64}\\z") || !Regex.IsMatch(values["run-id"], "\\A[a-f0-9]{32}\\z"))
                throw new ArgumentException("An explicit candidate, fixed module, precision and unique run ID are required.");
            return new PlayerMediaConfiguration { Bundle = Path.GetFullPath(values["bundle"]), Fixtures = Path.GetFullPath(values["fixtures"]),
                Output = Path.GetFullPath(values["output"]), Module = values["module"], Precision = values["precision"],
                DescriptorSha256 = values["descriptor"], RunId = values["run-id"] };
        }
    }
}
