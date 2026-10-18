using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;

namespace EmbeddingGemma.Validation
{
    /// <summary>Explicit Windows mixed-suite inputs. Parsing never downloads, creates or overwrites a file.</summary>
    public sealed class PlayerMixedConfiguration
    {
        public const string Fp32Descriptor = "40aefa5de06f575d31b7582c24314c1ab035da374116b3ce3036de0e919cd647";
        public const string Float16Descriptor = "8a6a90c5f656a00b024553a0412e5c4d32b4d2c9c7c5b3a0c2c146f557b4e27e";
        public string Bundle { get; private set; }
        public string Fixtures { get; private set; }
        public string Cache { get; private set; }
        public string Output { get; private set; }
        public string RunId { get; private set; }
        public string Phase { get; private set; }
        public string Precision { get; private set; }
        public string DescriptorSha256 { get; private set; }
        const string Prefix = "--embeddinggemma-mixed-";
        static bool Within(string path, string directory)
            => string.Equals(path, directory, StringComparison.OrdinalIgnoreCase)
                || path.StartsWith(directory.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar) + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase);
        public static PlayerMixedConfiguration Parse(string[] args)
        {
            var keys = new[] { "bundle", "fixtures", "cache", "output", "run-id", "phase", "precision", "descriptor" };
            if (args == null) throw new ArgumentException("Explicit mixed-suite arguments are required.");
            var values = new Dictionary<string, string>(StringComparer.Ordinal);
            for (var i = 0; i < args.Length; i++)
            {
                var arg = args[i];
                if (arg == null || !arg.StartsWith("--embeddinggemma-", StringComparison.Ordinal)) continue;
                if (!arg.StartsWith(Prefix, StringComparison.Ordinal)) throw new ArgumentException("Mixed and legacy validation arguments cannot be combined.");
                var key = arg.Substring(Prefix.Length);
                if (!keys.Contains(key) || values.ContainsKey(key) || i + 1 == args.Length || string.IsNullOrWhiteSpace(args[i + 1])
                    || args[i + 1].StartsWith("--embeddinggemma-", StringComparison.Ordinal))
                    throw new ArgumentException("Unique known mixed-suite arguments with explicit values are required.");
                values.Add(key, args[++i]);
            }
            if (values.Count != keys.Length || keys.Any(key => !values.ContainsKey(key)))
                throw new ArgumentException("All mixed-suite arguments are mandatory.");
            if (values["precision"] != "fp32" && values["precision"] != "float16"
                || values["phase"] != "cold" && values["phase"] != "warm"
                || !Regex.IsMatch(values["run-id"], "\\A[0-9a-f]{32}\\z"))
                throw new ArgumentException("Known precision, an explicit process phase and a new canonical run ID are required.");
            var pin = values["precision"] == "fp32" ? Fp32Descriptor : Float16Descriptor;
            if (values["descriptor"] != pin) throw new ArgumentException("The independent reviewed model descriptor pin differs.");
            var config = new PlayerMixedConfiguration { Bundle = Path.GetFullPath(values["bundle"]), Fixtures = Path.GetFullPath(values["fixtures"]),
                Cache = Path.GetFullPath(values["cache"]), Output = Path.GetFullPath(values["output"]), RunId = values["run-id"],
                Precision = values["precision"], Phase = values["phase"], DescriptorSha256 = pin };
            if (!config.Output.EndsWith(".json", StringComparison.OrdinalIgnoreCase)
                || Within(config.Cache, config.Bundle) || Within(config.Bundle, config.Cache)
                || Within(config.Cache, config.Fixtures) || Within(config.Fixtures, config.Cache)
                || Within(config.Output, config.Bundle) || Within(config.Output, config.Fixtures) || Within(config.Output, config.Cache))
                throw new ArgumentException("Writable cache and report paths must be isolated from each other and from original inputs.");
            return config;
        }
    }
}
