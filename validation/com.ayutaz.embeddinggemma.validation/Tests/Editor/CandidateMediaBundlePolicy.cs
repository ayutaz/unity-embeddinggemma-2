using System;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using Newtonsoft.Json.Linq;

namespace EmbeddingGemma.Validation.Tests
{
    // This is a separate gate; the retained generation's default pins stay unchanged.
    public static class CandidateMediaBundlePolicy
    {
        static bool Half(string precision)
        {
            if (precision=="fp32") return false;
            if (precision=="float16-weights") return true;
            throw new ArgumentOutOfRangeException(nameof(precision), "Choose an explicit reviewed precision.");
        }
        public static string DescriptorSha256(string precision) => Half(precision)
            ? "8a6a90c5f656a00b024553a0412e5c4d32b4d2c9c7c5b3a0c2c146f557b4e27e"
            : "40aefa5de06f575d31b7582c24314c1ab035da374116b3ce3036de0e919cd647";
        public static double MinimumCosine(string precision)
        { Half(precision); return GpuMediaExecutionPolicy.MinimumCosine(precision); }
        public static void VerifyCases(string cases)
        {
            var expected=new[]{"image-square:square-0","image-landscape:landscape-0","image-portrait:portrait-0","audio-1119:silence","audio-839:padded_noise"};
            var actual=(cases??"").Split(',');
            if(actual.Length!=expected.Length || !actual.OrderBy(v=>v,StringComparer.Ordinal).SequenceEqual(expected.OrderBy(v=>v,StringComparer.Ordinal)))
                throw new ArgumentException("The candidate sample gate requires all three image profiles and both original audio profiles.",nameof(cases));
        }
        public static void Verify(string directory, string precision)
        {
            var half=Half(precision);var path=Path.Combine(directory,"bundle.json");
            using (var stream=File.OpenRead(path))
            using (var hash=SHA256.Create())
                if (BitConverter.ToString(hash.ComputeHash(stream)).Replace("-","").ToLowerInvariant()!=DescriptorSha256(precision))
                    throw new InvalidDataException("Composed candidate descriptor differs from the reviewed precision.");
            var data=JObject.Parse(File.ReadAllText(path));var distribution=data["distribution"];
            var expectedBase=half?"1000f91d47c7def916ace1c7affb5fd01884aab9f3cf064ea62cc48e4d057292":"8363f9a63a95b307fbbb835f3551a41f6fa5a1d21aca3017670e6f678647e955";
            var expectedAudio=half?"aaabc35d9ceeb0c0a38e80a439730766468b1391f6991b07f8c1c8a44ec741ea":"264652e72f7e35fe15f7d0f461c7be521d6a2a0be59802b5a9115ff7dc40d2d7";
            var modules=new[]{"text","image-square","image-landscape","image-portrait","audio-1119","audio-839"};
            if (data.Value<string>("backend")!="GPUCompute" || data.Value<string>("sentis_version")!="2.6.1"
                || data.Value<string>("model_revision")!="914f7f89142e33e77833254d9c9b90c3cef7303b"
                || data.Value<string>("source_commit")!="567befbd64287a706edaac59c978927e0277188a"
                || data.Value<string>("object_layout")!="flat" || distribution?.Value<string>("precision")!=precision
                || distribution["source_descriptor_sha256"]?.Value<string>("base")!=expectedBase
                || distribution["source_descriptor_sha256"]?.Value<string>("audio")!=expectedAudio
                || !((JObject)data["modules"]).Properties().Select(p=>p.Name).OrderBy(n=>n,StringComparer.Ordinal)
                    .SequenceEqual(modules.OrderBy(n=>n,StringComparer.Ordinal)) || ((JObject)data["files"]).Count!=820)
                throw new InvalidDataException("Composed candidate provenance or module coverage differs.");
        }
        internal static void VerifyCurrent(string directory) => Verify(directory,Environment.GetEnvironmentVariable("EG_MEDIA_PRECISION"));
    }
}
