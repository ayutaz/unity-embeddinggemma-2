using System;
using System.IO;
using Newtonsoft.Json.Linq;

namespace EmbeddingGemma.Validation.Tests
{
    // Preparation is pinned separately from the original golden and from GPU
    // acceptance. Metadata checking never substitutes for the byte audit.
    public static class CandidateAudioPlanGuard
    {
        public const string Producer="b6ede71cbf4170ae10a61344455dbab56a8231cf";
        const string SeedSource="0cee4c7b77c89410895d52e4420816a5e66a8845";
        static (string signal,string graph,string plan,string receipt) Pins(string profile)
        {
            if(profile=="frames-1119-valid-1119")return (
                "silence","525871a8dbccbbd75b5c521ab959588323938e58fac361d69035eb80a61d2095",
                "d8440b4ffc7f8b37e11bee1da1da0b7cadb44e708a1790aab247e1fca40577f9",
                "bc0ce35dff656413d710c282d3b12bf893aa97721adb33de2000c44b2e6e58ac");
            if(profile=="frames-1119-valid-839")return (
                "padded_noise","87a2a058da07b0ce4338769723d0dbb572a3048ae782a8db79a37e952319ee99",
                "4f4a571a74489602e97913f326aafef4b72db15202b41c6fd1e7b21dcfb80c70",
                "6456e11b14da939eb1515af122f1b18ade0f69e854963428214da7482eccf8ca");
            throw new InvalidDataException("Only the two fixed candidate profiles are accepted.");
        }
        static void Equal(JObject value,string name,object expected)
        {
            if(value==null || !JToken.DeepEquals(value[name],JToken.FromObject(expected)))
                throw new InvalidDataException("Candidate preparation provenance or completion differs.");
        }
        public static void Check(JObject plan,JObject receipt,string profile)
        {
            var pin=Pins(profile);
            Equal(plan,"producer",Producer);Equal(plan,"profile",profile);Equal(plan,"source_graph_sha256",pin.graph);
            Equal(plan,"seed_source",SeedSource);Equal(plan,"seed_run",39117103059L);
            Equal(plan,"original_initializer_bytes_preserved",true);Equal(plan,"source_initializer_bytes_preserved",true);
            Equal(plan,"original_initializer_count",587);Equal(plan,"original_references_changed",false);
            Equal(receipt,"success",true);Equal(receipt,"source",Producer);Equal(receipt,"case_id",pin.signal);
            Equal(receipt,"seed_source",SeedSource);Equal(receipt,"seed_run",39117103059L);Equal(receipt,"plan_sha256",pin.plan);
            Equal(receipt,"operation_count",267);Equal(receipt,"segment_count",268);
            foreach(var name in new[]{"numeric_acceptance","gpu_validated","generation_qualified","original_golden_changed"})
            {
                Equal(plan,name,false);Equal(receipt,name,false);
            }
            if(!(plan["operations"] is JArray operations) || operations.Count!=267
                || !(plan["segments"] is JArray segments) || segments.Count!=268)
                throw new InvalidDataException("The complete candidate partition chain is required.");
        }
        public static string Hash(string path)
        {
            using var stream=File.OpenRead(path);return PlayerFileHash.Compute(stream,out _);
        }
        static string Leaf(string root,string name)
        {
            if(string.IsNullOrEmpty(name) || name.IndexOfAny(new[]{'/','\\',':'})>=0 || name=="." || name=="..")
                throw new InvalidDataException("Candidate files must have simple relative names.");
            var path=Path.Combine(root,name);
            if(!File.Exists(path) || (File.GetAttributes(path)&FileAttributes.ReparsePoint)!=0)
                throw new InvalidDataException("A regular candidate file is required.");
            return path;
        }
        public static JObject Load(string root,string profile)
        {
            var pin=Pins(profile);var planPath=Leaf(root,"partition-plan.json");var receiptPath=Leaf(root,"audio-gpu-refinement.json");
            if(Hash(planPath)!=pin.plan || Hash(receiptPath)!=pin.receipt)
                throw new InvalidDataException("Pinned preparation plan or receipt bytes differ.");
            var plan=JObject.Parse(File.ReadAllText(planPath));var receipt=JObject.Parse(File.ReadAllText(receiptPath));Check(plan,receipt,profile);
            if(!(plan["files"] is JObject files) || files.Count!=415)
                throw new InvalidDataException("All pinned model and coefficient files are required.");
            foreach(var entry in files)
            {
                var path=Leaf(root,entry.Key);var size=entry.Value.Value<long>("bytes");
                if(size<=0 || new FileInfo(path).Length!=size || Hash(path)!=entry.Value.Value<string>("sha256"))
                    throw new InvalidDataException("Candidate model or coefficient bytes differ.");
            }
            return plan;
        }
    }
}
