using System;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using Newtonsoft.Json.Linq;
using Unity.InferenceEngine;

namespace EmbeddingGemma.Validation.Tests
{
    /// <summary>Preparation only: saves qualified graphs/coefficients without executing or changing their arithmetic.</summary>
    public sealed class GpuMediaBundleWriter
    {
        readonly string directory;
        JObject descriptor;
        const string StateName = ".bundle-work.json";

        public GpuMediaBundleWriter(string directory, string sourceCommit)
        {
            if (string.IsNullOrWhiteSpace(directory) || !Regex.IsMatch(sourceCommit ?? "", "\\A[a-f0-9]{40}\\z"))
                throw new ArgumentException("A preparation directory and fixed source commit are required.");
            this.directory = Path.GetFullPath(directory);
            var state = Path.Combine(this.directory, StateName);
            if (File.Exists(state))
            {
                descriptor = JObject.Parse(File.ReadAllText(state));
                if ((string)descriptor["source_commit"] != sourceCommit || (string)descriptor["model_revision"] != GpuMediaBundle.PinnedRevision)
                    throw new InvalidDataException("Preparation producer or pinned model revision differs.");
            }
            else
            {
                if (Directory.Exists(this.directory) && Directory.EnumerateFileSystemEntries(this.directory).Any())
                    throw new IOException("Preparation destination is not empty and has no owned work record.");
                Directory.CreateDirectory(this.directory);
                descriptor = new JObject { ["schema_version"] = 1, ["backend"] = "GPUCompute", ["sentis_version"] = "2.6.1",
                    ["model_revision"] = GpuMediaBundle.PinnedRevision, ["source_commit"] = sourceCommit,
                    ["prepared_only"] = true, ["inference_executed"] = false, ["files"] = new JObject(), ["modules"] = new JObject() };
                SaveState();
            }
        }
        static string Hash(string path) { using var stream = File.OpenRead(path); return PlayerFileHash.Compute(stream, out _); }
        string Pending(string extension) => Path.Combine(directory, ".store-" + Guid.NewGuid().ToString("N") + extension);
        string Store(string pending, string extension)
        {
            var bytes = new FileInfo(pending).Length; var sha = Hash(pending); var relative = "objects/" + sha + extension;
            var destination = Path.Combine(directory, relative); Directory.CreateDirectory(Path.GetDirectoryName(destination));
            if (File.Exists(destination))
            {
                if (new FileInfo(destination).Length != bytes || Hash(destination) != sha) throw new InvalidDataException("Existing prepared object is corrupt: " + relative);
            }
            else File.Move(pending, destination);
            descriptor["files"][relative] = new JObject { ["bytes"] = bytes, ["sha256"] = sha }; SaveState(); return relative;
        }
        public string StoreModel(Model model)
        {
            if (model == null) throw new ArgumentNullException(nameof(model));
            var pending = Pending(".sentis");
            try { ModelWriter.Save(pending, model); return Store(pending, ".sentis"); }
            finally { if (File.Exists(pending)) File.Delete(pending); }
        }
        public string StoreFile(string source)
        {
            var extension = Path.GetExtension(source); if (string.IsNullOrEmpty(extension)) extension = ".bin";
            var pending = Pending(extension);
            try { File.Copy(source, pending); return Store(pending, extension); }
            finally { if (File.Exists(pending)) File.Delete(pending); }
        }
        public string StoreWeights(float[] values)
        {
            if (values == null) throw new ArgumentNullException(nameof(values));
            var bytes = new byte[checked(values.Length * 4)]; Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
            if (!BitConverter.IsLittleEndian) for (var i = 0; i < bytes.Length; i += 4) Array.Reverse(bytes, i, 4);
            var pending = Pending(".f32");
            try { File.WriteAllBytes(pending, bytes); return Store(pending, ".f32"); }
            finally { if (File.Exists(pending)) File.Delete(pending); }
        }
        public void CommitModule(string name, JObject module)
        {
            var candidate = (JObject)descriptor.DeepClone();
            var previous = candidate["modules"]?[name];
            if (previous != null && !JToken.DeepEquals(previous, module)) throw new InvalidDataException("An existing prepared module cannot be silently replaced.");
            candidate["modules"][name] = module?.DeepClone();
            GpuMediaBundle.Parse(candidate.ToString(), directory);
            descriptor = candidate; SaveState(); AtomicWrite(Path.Combine(directory, "bundle.json"), descriptor.ToString());
        }
        void SaveState() => AtomicWrite(Path.Combine(directory, StateName), descriptor.ToString());
        static void AtomicWrite(string path, string text)
        {
            var pending = path + "." + Guid.NewGuid().ToString("N") + ".pending";
            try
            {
                File.WriteAllText(pending, text + "\n");
                if (File.Exists(path)) File.Replace(pending, path, null); else File.Move(pending, path);
            }
            finally { if (File.Exists(pending)) File.Delete(pending); }
        }
    }
}
