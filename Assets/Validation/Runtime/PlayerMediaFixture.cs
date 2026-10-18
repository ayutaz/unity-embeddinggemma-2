using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using Newtonsoft.Json.Linq;
using UnityEngine;

namespace EmbeddingGemma.Validation
{
    public sealed class PlayerMediaFixture : IDisposable
    {
        public IReadOnlyList<PlayerMediaCase> Cases { get; private set; }
        public JObject Evidence { get; private set; }
        readonly List<Texture2D> textures = new();
        public static string VerifyFile(string root, string relative, long size, string sha256)
        {
            if (string.IsNullOrWhiteSpace(root) || string.IsNullOrEmpty(relative) || Path.IsPathRooted(relative) ||
                relative.IndexOfAny(new[] { '\\', ':' }) >= 0 || relative.Split('/').Any(part => part.Length == 0 || part == "." || part == "..") ||
                size < 0 || !Regex.IsMatch(sha256 ?? "", "\\A[a-f0-9]{64}\\z"))
                throw new ArgumentException("A canonical relative fixture and fixed size/digest are required.");
            var path = Path.GetFullPath(Path.Combine(root, relative));
            using var stream = File.OpenRead(path);
            if (stream.Length != size || PlayerFileHash.Compute(stream, out _) != sha256)
                throw new InvalidDataException("Fixture bytes differ from their independent reference.");
            return path;
        }
        static string[] Ids(bool audio) => audio ? new[] { "silence", "impulse", "sine", "nyquist", "noise", "padded_noise" }
            : new[] { "square-0", "square-1", "landscape-0", "landscape-1", "portrait-0", "portrait-1" };
        public static JToken[] SelectCases(JObject reference, string module)
        {
            var role = (string)reference?["metadata"]?["role"];
            if (role != "document" && role != "query" && role != "raw") throw new ArgumentException("An explicit original role is required.");
            return SelectCasesForRole(reference, module, role);
        }
        static JToken[] SelectCasesForRole(JObject reference, string module, string role)
        {
            if (!new[] { "image-square", "image-landscape", "image-portrait", "audio-1119", "audio-839" }.Contains(module))
                throw new ArgumentException("Unknown fixed media module.");
            var audio = module.StartsWith("audio-", StringComparison.Ordinal); var ids = Ids(audio);
            if (!(reference?["cases"] is JArray rows) || rows.Count != 6 ||
                !rows.Select(row => (string)row["id"]).OrderBy(id => id, StringComparer.Ordinal).SequenceEqual(ids.OrderBy(id => id, StringComparer.Ordinal)))
                throw new ArgumentException("All six unique original cases are required.");
            foreach (var row in rows)
            {
                var id = (string)row["id"];
                var profile = audio ? "frames-1119-valid-" + (id == "padded_noise" ? "839" : "1119") : id.Split('-')[0];
                if ((string)row["role"] != role || (string)row["profile"] != profile || !(row["embedding"] is JArray embedding))
                    throw new ArgumentException("Matching original role, profile and embedding are required.");
                var vector = embedding.Values<float>().ToArray(); PlayerValidation.CheckEmbedding(vector, vector, "fp32");
            }
            var selected = audio ? "frames-1119-valid-" + module.Substring(6) : module.Substring(6);
            return ids.Select(id => rows.Single(row => (string)row["id"] == id)).Where(row => (string)row["profile"] == selected).ToArray();
        }
        public static PlayerMediaFixture Load(string root, string module, JObject metadata)
        {
            var fixture = new PlayerMediaFixture { Evidence = new JObject { ["files"] = new JObject(), ["roles"] = new JArray("document", "query", "raw") } };
            string Verified(string name, long bytes, string sha)
            {
                var path = VerifyFile(root, name, bytes, sha);
                ((JObject)fixture.Evidence["files"])[name] = new JObject { ["bytes"] = bytes, ["sha256"] = sha }; return path;
            }
            JObject Pinned(string name, string sha, bool parse = true)
            {
                var bytes = new FileInfo(Path.Combine(root, name)).Length;
                if (bytes > (parse ? 4 : 64) * 1024 * 1024) throw new InvalidDataException("Reference JSON exceeds its bounded input allowance.");
                var path = Verified(name, bytes, sha);
                return parse ? JObject.Parse(File.ReadAllText(path)) : null;
            }
            try
            {
                var audio = module != null && module.StartsWith("audio-", StringComparison.Ordinal);
                var reference = Pinned(audio ? "audio-embedding-reference.json" : "image-embedding-reference.json", audio
                    ? "281d12573b431667e8841033db1c6a10dce4382ef98c9a5345c18044a1194001"
                    : "90692ad38c6d7d844dc6b6e79c7d5cfa719ea59866946dbf94a40bca10ac9eac");
                var rows = new List<(JToken Row, TextRole Role)>();
                foreach (var role in new[] { TextRole.Document, TextRole.Query, TextRole.Raw })
                {
                    var name = role.ToString().ToLowerInvariant();
                    var roleReference = role == TextRole.Document ? reference : Pinned("roles/" + (audio ? "audio" : "image") + "/" + name + "/" +
                        (audio ? "audio-embedding-reference.json" : "image-embedding-reference.json"), audio
                        ? role == TextRole.Query ? "515a32124c94437b372f81b3be4eb605717b1c4ae9293bf2c3b851450f35565b" : "ea4298313a536aa3b0c1f5eac7934240abaf3cb68170be36deb64c214fbb92a9"
                        : role == TextRole.Query ? "dc4ad0a8e43e391a251e7ff3fb4f569b6bddb7442852f8721b1b7657ea91f82d" : "120faf6aea190466b625f7c606a4c14e38315101a265c7dd457af13694512e9c");
                    var declaredRole = (string)roleReference["metadata"]?["role"];
                    // The original audio Document JSON predates metadata.role; its exact
                    // pinned hash and every row's explicit role still establish this role.
                    if (declaredRole != name && !(audio && role == TextRole.Document && declaredRole == null))
                        throw new InvalidDataException("Pinned reference role differs.");
                    rows.AddRange(SelectCasesForRole(roleReference, module, name).Select(row => (row, role)));
                }
                if ((string)metadata?["kind"] != (audio ? "audio" : "image")) throw new ArgumentException("Module kind differs from its fixed fixture.");
                var result = new List<PlayerMediaCase>();
                var inputs = new Dictionary<string, object>(StringComparer.Ordinal);
                if (audio)
                {
                    var frames = module == "audio-1119" ? 1119 : 839;
                    if ((int?)metadata["valid_audio_frames"] != frames) throw new ArgumentException("Audio module profile differs.");
                    var audit = Pinned("audio/audio-validation.json", "4e325f708e8d103a9de358a411380ddbb51cfe953984e9dcb73e291d919630fa");
                    byte[] Read(string name) => File.ReadAllBytes(Verified("audio/" + name, (long)audit["files"][name]["bytes"], (string)audit["files"][name]["sha256"]));
                    foreach (var entry in rows)
                    {
                        var row = entry.Row;
                        var id = (string)row["id"];
                        if (!inputs.TryGetValue(id, out var input)) inputs[id] = input = PlayerMediaInput.DecodePcm(Read(id + "-pcm-fp32.bin"), Read(id + "-samples-int32.bin"), frames);
                        result.Add(new PlayerMediaCase { Id = id, Role = entry.Role, Expected = row["embedding"].Values<float>().ToArray(), Input = input });
                    }
                }
                else
                {
                    Pinned("vision/vision-reference.json", "3dfccd1a487ce04df3f8d640483ca045f7a7a56231bc1096ad0f845c923a29c6", parse: false);
                    var audit = Pinned("vision/vision-validation.json", "ab1aa775b553f33a5e2892abed2cdcef7768e35fbad6fa5af9ec3286a7ba90ba");
                    var width = module == "image-landscape" ? 1008 : module == "image-portrait" ? 576 : 768;
                    var height = module == "image-landscape" ? 576 : module == "image-portrait" ? 1008 : 768;
                    if ((int?)metadata["resized_width"] != width || (int?)metadata["resized_height"] != height) throw new ArgumentException("Image module geometry differs.");
                    foreach (var entry in rows)
                    {
                        var row = entry.Row;
                        var id = (string)row["id"]; var name = id.Replace('-', '/') + ".png";
                        if (!inputs.TryGetValue(id, out var input))
                        {
                            var path = Verified("vision/" + name, (long)audit["files"][name]["bytes"], (string)audit["files"][name]["sha256"]);
                            var texture = new Texture2D(2, 2, TextureFormat.RGBA32, false, true); fixture.textures.Add(texture);
                            if (!ImageConversion.LoadImage(texture, File.ReadAllBytes(path), true)) throw new InvalidDataException("Original PNG could not be decoded.");
                            inputs[id] = input = texture;
                        }
                        result.Add(new PlayerMediaCase { Id = id, Role = entry.Role, Expected = row["embedding"].Values<float>().ToArray(), Input = input });
                    }
                }
                fixture.Cases = result; return fixture;
            }
            catch { fixture.Dispose(); throw; }
        }
        public void Dispose()
        {
            foreach (var texture in textures)
                if (texture != null) { if (Application.isPlaying) UnityEngine.Object.Destroy(texture); else UnityEngine.Object.DestroyImmediate(texture); }
            textures.Clear();
        }
    }
}
