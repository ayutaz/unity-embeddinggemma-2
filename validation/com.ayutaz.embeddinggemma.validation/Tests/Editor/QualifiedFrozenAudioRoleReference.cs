using System;
using System.IO;
using System.Linq;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using UnityEngine;

namespace EmbeddingGemma.Validation.Tests
{
    // Reference-only artifact11990006373 / Actions38764026134. All original
    // Document anchors passed; the CPU encoder was not executed in this mode.
    public static class QualifiedFrozenAudioRoleReference
    {
        public static string ReferenceSha(TextRole role)=>role==TextRole.Query?"515a32124c94437b372f81b3be4eb605717b1c4ae9293bf2c3b851450f35565b":
            role==TextRole.Raw?"ea4298313a536aa3b0c1f5eac7934240abaf3cb68170be36deb64c214fbb92a9":
            role==TextRole.Document?"064e01856ef5f425eb119540e1e88304a813d3572ccd097f8191207840e9b458":throw new ArgumentOutOfRangeException(nameof(role));
        public static JObject Load(TextRole role)
        {
            var name=role.ToString().ToLowerInvariant();var root=Path.GetFullPath(Path.Combine(Application.dataPath,"../../numerical-acceptance-state/audio-frozen-role-reference-38764026134/audio-frozen-role-reference",name));
            var path=Path.Combine(root,"audio-embedding-reference.json");var auditPath=Path.Combine(root,"audio-embedding-validation.json");
            Assert.That(QualifiedImageBackboneSquare.Hash(path),Is.EqualTo(ReferenceSha(role)));
            var auditSha=role==TextRole.Query?"f7d50f1a47d85abeebf614f5a7c57d2478c7a1cee81cbddaae70ee77e8590ac9":
                role==TextRole.Raw?"750a88ba7901b46c15e2e71223618eab075e3122535e92209da765996760c281":"9c2cad5a24e7ac6793818ca573d011c87296a6be3d08969334179e1f05b7ed45";
            Assert.That(QualifiedImageBackboneSquare.Hash(auditPath),Is.EqualTo(auditSha));
            var reference=JObject.Parse(File.ReadAllText(path));var audit=JObject.Parse(File.ReadAllText(auditPath));var metadata=reference["metadata"];
            Assert.That(audit.Value<bool>("success"),Is.True);Assert.That(audit.Value<string>("stage"),Is.EqualTo("python_audio_frozen_feature_role_reference"));
            Assert.That(JToken.DeepEquals(metadata,audit["metadata"]),Is.True);
            Assert.That(metadata.Value<string>("source_commit"),Is.EqualTo("0a8ad5b5c7b9e2582e65a381c0f22c3d1289cfb6"));
            foreach(var key in new[]{"encoder_source_commit","backbone_source_commit"})Assert.That(metadata.Value<string>(key),Is.EqualTo("4edbe17a8105233c5aaa1514faa2862b08a4f7e0"));
            Assert.That(metadata.Value<string>("model_revision"),Is.EqualTo("914f7f89142e33e77833254d9c9b90c3cef7303b"));
            Assert.That(metadata.Value<string>("reference_boundary"),Is.EqualTo("original_frozen_encoder_outputs_to_official_backbone"));
            Assert.That(metadata.Value<string>("backbone_reference_sha256"),Is.EqualTo("281d12573b431667e8841033db1c6a10dce4382ef98c9a5345c18044a1194001"));
            Assert.That(metadata.Value<string>("encoder_reference_sha256"),Is.EqualTo("ac23f39904b8149dd5ccb6485afcc93d96047f0977b5faf1d640c8bd018c13c9"));
            Assert.That(metadata.Value<string>("role"),Is.EqualTo(name));Assert.That(audit.Value<int>("original_document_anchor_cases"),Is.EqualTo(6));
            Assert.That(audit.Value<bool>("frozen_encoder_features_validated"),Is.True);
            foreach(var key in new[]{"encoder_execution_validated","python_native_saved_chain_validated","gpu_chain_validated","end_to_end_validated"})Assert.That(audit.Value<bool>(key),Is.False);
            var cases=(JArray)reference["cases"];Assert.That(cases.Count,Is.EqualTo(6));
            Assert.That(cases.Select(row=>row.Value<string>("id")),Is.EquivalentTo(new[]{"silence","impulse","sine","nyquist","noise","padded_noise"}));
            Assert.That(cases.All(row=>row.Value<double>("document_anchor_cosine")>=0.999999),Is.True);return reference;
        }
    }
}
