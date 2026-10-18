using System;
using System.IO;
using System.Reflection;
using NUnit.Framework;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class CandidateAudioBundlePreparationTests
    {
        static MethodInfo Required()
        {
            var type=typeof(CandidateAudioBundlePreparationTests).Assembly.GetType("EmbeddingGemma.Validation.Tests.CandidateAudioBundlePreparation");
            Assert.That(type,Is.Not.Null,"Pinned candidate bundle preparation is missing.");
            var method=type.GetMethod("Prepare",BindingFlags.Public|BindingFlags.Static);Assert.That(method,Is.Not.Null);return method;
        }
        [TestCase("directory")]
        [TestCase("file")]
        public void ExistingEvidenceIsRejectedBeforeAnyModelOrGitRead(string kind)
        {
            var root=Path.Combine(Path.GetTempPath(),"eg-candidate-bundle-"+Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(root);var output=Path.Combine(root,"output");
            if(kind=="directory")Directory.CreateDirectory(output);
            var marker=kind=="file"?output:Path.Combine(output,"evidence.txt");File.WriteAllText(marker,"retain");
            try
            {
                var method=Required();var error=Assert.Throws<TargetInvocationException>(()=>method.Invoke(null,new object[]{new string('a',40),output}));
                Assert.That(error.InnerException,Is.InstanceOf<IOException>());Assert.That(File.ReadAllText(marker),Is.EqualTo("retain"));
            }
            finally
            {
                var full=Path.GetFullPath(root);var temporary=Path.GetFullPath(Path.GetTempPath()).TrimEnd(Path.DirectorySeparatorChar)+Path.DirectorySeparatorChar;
                Assert.That(full.StartsWith(temporary,StringComparison.OrdinalIgnoreCase)&&Path.GetFileName(full).StartsWith("eg-candidate-bundle-",StringComparison.Ordinal),Is.True);
                Directory.Delete(full,true);
            }
        }
        [Test]
        public void MutableExecutionSourceIsRejectedBeforeAnyModelOrGitRead()
        {
            var method=Required();var error=Assert.Throws<TargetInvocationException>(()=>method.Invoke(null,new object[]{"main","unused-new-output"}));
            Assert.That(error.InnerException,Is.InstanceOf<ArgumentException>());
        }
    }
}
