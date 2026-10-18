using System;
using System.IO;
using System.Reflection;
using NUnit.Framework;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class PlayerDeviceStreamingSourceTests
    {
        static string Source(string path)
        {
            var method = typeof(PlayerDeviceMixedConfiguration).GetMethod("FixtureSource");
            Assert.That(method, Is.Not.Null, "Device ingress must normalize native StreamingAssets paths to local file URIs while retaining Android jar URIs.");
            return (string)method.Invoke(null, new object[] { path });
        }
        [Test] public void NativeStreamingAssetsPathBecomesAnEscapedFileUri()
        {
            var path = Path.Combine(Path.GetTempPath(), "EmbeddingGemma synthetic assets");
            var expected = new Uri(Path.Combine(path, PlayerDeviceMixedConfiguration.DirectoryName, "fixtures")).AbsoluteUri;
            Assert.That(Source(path), Is.EqualTo(expected)); Assert.That(Directory.Exists(path), Is.False);
        }
        [Test] public void AlreadyLocalFileUriRetainsItsSchemeAndFixedChild()
        { Assert.That(Source("file:///synthetic/assets/"), Is.EqualTo("file:///synthetic/assets/EmbeddingGemmaDeviceValidation/fixtures")); }
        [Test] public void AndroidApkUriRetainsItsJarBoundary()
        { Assert.That(Source("jar:file:///data/app/synthetic/base.apk!/assets"), Is.EqualTo("jar:file:///data/app/synthetic/base.apk!/assets/EmbeddingGemmaDeviceValidation/fixtures")); }
        [TestCase("")] [TestCase(null)] [TestCase("relative")]
        [TestCase("https://example.invalid/assets")] [TestCase("file://server/assets")]
        [TestCase("jar:file://server/base.apk!/assets")] [TestCase("file:///assets?x=1")]
        [TestCase("file:///assets#fragment")] [TestCase("jar:file:///base.apk")] [TestCase("ftp://example.invalid/assets")]
        public void RemoteRelativeOrAmbiguousSourcesCannotEnterTheDeviceFixtureTransfer(string path)
        {
            var method = typeof(PlayerDeviceMixedConfiguration).GetMethod("FixtureSource");
            Assert.That(method, Is.Not.Null, "Device ingress must normalize native StreamingAssets paths to local file URIs while retaining Android jar URIs.");
            var error = Assert.Throws<TargetInvocationException>(() => Source(path));
            Assert.That(error.InnerException, Is.InstanceOf<ArgumentException>().Or.InstanceOf<NotSupportedException>());
        }
    }
}
