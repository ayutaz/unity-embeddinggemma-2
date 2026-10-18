using System;
using NUnit.Framework;

namespace EmbeddingGemma.Validation.Tests
{
    public sealed class EditorIndexRebuildTests
    {
        sealed class Database
        {
            public int Calls;
            public Exception Failure;
            void Build() { Calls++; if (Failure != null) throw Failure; }
        }
        sealed class WrongContract { int Build() => 1; }

        [Test] public void ExplicitBuildIsInvokedOnce()
        {
            var database = new Database();
            EditorIndexRebuild.Begin(database);
            Assert.That(database.Calls, Is.EqualTo(1));
        }
        [Test] public void OriginalBuildFailureIsPreserved()
        {
            var expected = new InvalidOperationException("index artifact unavailable");
            var database = new Database { Failure = expected };
            var actual = Assert.Throws<InvalidOperationException>(() => EditorIndexRebuild.Begin(database));
            Assert.That(actual, Is.SameAs(expected));
            Assert.That(database.Calls, Is.EqualTo(1));
        }
        [Test] public void NullDatabaseIsRejected()
            => Assert.Throws<ArgumentNullException>(() => EditorIndexRebuild.Begin(null));
        [Test] public void MissingBuildContractIsRejected()
            => Assert.Throws<NotSupportedException>(() => EditorIndexRebuild.Begin(new object()));
        [Test] public void NonVoidBuildContractIsRejected()
            => Assert.Throws<NotSupportedException>(() => EditorIndexRebuild.Begin(new WrongContract()));
    }
}
