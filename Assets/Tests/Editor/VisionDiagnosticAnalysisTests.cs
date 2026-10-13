using System;
using NUnit.Framework;

namespace EmbeddingGemma.Tests
{
    public sealed class VisionDiagnosticAnalysisTests
    {
        [Test]
        public void DifferentNativeReferenceCannotClaimLocalizationEvenWhenSentisFinalsAgree()
        {
            var result = VisionDiagnosticAnalysis.AnalyzePinnedReference(new[] { 1f }, new[] { 1.02f }, new[] { 1.02f }, new[] { 1.01f }, true);
            Assert.That(result.TraceMatchesOriginal, Is.True);
            Assert.That(result.NativeReference.Mismatches, Is.EqualTo(1));
            Assert.That(result.CanLocalize, Is.False, "The native reference gate must be checked before publishing localization.");
        }

        [Test]
        public void MatchingNativeReferenceAllowsLocalizationAtTheFixedTolerance()
        {
            var result = VisionDiagnosticAnalysis.AnalyzePinnedReference(new[] { 1f }, new[] { 1.02f }, new[] { 1.02f }, new[] { 1.000001f }, true);
            Assert.That(result.NativeReference.Passed, Is.True);
            Assert.That(result.CanLocalize, Is.True);
        }

        [Test]
        public void MatchingNativeReferenceDoesNotOverrideFailedPythonContext()
        {
            var result = VisionDiagnosticAnalysis.AnalyzePinnedReference(new[] { 1f }, new[] { 1.02f }, new[] { 1.02f }, new[] { 1f }, false);
            Assert.That(result.CanLocalize, Is.False);
        }

        [Test]
        public void SameFailedFinalCanBeLocalizedOnlyWithPinnedPythonContext()
        {
            var result = VisionDiagnosticAnalysis.Analyze(new[] { 1f }, new[] { 1.01f }, new[] { 1.01f }, true);
            Assert.That(result.Original.Mismatches, Is.EqualTo(1));
            Assert.That(result.Original.MaximumToleranceRatio, Is.GreaterThan(1));
            Assert.That(result.TraceMatchesOriginal, Is.True);
            Assert.That(result.CanLocalize, Is.True);
        }

        [Test]
        public void TwoFailingGraphsWithDifferentFinalsCannotProveLocalization()
        {
            var result = VisionDiagnosticAnalysis.Analyze(new[] { 1f }, new[] { 1.01f }, new[] { 1.02f }, true);
            Assert.That(result.TraceMatchesOriginal, Is.False);
            Assert.That(result.CanLocalize, Is.False);
        }

        [Test]
        public void PassingOriginalCannotClaimTheKnownFailureWasReproduced()
        {
            var result = VisionDiagnosticAnalysis.Analyze(new[] { 1f }, new[] { 1.000001f }, new[] { 1.000001f }, true);
            Assert.That(result.Original.Passed, Is.True);
            Assert.That(result.CanLocalize, Is.False);
        }

        [Test]
        public void UnmatchedPythonFinalCannotBeRepairedBySentisAgreement()
        {
            var result = VisionDiagnosticAnalysis.Analyze(new[] { 1f }, new[] { 1.01f }, new[] { 1.01f }, false);
            Assert.That(result.CanLocalize, Is.False);
        }

        [Test]
        public void ChangedTraceShapeCannotClaimSamePath()
        {
            var result = VisionDiagnosticAnalysis.Analyze(new[] { 1f }, new[] { 1.01f }, new[] { 1.01f, 0 }, true);
            Assert.That(result.TraceMatchesOriginal, Is.False);
            Assert.That(result.CanLocalize, Is.False);
        }

        [TestCase(float.NaN)]
        [TestCase(float.PositiveInfinity)]
        public void NonfiniteTraceCannotClaimSamePath(float value)
        {
            var result = VisionDiagnosticAnalysis.Analyze(new[] { 1f }, new[] { 1.01f }, new[] { value }, true);
            Assert.That(result.TraceMatchesOriginal, Is.False);
            Assert.That(result.CanLocalize, Is.False);
        }

        [Test]
        public void FixedToleranceUsesReferenceMagnitudeAndRecordsSubthresholdDifference()
        {
            var result = VisionDiagnosticAnalysis.Compare(new[] { 1f }, new[] { 1.000001f });
            Assert.That(result.Passed, Is.True);
            Assert.That(result.MaximumAbsoluteError, Is.GreaterThan(0));
            Assert.That(result.MaximumToleranceRatio, Is.GreaterThan(0).And.LessThan(1));
        }

        [Test]
        public void NullVectorIsRejected() =>
            Assert.Throws<ArgumentNullException>(() => VisionDiagnosticAnalysis.Compare(null, new[] { 1f }));

        [Test]
        public void DifferentVectorSizeIsRejected() =>
            Assert.Throws<ArgumentException>(() => VisionDiagnosticAnalysis.Compare(new[] { 1f }, new[] { 1f, 2 }));

        [TestCase(float.NaN)]
        [TestCase(float.NegativeInfinity)]
        public void NonfiniteVectorIsRejected(float value) =>
            Assert.Throws<ArgumentException>(() => VisionDiagnosticAnalysis.Compare(new[] { 1f }, new[] { value }));
    }
}
