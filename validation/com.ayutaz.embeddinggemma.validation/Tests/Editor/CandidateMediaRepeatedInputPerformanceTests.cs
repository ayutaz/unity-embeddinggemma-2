using System;
using System.Collections;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;

namespace EmbeddingGemma.Validation.Tests
{
    /// <summary>Explicit reviewed-distribution entry; keeps the retained measurement independently pinned.</summary>
    public sealed class CandidateMediaRepeatedInputPerformanceTests
    {
        public static MediaRepeatedInputPolicy CreatePolicy()=>MediaRepeatedInputPolicy.CreateCandidate(
            Environment.GetEnvironmentVariable("EG_MEDIA_PRECISION"),Environment.GetEnvironmentVariable("EG_MEDIA_BUNDLE_MODULE"));
        [UnityTearDown] public IEnumerator Exit()
        {if(Application.isPlaying)yield return new ExitPlayMode();}
        [UnityTest,Timeout(1200000)]
        public IEnumerator ReviewedDistributionFixedInputFirstWarmRepeatedCallsKeepReferenceAndMeasureStages()
            =>MediaRepeatedInputPerformanceTests.ExecuteWithPolicy(CreatePolicy());
    }
}
