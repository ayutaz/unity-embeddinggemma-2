using System;
using System.Collections;
using System.IO;
using System.Threading;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;

namespace EmbeddingGemma.Validation.Tests
{
    // Explicit new-distribution entry points reuse original PCM/query/reference checks.
    public sealed class CandidateMultimodalSearchSampleGpuTests
    {
        [UnityTearDown] public IEnumerator LeavePlayMode()
        { if (Application.isPlaying) yield return new ExitPlayMode(); }

        [UnityTest, Timeout(900000)]
        public IEnumerator ReviewedDistributionSampleKeepsOriginalCrossModalReferences()
            => Execute(false);

        [UnityTest, Timeout(900000)]
        public IEnumerator ReviewedDistributionSampleRendersOriginalSearchResultsInPlayMode()
            => Execute(true);

        static IEnumerator Execute(bool render)
        {
            CandidateMediaBundlePolicy.VerifyCases(Environment.GetEnvironmentVariable("EG_MEDIA_SAMPLE_CASES"));
            if (render)
            {
                Assert.That(Environment.GetEnvironmentVariable("EG_MEDIA_SAMPLE_SCREENSHOT"),Is.Not.Null.And.Not.Empty);
                UnityEditor.SceneManagement.EditorSceneManager.OpenScene("Assets/EmbeddingGemmaMultimodalSearch/MultimodalSearch.unity");
                Assert.That(UnityEditor.EditorApplication.ExecuteMenuItem("Window/General/Game"),Is.True);
                yield return new EnterPlayMode();
            }
            using var cancellation=new CancellationTokenSource();long peak=0;
            var task=MultimodalSearchSampleGpuTests.RunWithPolicy(cancellation.Token,CandidateMediaBundlePolicy.VerifyCurrent);
            while(!task.IsCompleted)
            {
                peak=Math.Max(peak,MediaExecutionMemory.Peak());
                if(peak>GpuMediaExecutionPolicy.CurrentMemoryLimitBytes)cancellation.Cancel();
                if(!Application.isPlaying)UnityEditor.EditorApplication.QueuePlayerLoopUpdate();
                yield return null;
            }
            var output=Environment.GetEnvironmentVariable("EG_MEDIA_SAMPLE_REPORT");
            if(!string.IsNullOrEmpty(output)&&File.Exists(output))
            {
                var report=JObject.Parse(File.ReadAllText(output));report["host_native_peak_bytes"]=peak;
                report["composed_candidate_distribution"]=true;report["play_mode_execution"]=render;
                report["full_multimodal_release_qualified"]=false;File.WriteAllText(output,report.ToString());
            }
            task.GetAwaiter().GetResult();
            Assert.That(peak,Is.GreaterThan(0).And.LessThanOrEqualTo(GpuMediaExecutionPolicy.CurrentMemoryLimitBytes));
            if(render)yield return new ExitPlayMode();
        }
    }
}
