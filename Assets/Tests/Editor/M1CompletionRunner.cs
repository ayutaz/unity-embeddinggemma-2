using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEditor;
using UnityEngine;
using UnityEngine.Profiling;
using Debug = UnityEngine.Debug;

namespace EmbeddingGemma.Tests
{
    // Runs in the Editor test assembly; production API has no Editor dependency.
    public static class M1CompletionRunner
    {
        [Serializable] public sealed class CaseResult { public string id; public double cosine, roundTripError, norm; }
        [Serializable] public sealed class Memory
        {
            public string stage;
            public long unityAllocatedBytes, unityReservedBytes, graphicsDriverBytes, processWorkingSetBytes, processPrivateBytes;
            public long managedHeapBytes;
            public bool graphicsCounterAvailable, processCountersAvailable;
        }
        [Serializable] public sealed class BackendResult
        {
            public string backend;
            public double modelLoadMs, tokenizerPreparationMs, workerAndTokenizerMs, firstInferenceMs, medianMs, p95Ms;
            public double minimumCosine = 1, maximumRoundTripError;
            public List<CaseResult> cases = new();
            public List<double> samplesMs = new();
            public List<Memory> memory = new();
        }
        [Serializable] public sealed class FormatResult
        {
            public string precision, path, sha256;
            public long bytes;
            public double sourceLoadMs, quantizeMs, saveMs;
            public List<BackendResult> backends = new();
        }
        [Serializable] public sealed class Report
        {
            public bool success;
            public string error, startedUtc, completedUtc, unity, sentis = "2.6.1", os, cpu, gpu, graphicsApi, modelRevision, sourceCommit;
            public int processorCount, systemMemoryMB, gpuCapacityMB, sequenceLength = 128, batchSize = 1, embeddingDimension = 768;
            public int warmupRounds = 2, measurementRounds = 3;
            public string timingScope = "Synchronous public API, including prompting, tokenization, scheduling and final vector readback. Samples follow reference case order in each round.";
            public string memoryScope = "Stage samples of the whole Unity Editor/process, not model-exclusive VRAM or a continuously observed peak. Zero graphics or process counters mean unavailable; availability flags are explicit. Managed heap is measured separately and must not be summed with overlapping native/process counters.";
            public double forcedAssetImportMs;
            public List<FormatResult> formats = new();
            public List<Memory> memory = new();
        }
        [Serializable] sealed class Reference { public int schema_version; public Metadata metadata; public Item[] cases; }
        [Serializable] sealed class Metadata { public string model_revision, source_commit; public int sequence_length, embedding_dimension; }
        [Serializable] sealed class Item { public string id, formatted_text; public int[] input_ids, attention_mask; public float[] embedding; }

        public static Memory CaptureMemory(string stage)
        {
            using var process = Process.GetCurrentProcess();
            var graphics = Profiler.GetAllocatedMemoryForGraphicsDriver();
            return new Memory { stage = stage, unityAllocatedBytes = Profiler.GetTotalAllocatedMemoryLong(),
                unityReservedBytes = Profiler.GetTotalReservedMemoryLong(), graphicsDriverBytes = graphics,
                graphicsCounterAvailable = graphics > 0, managedHeapBytes = GC.GetTotalMemory(false),
                processCountersAvailable = process.WorkingSet64 > 0 && process.PrivateMemorySize64 > 0,
                processWorkingSetBytes = process.WorkingSet64, processPrivateBytes = process.PrivateMemorySize64 };
        }
        static double Time(Action action) { var watch = Stopwatch.StartNew(); action(); return watch.Elapsed.TotalMilliseconds; }
        static string Hash(string path)
        {
            using var stream = File.OpenRead(path); using var sha = SHA256.Create();
            return BitConverter.ToString(sha.ComputeHash(stream)).Replace("-", "").ToLowerInvariant();
        }
        static double Norm(float[] values) => Math.Sqrt(values.Sum(v => (double)v * v));
        static double Cosine(float[] actual, float[] expected) => actual.Zip(expected, (a,b) => (double)a*b).Sum() / (Norm(actual)*Norm(expected));

        public static Report Run()
        {
            var root = Path.GetFullPath(Path.Combine(Application.dataPath, ".."));
            var input = Path.Combine(root, "artifacts/m1");
            var output = Path.Combine(root, "artifacts/m1-completion");
            Directory.CreateDirectory(output);
            var reportPath = Path.Combine(output, "results.json");
            var report = new Report { startedUtc = DateTime.UtcNow.ToString("O"), unity = Application.unityVersion,
                os = SystemInfo.operatingSystem, cpu = SystemInfo.processorType, processorCount = SystemInfo.processorCount,
                systemMemoryMB = SystemInfo.systemMemorySize, gpu = SystemInfo.graphicsDeviceName,
                gpuCapacityMB = SystemInfo.graphicsMemorySize, graphicsApi = SystemInfo.graphicsDeviceType.ToString() };
            void Save() => File.WriteAllText(reportPath, JsonUtility.ToJson(report, true));
            Save(); // Invalidate old success even if import, loading or tests fail.
            try
            {
                var reference = JsonUtility.FromJson<Reference>(File.ReadAllText(Path.Combine(input, "reference.json")));
                Assert.That(reference.schema_version, Is.EqualTo(1));
                Assert.That(reference.metadata.model_revision, Is.EqualTo("914f7f89142e33e77833254d9c9b90c3cef7303b"));
                Assert.That(reference.metadata.sequence_length, Is.EqualTo(128));
                Assert.That(reference.metadata.embedding_dimension, Is.EqualTo(768));
                Assert.That(reference.cases, Has.Length.EqualTo(15));
                Assert.That(SystemInfo.supportsComputeShaders, Is.True, "GPU must execute without skip or fallback.");
                report.modelRevision = reference.metadata.model_revision; report.sourceCommit = reference.metadata.source_commit;
                var tokenizerJson = File.ReadAllText(Path.Combine(input, "tokenizer.json"));
                var tokenizer = new TextTokenizer(tokenizerJson);
                foreach (var item in reference.cases)
                {
                    var tokens = tokenizer.Encode(item.formatted_text);
                    Assert.That(tokens.GetIds(), Is.EqualTo(item.input_ids), item.id);
                    Assert.That(tokens.GetAttentionMask(), Is.EqualTo(item.attention_mask), item.id);
                }
                report.memory.Add(CaptureMemory("before forced synchronous .pt2 import"));
                const string assetPath = "Assets/M1Generated/model.pt2";
                report.forcedAssetImportMs = Time(() => AssetDatabase.ImportAsset(assetPath, ImportAssetOptions.ForceUpdate | ImportAssetOptions.ForceSynchronousImport));
                var asset = AssetDatabase.LoadAssetAtPath<ModelAsset>(assetPath);
                Assert.That(asset, Is.Not.Null);
                report.memory.Add(CaptureMemory("after import; Editor caches included"));
                // Independent before-save outputs, retained as small vectors only.
                var baseline = new Dictionary<BackendType, float[][]>();
                foreach (var backend in new[] { BackendType.CPU, BackendType.GPUCompute })
                {
                    using var embedder = new TextEmbedder(ModelLoader.Load(asset), tokenizerJson, backend);
                    baseline[backend] = reference.cases.Select(item => embedder.EmbedRaw(item.formatted_text)).ToArray();
                }
                foreach (var precision in new[] { "fp32", "float16-weights" })
                {
                    var format = new FormatResult { precision = precision, path = Path.Combine(output, precision + ".sentis") };
                    report.formats.Add(format);
                    Model model = null;
                    format.sourceLoadMs = Time(() => model = precision == "fp32" ? ModelLoader.Load(asset) : TextModelFile.Load(report.formats[0].path));
                    if (precision != "fp32") format.quantizeMs = Time(() => ModelQuantizer.QuantizeWeights(QuantizationType.Float16, ref model));
                    format.saveMs = Time(() => TextModelFile.Save(model, format.path));
                    format.bytes = new FileInfo(format.path).Length; format.sha256 = Hash(format.path);
                    model = null; GC.Collect(); GC.WaitForPendingFinalizers();
                    foreach (var backend in new[] { BackendType.CPU, BackendType.GPUCompute })
                    {
                        var result = new BackendResult { backend = backend.ToString() }; format.backends.Add(result);
                        result.memory.Add(CaptureMemory("before model load"));
                        Model loaded = null;
                        result.modelLoadMs = Time(() => loaded = TextModelFile.Load(format.path));
                        result.memory.Add(CaptureMemory("after saved model load"));
                        result.tokenizerPreparationMs = Time(() => { var independent = new TextTokenizer(tokenizerJson); independent.Encode(reference.cases[0].formatted_text); });
                        TextEmbedder embedder = null;
                        result.workerAndTokenizerMs = Time(() => embedder = new TextEmbedder(loaded, tokenizerJson, backend));
                        using (embedder)
                        {
                            Assert.That(embedder.Backend, Is.EqualTo(backend));
                            result.firstInferenceMs = Time(() => embedder.EmbedRaw(reference.cases[0].formatted_text));
                            result.memory.Add(CaptureMemory("after first inference and readback"));
                            for (var i = 0; i < reference.cases.Length; i++)
                            {
                                var item = reference.cases[i]; var actual = embedder.EmbedRaw(item.formatted_text);
                                Assert.That(actual, Has.Length.EqualTo(768), item.id);
                                Assert.That(actual.All(v => !float.IsNaN(v) && !float.IsInfinity(v)), Is.True, item.id);
                                var norm = Norm(actual); Assert.That(norm, Is.EqualTo(1).Within(1e-3), item.id);
                                var cosine = Cosine(actual, item.embedding);
                                Assert.That(cosine, Is.GreaterThanOrEqualTo(precision == "fp32" ? 0.999 : 0.99), item.id);
                                var difference = actual.Zip(baseline[backend][i], (a,b) => Math.Abs((double)a-b)).Max();
                                if (precision == "fp32") Assert.That(difference, Is.LessThanOrEqualTo(1e-5), item.id);
                                result.minimumCosine = Math.Min(result.minimumCosine, cosine);
                                result.maximumRoundTripError = Math.Max(result.maximumRoundTripError, difference);
                                result.cases.Add(new CaseResult { id = item.id, cosine = cosine, roundTripError = difference, norm = norm });
                            }
                            for (var round = 0; round < report.warmupRounds; round++)
                                foreach (var item in reference.cases) embedder.EmbedRaw(item.formatted_text);
                            result.memory.Add(CaptureMemory("after warmup"));
                            for (var round = 0; round < report.measurementRounds; round++)
                                foreach (var item in reference.cases) result.samplesMs.Add(Time(() => embedder.EmbedRaw(item.formatted_text)));
                            var sorted = result.samplesMs.OrderBy(v => v).ToArray();
                            result.medianMs = sorted[sorted.Length / 2]; result.p95Ms = sorted[(int)Math.Ceiling(sorted.Length * 0.95) - 1];
                            result.memory.Add(CaptureMemory("after measurement"));
                        }
                        loaded = null; embedder = null; GC.Collect(); GC.WaitForPendingFinalizers();
                        result.memory.Add(CaptureMemory("after Worker Dispose and managed GC; Editor caches retained"));
                        Debug.Log($"M1 completion {precision} / {backend}: cases=15, min_cosine={result.minimumCosine.ToString("R", CultureInfo.InvariantCulture)}, median_ms={result.medianMs:F3}");
                        Save();
                    }
                }
                Assert.That(report.formats[1].bytes, Is.LessThan(report.formats[0].bytes));
                report.success = true;
                return report;
            }
            catch (Exception error) { report.error = error.ToString(); throw; }
            finally { report.completedUtc = DateTime.UtcNow.ToString("O"); Save(); }
        }
    }
}
