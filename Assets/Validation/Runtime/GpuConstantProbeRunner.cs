using System;
using System.IO;
using System.Text;
using System.Text.RegularExpressions;
using Newtonsoft.Json.Linq;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.Rendering;

namespace EmbeddingGemma.Validation
{
    public static class GpuConstantProbeRunner
    {
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        static void Start()
        {
            var marker = Resources.Load<TextAsset>("GpuConstantProbeBuild");
            if (marker == null) return;
            try
            {
                var args = Environment.GetCommandLineArgs();
                string output = null;
                for (var index = 0; index < args.Length; index++)
                    if (args[index] == "-gpuConstantProbeOutput")
                    {
                        if (output != null || index + 1 >= args.Length) throw new ArgumentException("A single probe output argument is required.");
                        output = args[++index];
                    }
                Run(JObject.Parse(marker.text), output);
            }
            catch (Exception error) { Debug.LogException(error); Application.Quit(2); }
        }

        public static JObject Run(JObject marker, string output, Func<BackendType, JObject> evaluate = null, Action<int> quit = null)
        {
            if (string.IsNullOrWhiteSpace(output) || !Path.IsPathRooted(output) || File.Exists(output))
                throw new ArgumentException("A new absolute probe output path is required.");
            Directory.CreateDirectory(Path.GetDirectoryName(output));
            var cases = new JArray();
            var report = new JObject
            {
                ["success"] = false, ["checks_passed"] = false, ["injected_checks"] = evaluate != null,
                ["is_editor"] = Application.isEditor, ["development_build"] = Debug.isDebugBuild,
                ["unity"] = Application.unityVersion, ["platform"] = Application.platform.ToString(),
                ["graphics_api"] = SystemInfo.graphicsDeviceType.ToString(), ["gpu"] = SystemInfo.graphicsDeviceName,
                ["build_info"] = marker?.DeepClone(), ["cases"] = cases, ["started_utc"] = DateTime.UtcNow.ToString("O"),
                ["scope"] = "Weights-free bounded GPU constant storage and stripping probe; not real-model or platform acceptance."
            };
#if ENABLE_IL2CPP
            report["il2cpp"] = true;
#else
            report["il2cpp"] = false;
#endif
            try
            {
                if (marker == null || marker.Value<bool?>("injected_build") != false ||
                    !Regex.IsMatch(marker.Value<string>("source_commit") ?? "", "\\A[a-f0-9]{40}\\z") ||
                    marker.Value<string>("scripting_backend") != "IL2CPP" || marker.Value<string>("stripping") != "High" ||
                    marker.Value<string>("compiler") != "Release" || marker.Value<string>("graphics_api") != "Direct3D12")
                    throw new InvalidOperationException("Probe build provenance is invalid.");
                foreach (var key in new[] { "runtime/GpuConstantWorker.cs", "runtime/link.xml" })
                    if (!Regex.IsMatch((string)marker["source_sha256"]?[key] ?? "", "\\A[a-f0-9]{64}\\z"))
                        throw new InvalidOperationException("Probe source hash is missing: " + key);
                var passed = true;
                foreach (var backend in new[] { BackendType.CPU, BackendType.GPUCompute })
                {
                    try
                    {
                        var result = (evaluate ?? GpuConstantProbeChecks.Evaluate)(backend);
                        if (result == null) throw new InvalidOperationException("Probe returned no result.");
                        cases.Add(result);
                        var gpu = backend == BackendType.GPUCompute;
                        passed &= result.Value<bool?>("success") == true && result.Value<string>("actual_backend") == backend.ToString() &&
                            result.Value<int?>("output_elements") == 2097153 && result.Value<int?>("mismatches") == 0 &&
                            result.Value<bool?>("borrowed_model_reusable") == true &&
                            result.Value<int?>("uploaded_constants") == (gpu ? 1 : 0) &&
                            result.Value<int?>("upload_calls") == (gpu ? 2 : 0) &&
                            result.Value<int?>("maximum_upload_bytes") == (gpu ? 8388608 : 0);
                    }
                    catch (Exception error)
                    {
                        passed = false;
                        cases.Add(new JObject { ["success"] = false, ["requested_backend"] = backend.ToString(), ["error"] = error.ToString() });
                    }
                }
                report["checks_passed"] = passed;
                report["success"] = passed && evaluate == null && !Application.isEditor && !Debug.isDebugBuild &&
                    report.Value<bool>("il2cpp") && SystemInfo.graphicsDeviceType == GraphicsDeviceType.Direct3D12;
            }
            catch (Exception error) { report["error"] = error.ToString(); }
            report["completed_utc"] = DateTime.UtcNow.ToString("O");
            File.WriteAllText(output, report.ToString(), new UTF8Encoding(false));
            (quit ?? Application.Quit)(report.Value<bool>("success") ? 0 : 1);
            return report;
        }
    }
}
