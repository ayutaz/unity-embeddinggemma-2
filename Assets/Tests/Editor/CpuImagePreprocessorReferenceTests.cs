using System;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.Experimental.Rendering;

namespace EmbeddingGemma.Tests
{
    public sealed class CpuImagePreprocessorReferenceTests
    {
        const string RuntimeSource="1193e9ae50f7997b9436d3b6f3c887446337cbeb";
        static string DirectoryPath => Path.Combine(Application.dataPath,"../artifacts/image-preprocessing");
        static string OutputPath => Path.Combine(DirectoryPath,"cpu-unity-results-1193e9a.json");

        [TestCase("readable_linear")] [TestCase("readable_srgb")]
        [TestCase("texture_linear")] [TestCase("texture_srgb")]
        [TestCase("render_linear")] [TestCase("render_srgb")]
        public void OfficialPixelsPositionsAndCpuConsumerAgree(string path)
        {
            var referencePath=Path.Combine(DirectoryPath,"image-preprocessing-reference.json");
            Assert.That(Hash(File.ReadAllBytes(referencePath)),Is.EqualTo("8076c38ded6c2486082092c0f0a43ec553760280ac46d8b98bfaccce829c3334"));
            var reference=JObject.Parse(File.ReadAllText(referencePath));
            Assert.That(reference["metadata"].Value<string>("source_commit"),Is.EqualTo("1fd94e28045c9bb4752bf92439f02912d57dd1f2"));
            Assert.That(reference["metadata"].Value<string>("model_revision"),Is.EqualTo("914f7f89142e33e77833254d9c9b90c3cef7303b"));
            foreach(var item in (JObject)reference["files"])
            {
                var bytes=File.ReadAllBytes(Path.Combine(DirectoryPath,item.Key));
                Assert.That(bytes.LongLength,Is.EqualTo(item.Value.Value<long>("bytes")));
                Assert.That(Hash(bytes),Is.EqualTo(item.Value.Value<string>("sha256")));
            }
            var locked=JObject.Parse(File.ReadAllText(Path.Combine(Application.dataPath,"../Packages/packages-lock.json")));
            Assert.That(locked["dependencies"]["com.ayutaz.embeddinggemma"].Value<string>("hash"),Is.EqualTo(RuntimeSource));
            var report=File.Exists(OutputPath)?JObject.Parse(File.ReadAllText(OutputPath)):new JObject();
            report["scope"]="official_image_cpu_preprocessing_only";
            report["runtime_source"]=RuntimeSource; report["reference_source"]=reference["metadata"]["source_commit"];
            report["reference_sha256"]="8076c38ded6c2486082092c0f0a43ec553760280ac46d8b98bfaccce829c3334";
            report["unity"]=Application.unityVersion; report["sentis"]="2.6.1";
            report["gpu"]=SystemInfo.graphicsDeviceName; report["graphics_api"]=SystemInfo.graphicsDeviceType.ToString();
            report["encoder_validated"]=false; report["end_to_end_validated"]=false;
            var rows=new JArray(); report[path]=rows;
            void Save() => File.WriteAllText(OutputPath,report.ToString());
            Save(); using var processor=new ImagePreprocessor(BackendType.CPU);
            var graph=new FunctionalGraph(); var inputPixels=graph.AddInput<float>(new TensorShape(1,2520,768),"pixels");
            var model=graph.Compile(inputPixels+0.25f); using var consumer=new Worker(model,BackendType.CPU);
            foreach(var row in (JArray)reference["cases"])
            {
                var item=new JObject {["id"]=row.Value<string>("id"),["success"]=false}; rows.Add(item); Save();
                var width=row.Value<int>("width"); var height=row.Value<int>("height");
                var raw=File.ReadAllBytes(Path.Combine(DirectoryPath,row.Value<string>("rgb_file")));
                var linear=path.EndsWith("linear",StringComparison.Ordinal); var readable=path.StartsWith("readable",StringComparison.Ordinal);
                var texture=new Texture2D(width,height,TextureFormat.RGBA32,false,linear); RenderTexture rendered=null;
                try
                {
                    var colors=new Color32[width*height];
                    for(var y=0;y<height;y++) for(var x=0;x<width;x++)
                    {
                        var offset=(y*width+x)*3;
                        colors[(height-y-1)*width+x]=new Color32(raw[offset],raw[offset+1],raw[offset+2],(byte)((x*19+y*23)%256));
                    }
                    texture.SetPixels32(colors); texture.Apply(false,!readable); Texture input=texture;
                    if(path.StartsWith("render",StringComparison.Ordinal))
                    {
                        rendered=new RenderTexture(new RenderTextureDescriptor(width,height)
                            {graphicsFormat=linear?GraphicsFormat.R8G8B8A8_UNorm:GraphicsFormat.R8G8B8A8_SRGB,depthBufferBits=0});
                        Assert.That(rendered.Create(),Is.True); Graphics.CopyTexture(texture,rendered); input=rendered;
                    }
                    using var result=processor.Process(input);
                    Assert.That(result.Pixels.backendType,Is.EqualTo(BackendType.CPU)); Assert.That(consumer.backendType,Is.EqualTo(BackendType.CPU));
                    Assert.That(result.ResizedWidth,Is.EqualTo(row.Value<int>("resized_width")));
                    Assert.That(result.ResizedHeight,Is.EqualTo(row.Value<int>("resized_height")));
                    Assert.That(result.SoftTokenCount,Is.EqualTo(row.Value<int>("soft_tokens")));
                    Assert.That(result.Positions.ToArray(),Is.EqualTo(Read<int>(row.Value<string>("positions_file"))));
                    consumer.Schedule(result.Pixels); var final=((Tensor<float>)consumer.PeekOutput()).DownloadToArray();
                    var actual=result.Pixels.DownloadToArray(); var expected=Read<float>(row.Value<string>("pixels_file"));
                    Assert.That(actual.Length,Is.EqualTo(expected.Length)); double maximum=0,chainMaximum=0; var mismatches=0;
                    for(var index=0;index<expected.Length;index++)
                    {
                        var difference=Math.Abs((double)actual[index]-expected[index]);
                        var chained=Math.Abs((double)final[index]-(expected[index]+0.25f));
                        if(double.IsNaN(difference)||double.IsInfinity(difference)||difference>1e-6||chained>1e-6) mismatches++;
                        maximum=Math.Max(maximum,difference); chainMaximum=Math.Max(chainMaximum,chained);
                    }
                    item["maximum_absolute_error"]=maximum; item["downstream_maximum_absolute_error"]=chainMaximum;
                    item["mismatched_elements"]=mismatches; item["actual_backend"]="CPU"; item["input_texture_readback_required"]=!readable;
                    Save(); Assert.That(mismatches,Is.Zero,row.Value<string>("id"));
                    Assert.That(input==null,Is.False); if(rendered!=null) Assert.That(rendered.IsCreated(),Is.True);
                    item["success"]=true; Save();
                }
                finally {if(rendered!=null){rendered.Release();UnityEngine.Object.DestroyImmediate(rendered);}UnityEngine.Object.DestroyImmediate(texture);}
            }
            report["last_completed_utc"]=DateTime.UtcNow.ToString("O"); Save();
        }
        static string Hash(byte[] bytes) {using var sha=SHA256.Create();return BitConverter.ToString(sha.ComputeHash(bytes)).Replace("-","").ToLowerInvariant();}
        static T[] Read<T>(string name) where T:struct
        {var bytes=File.ReadAllBytes(Path.Combine(DirectoryPath,name));var result=new T[bytes.Length/4];Buffer.BlockCopy(bytes,0,result,0,bytes.Length);return result;}
    }
}
