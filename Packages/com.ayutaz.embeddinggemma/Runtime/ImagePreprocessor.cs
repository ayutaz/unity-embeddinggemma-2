using System;
using System.Collections.Generic;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;

namespace EmbeddingGemma
{
    /// <summary>Official 8-bit RGB image preprocessing on CPU or GPUCompute, batch one.</summary>
    /// <remarks>Main thread only. Input textures are borrowed. Keep each owned result alive
    /// until its downstream work completes. GPUCompute performs no intermediate pixel readback.
    /// CPU reads readable Texture2D storage directly; GPU-only inputs require one synchronous input readback.</remarks>
    public sealed class ImagePreprocessor : IDisposable, IImagePreprocessor
    {
        public BackendType Backend { get; }
        bool disposed;
        ComputeShader shader;

        public ImagePreprocessor() : this(BackendType.GPUCompute) { }

        public ImagePreprocessor(BackendType backend)
        {
            if (backend != BackendType.CPU && backend != BackendType.GPUCompute)
                throw new ArgumentOutOfRangeException(nameof(backend), "Use CPU or GPUCompute explicitly.");
            Backend = backend;
        }

        public ImagePreprocessingResult Process(Texture input)
        {
            if (disposed) throw new ObjectDisposedException(nameof(ImagePreprocessor));
            if (ReferenceEquals(input, null)) throw new ArgumentNullException(nameof(input));
            if (input == null) throw new ArgumentException("The borrowed texture has been destroyed.", nameof(input));
            if (!(input is Texture2D) && !(input is RenderTexture))
                throw new ArgumentException("Expected a 2D Texture or RenderTexture.", nameof(input));
            var format = input.graphicsFormat;
            // LoadImage can expose legacy ARGB32 storage with a native graphics
            // format absent from the managed enum. GPU sampling handles its
            // channel order; qualify the explicit byte TextureFormat instead.
            var nativeArgb = Backend == BackendType.GPUCompute && input is Texture2D decoded && decoded.format == TextureFormat.ARGB32;
            if (!nativeArgb && format != GraphicsFormat.R8G8B8_UNorm && format != GraphicsFormat.R8G8B8_SRGB &&
                format != GraphicsFormat.R8G8B8A8_UNorm && format != GraphicsFormat.R8G8B8A8_SRGB &&
                format != GraphicsFormat.B8G8R8A8_UNorm && format != GraphicsFormat.B8G8R8A8_SRGB)
                throw new ArgumentException("Expected uncompressed 8-bit RGB / RGBA storage; alpha is ignored.", nameof(input));
            if (input is RenderTexture render && (!render.IsCreated() || render.antiAliasing != 1 ||
                render.dimension != TextureDimension.Tex2D))
                throw new ArgumentException("RenderTexture must already exist and be single-sample 2D.", nameof(input));
            if (input.width < 1 || input.height < 1 || input.width > 8192 || input.height > 8192)
                throw new ArgumentException("Expected image dimensions in [1,8192]. Input is never cropped.", nameof(input));
            if (Backend == BackendType.GPUCompute && !SystemInfo.supportsComputeShaders)
                throw new NotSupportedException("Image preprocessing requires an actual compute GPU.");

            var factor = Math.Sqrt(645120.0 / ((double)input.width * input.height));
            var width = (int)Math.Floor(factor * input.width / 48) * 48;
            var height = (int)Math.Floor(factor * input.height / 48) * 48;
            if (height == 0) { height = 48; width = Math.Min((int)Math.Floor((double)input.width / input.height) * 48, 13440); }
            else if (width == 0) { width = 48; height = Math.Min((int)Math.Floor((double)input.height / input.width) * 48, 13440); }
            if (width * height > 645120) throw new ArgumentException("Image exceeds the official patch budget.", nameof(input));

            if (Backend == BackendType.CPU) return ProcessCpu(input, width, height);

            if (shader == null)
            {
                var resource = Resources.Load<ComputeShader>("EmbeddingGemmaImagePreprocessor");
                if (resource == null) throw new InvalidOperationException("GPU preprocessing shader is missing from the package.");
                shader = UnityEngine.Object.Instantiate(resource);
            }
            var tensors = new List<Tensor>();
            var buffers = new List<ComputeBuffer>();
            try
            {
                var source = new Tensor<float>(new TensorShape(1, input.height, input.width, 3)); tensors.Add(source);
                var intermediate = new Tensor<int>(new TensorShape(1, input.height, width, 3)); tensors.Add(intermediate);
                var output = new Tensor<float>(new TensorShape(1, 2520, 768)); tensors.Add(output);
                var horizontal = Axis(input.width, width);
                var vertical = Axis(input.height, height);
                ComputeBuffer Buffer(int[] values)
                {
                    var buffer = new ComputeBuffer(values.Length, sizeof(int)); buffers.Add(buffer);
                    buffer.SetData(values); return buffer;
                }
                var xr = Buffer(horizontal.Ranges); var xw = Buffer(horizontal.Weights);
                var yr = Buffer(vertical.Ranges); var yw = Buffer(vertical.Weights);
                var positions = Positions(width, height);
                var patchCount = width / 16 * (height / 16);

                using var commands = new CommandBuffer { name = "EmbeddingGemma image preprocessing" };
                commands.ToTensor(input, source, new TextureTransform().SetTensorLayout(TensorLayout.NHWC)
                    .SetCoordOrigin(CoordOrigin.TopLeft)); // Exact dimensions; Sentis does no resize here.
                var horizontalKernel = shader.FindKernel("Horizontal");
                var patchKernel = shader.FindKernel("VerticalPatches");
                commands.SetComputeIntParam(shader, "SourceWidth", input.width);
                commands.SetComputeIntParam(shader, "SourceHeight", input.height);
                commands.SetComputeIntParam(shader, "TargetWidth", width);
                commands.SetComputeIntParam(shader, "TargetHeight", height);
                commands.SetComputeIntParam(shader, "XStride", horizontal.Stride);
                commands.SetComputeIntParam(shader, "YStride", vertical.Stride);
                commands.SetComputeIntParam(shader, "XPrecision", horizontal.Precision);
                commands.SetComputeIntParam(shader, "YPrecision", vertical.Precision);
                var intermediateData = ComputeTensorData.Pin(intermediate);
                commands.SetComputeBufferParam(shader, horizontalKernel, "Source", ComputeTensorData.Pin(source).buffer);
                commands.SetComputeBufferParam(shader, horizontalKernel, "Intermediate", intermediateData.buffer);
                commands.SetComputeBufferParam(shader, horizontalKernel, "XRanges", xr);
                commands.SetComputeBufferParam(shader, horizontalKernel, "XWeights", xw);
                // D3D / Vulkan dispatch dimensions are limited to 65535 groups.
                // Tile the flattened horizontal pass into rows without changing pixels.
                var groups = (input.height * width * 3 + 255) / 256;
                var groupsX = Math.Min(groups, 65535);
                commands.SetComputeIntParam(shader, "HorizontalDispatchWidth", groupsX * 256);
                commands.DispatchCompute(shader, horizontalKernel, groupsX, (groups + groupsX - 1) / groupsX, 1);
                commands.SetComputeBufferParam(shader, patchKernel, "Intermediate", intermediateData.buffer);
                commands.SetComputeBufferParam(shader, patchKernel, "Pixels", ComputeTensorData.Pin(output).buffer);
                commands.SetComputeBufferParam(shader, patchKernel, "YRanges", yr);
                commands.SetComputeBufferParam(shader, patchKernel, "YWeights", yw);
                commands.DispatchCompute(shader, patchKernel, (2520 * 768 + 255) / 256, 1, 1);
                Graphics.ExecuteCommandBuffer(commands);
                return new ImagePreprocessingResult(output, width, height, patchCount / 9, positions, tensors, buffers);
            }
            catch
            {
                foreach (var tensor in tensors) tensor.Dispose();
                foreach (var buffer in buffers) buffer.Dispose();
                throw;
            }
        }

        static int[] Positions(int width, int height)
        {
            var positions = new int[5040]; Array.Fill(positions, -1);
            for (var patch = 0; patch < width / 16 * (height / 16); patch++)
            { positions[2 * patch] = patch % (width / 16); positions[2 * patch + 1] = patch / (width / 16); }
            return positions;
        }

        static Color32[] CpuColors(Texture input)
        {
            if (input is Texture2D readable && readable.isReadable) return readable.GetPixels32(0);
            if (!SystemInfo.supportsAsyncGPUReadback)
                throw new NotSupportedException("CPU preprocessing of GPU-only inputs requires texture readback.");
            var format = input.graphicsFormat;
            if (!SystemInfo.IsFormatSupported(format, GraphicsFormatUsage.ReadPixels))
            {
                // The source format itself must support readback; requesting another destination format is insufficient.
                var srgb = format == GraphicsFormat.R8G8B8_SRGB || format == GraphicsFormat.R8G8B8A8_SRGB || format == GraphicsFormat.B8G8R8A8_SRGB;
                var converted = new Texture2D(input.width, input.height, TextureFormat.RGBA32, false, !srgb);
                try
                {
                    if (!SystemInfo.IsFormatSupported(converted.graphicsFormat, GraphicsFormatUsage.ReadPixels))
                        throw new NotSupportedException("This GPU has no readable RGBA format for the input encoding.");
                    // ConvertTexture writes GPU storage; never return the destination's unmodified CPU copy.
                    converted.Apply(false, true);
                    if (!Graphics.ConvertTexture(input, converted))
                        throw new NotSupportedException("This GPU cannot convert the input storage to readable RGBA.");
                    return CpuColors(converted);
                }
                finally
                {
                    if (Application.isPlaying) UnityEngine.Object.Destroy(converted);
                    else UnityEngine.Object.DestroyImmediate(converted);
                }
            }
            // Preserve the source transfer encoding during the input readback.
            var request = AsyncGPUReadback.Request(input, 0, format);
            request.WaitForCompletion();
            if (request.hasError) throw new InvalidOperationException("Input texture readback failed.");
            var bytes = request.GetData<byte>();
            var channels = format == GraphicsFormat.R8G8B8_UNorm || format == GraphicsFormat.R8G8B8_SRGB ? 3 : 4;
            var bgra = format == GraphicsFormat.B8G8R8A8_UNorm || format == GraphicsFormat.B8G8R8A8_SRGB;
            var count = input.width * input.height;
            if (bytes.Length != count * channels) throw new InvalidOperationException("Input readback has an unexpected storage size.");
            var colors = new Color32[count];
            for (var pixel = 0; pixel < count; pixel++)
            {
                var offset = pixel * channels;
                colors[pixel] = new Color32(bytes[offset + (bgra ? 2 : 0)], bytes[offset + 1],
                    bytes[offset + (bgra ? 0 : 2)], channels == 4 ? bytes[offset + 3] : (byte)255);
            }
            return colors;
        }

        static ImagePreprocessingResult ProcessCpu(Texture input, int width, int height)
        {
            var colors = CpuColors(input);
            var horizontal = Axis(input.width, width);
            var vertical = Axis(input.height, height);
            // The horizontal pass is clamped to uint8 before the vertical pass, as in torchvision.
            var intermediate = new byte[input.height * width * 3];
            for (var y = 0; y < input.height; y++)
                for (var x = 0; x < width; x++)
                {
                    var minimum = horizontal.Ranges[2 * x]; var count = horizontal.Ranges[2 * x + 1];
                    int red = 1 << (horizontal.Precision - 1), green = red, blue = red;
                    for (var k = 0; k < count; k++)
                    {
                        // Unity's CPU texture rows start at the bottom; official RGB rows start at the top.
                        var color = colors[(input.height - y - 1) * input.width + minimum + k];
                        var weight = horizontal.Weights[x * horizontal.Stride + k];
                        red += color.r * weight; green += color.g * weight; blue += color.b * weight;
                    }
                    var offset = (y * width + x) * 3;
                    intermediate[offset] = (byte)Math.Clamp(red >> horizontal.Precision, 0, 255);
                    intermediate[offset + 1] = (byte)Math.Clamp(green >> horizontal.Precision, 0, 255);
                    intermediate[offset + 2] = (byte)Math.Clamp(blue >> horizontal.Precision, 0, 255);
                }
            var values = new float[2520 * 768];
            var columns = width / 16;
            for (var y = 0; y < height; y++)
                for (var x = 0; x < width; x++)
                {
                    var minimum = vertical.Ranges[2 * y]; var count = vertical.Ranges[2 * y + 1];
                    var patch = y / 16 * columns + x / 16;
                    var output = patch * 768 + (y % 16 * 16 + x % 16) * 3;
                    for (var channel = 0; channel < 3; channel++)
                    {
                        var sum = 1 << (vertical.Precision - 1);
                        for (var k = 0; k < count; k++)
                            sum += intermediate[((minimum + k) * width + x) * 3 + channel] * vertical.Weights[y * vertical.Stride + k];
                        values[output + channel] = Math.Clamp(sum >> vertical.Precision, 0, 255) * (1f / 255f);
                    }
                }
            var pixels = new Tensor<float>(new TensorShape(1, 2520, 768), values);
            return new ImagePreprocessingResult(pixels, width, height, width / 48 * (height / 48),
                Positions(width, height), new List<Tensor> {pixels}, new List<ComputeBuffer>());
        }

        // Geometry-only coefficients shared by CPU and GPU. Matches PyTorch
        // v2.14.1 uint8 bicubic antialias coefficient normalization and signed rounding.
        static (int[] Ranges, int[] Weights, int Stride, int Precision) Axis(int source, int target)
        {
            var scale = (double)source / target;
            var support = 2 * Math.Max(scale, 1);
            var stride = (int)Math.Ceiling(support) * 2 + 1;
            var values = new double[target * stride]; var ranges = new int[target * 2];
            double maximum = 0;
            for (var index = 0; index < target; index++)
            {
                var center = scale * (index + 0.5);
                var minimum = Math.Max((int)(center - support + 0.5), 0);
                var count = Math.Min(Math.Min((int)(center + support + 0.5), source) - minimum, stride);
                ranges[2 * index] = minimum; ranges[2 * index + 1] = count;
                double total = 0;
                for (var k = 0; k < count; k++)
                {
                    var x = Math.Abs((k + minimum - center + 0.5) * (scale >= 1 ? 1 / scale : 1));
                    var weight = x < 1 ? ((1.5 * x - 2.5) * x) * x + 1 :
                        x < 2 ? ((-0.5 * x + 2.5) * x - 4) * x + 2 : 0;
                    values[index * stride + k] = weight; total += weight;
                }
                for (var k = 0; k < count; k++)
                {
                    if (total != 0) values[index * stride + k] /= total;
                    maximum = Math.Max(maximum, values[index * stride + k]);
                }
            }
            var precision = 0;
            for (; precision < 22; precision++)
                if ((int)(0.5 + maximum * (1 << (precision + 1))) >= (1 << 15)) break;
            var weights = new int[values.Length];
            for (var index = 0; index < weights.Length; index++)
            {
                var scaled = values[index] * (1 << precision);
                weights[index] = (int)(scaled < 0 ? scaled - 0.5 : scaled + 0.5);
            }
            return (ranges, weights, stride, precision);
        }

        public void Dispose()
        {
            if (disposed) return;
            disposed = true;
            if (shader != null)
            {
                if (Application.isPlaying) UnityEngine.Object.Destroy(shader);
                else UnityEngine.Object.DestroyImmediate(shader);
                shader = null;
            }
        }
    }

    public sealed class ImagePreprocessingResult : IDisposable
    {
        readonly Tensor<float> pixels;
        readonly int[] positions;
        readonly List<Tensor> tensors;
        readonly List<ComputeBuffer> buffers;
        bool disposed;
        public Tensor<float> Pixels => disposed ? throw new ObjectDisposedException(nameof(ImagePreprocessingResult)) : pixels;
        public int ResizedWidth { get; }
        public int ResizedHeight { get; }
        public int SoftTokenCount { get; }
        public ReadOnlySpan<int> Positions => disposed ? throw new ObjectDisposedException(nameof(ImagePreprocessingResult)) : positions;
        internal ImagePreprocessingResult(Tensor<float> pixels, int width, int height, int softTokens,
            int[] positions, List<Tensor> tensors, List<ComputeBuffer> buffers)
        {
            this.pixels = pixels; this.positions = positions; this.tensors = tensors; this.buffers = buffers;
            ResizedWidth = width; ResizedHeight = height; SoftTokenCount = softTokens;
        }
        public void Dispose()
        {
            if (disposed) return;
            disposed = true;
            foreach (var tensor in tensors) tensor.Dispose();
            foreach (var buffer in buffers) buffer.Dispose();
        }
    }
}
