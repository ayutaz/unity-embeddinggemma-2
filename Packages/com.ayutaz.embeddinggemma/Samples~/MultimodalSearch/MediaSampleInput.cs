using System;
using System.IO;
using System.Text;

namespace EmbeddingGemma.Samples
{
    /// <summary>Mono 16 kHz WAV (PCM16 / IEEE float32) or little-endian raw float32. Never resamples, clips or truncates.</summary>
    public static class MediaSampleInput
    {
        public static AudioPcmInput ReadPcm(string path)
        {
            if (string.Equals(Path.GetExtension(path), ".f32", StringComparison.OrdinalIgnoreCase))
            {
                var bytes = File.ReadAllBytes(path);
                if (bytes.Length % 4 != 0 || bytes.Length > AudioPcmInput.ProfileSampleCount * 4)
                    throw new InvalidDataException("Raw PCM must contain complete float32 mono samples, at most 11.2 seconds.");
                if (!BitConverter.IsLittleEndian) for (var index = 0; index < bytes.Length; index += 4) Array.Reverse(bytes, index, 4);
                var values = new float[bytes.Length / 4]; Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length); return new AudioPcmInput(values);
            }
            if (!string.Equals(Path.GetExtension(path), ".wav", StringComparison.OrdinalIgnoreCase))
                throw new ArgumentException("Choose a .wav or little-endian mono 16 kHz .f32 file.", nameof(path));
            using var stream = File.OpenRead(path); using var reader = new BinaryReader(stream);
            string Tag() => Encoding.ASCII.GetString(reader.ReadBytes(4));
            if (stream.Length < 12 || Tag() != "RIFF") throw new InvalidDataException("A RIFF WAVE file is required.");
            var end = 8L + reader.ReadUInt32();
            if (end > stream.Length || Tag() != "WAVE") throw new InvalidDataException("Truncated or invalid WAVE file.");
            int format = 0, channels = 0, rate = 0, bits = 0, alignment = 0; long dataOffset = -1, dataBytes = 0;
            while (stream.Position + 8 <= end)
            {
                var tag = Tag(); var length = reader.ReadUInt32(); var next = stream.Position + length;
                if (next > end) throw new InvalidDataException("Truncated WAVE chunk.");
                if (tag == "fmt ")
                {
                    if (length < 16) throw new InvalidDataException("Incomplete WAVE format.");
                    format = reader.ReadUInt16(); channels = reader.ReadUInt16(); rate = reader.ReadInt32(); reader.ReadUInt32(); alignment = reader.ReadUInt16(); bits = reader.ReadUInt16();
                }
                if (tag == "data") { if (dataOffset >= 0) throw new InvalidDataException("Use one WAVE data chunk."); dataOffset = stream.Position; dataBytes = length; }
                stream.Position = next + (length & 1);
            }
            if (channels != 1 || rate != 16000 || !(format == 1 && bits == 16 || format == 3 && bits == 32))
                throw new ArgumentException("Expected mono 16000 Hz PCM16 or IEEE float32 WAVE; convert unsupported input explicitly.");
            if (dataOffset < 0 || alignment != bits / 8 || dataBytes % alignment != 0 || dataBytes / alignment > AudioPcmInput.ProfileSampleCount)
                throw new InvalidDataException("Invalid WAVE sample data or duration; input is never truncated.");
            var samples = new float[dataBytes / alignment]; stream.Position = dataOffset;
            for (var index = 0; index < samples.Length; index++) samples[index] = format == 3 ? reader.ReadSingle() : reader.ReadInt16() / 32768f;
            return new AudioPcmInput(samples);
        }
    }
}
