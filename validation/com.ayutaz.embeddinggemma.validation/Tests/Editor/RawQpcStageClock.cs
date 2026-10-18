using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using Newtonsoft.Json.Linq;

namespace EmbeddingGemma.Validation.Tests
{
    public static class RawQpcStageClock
    {
        [DllImport("kernel32.dll",SetLastError=true)] [return:MarshalAs(UnmanagedType.Bool)]
        static extern bool QueryPerformanceCounter(out long value);
        [DllImport("kernel32.dll",SetLastError=true)] [return:MarshalAs(UnmanagedType.Bool)]
        static extern bool QueryPerformanceFrequency(out long value);

        public static void ValidateBridge(long before,long sample,long after,long nativeFrequency,long runtimeFrequency,bool highResolution)
        {
            if(!highResolution || nativeFrequency<=0 || nativeFrequency!=runtimeFrequency || before<0 || sample<before || after<sample)
                throw new ArgumentException("The stage counter must share the native QPC frequency and epoch bracket.");
        }
        public static long ReadCounter()
        {
            if(Environment.OSVersion.Platform!=PlatformID.Win32NT)throw new PlatformNotSupportedException("Windows QPC capture required.");
            if(!QueryPerformanceCounter(out var counter) || counter<0)throw new InvalidOperationException("Native QPC query failed.");
            return counter;
        }
        public static long ReadFrequency()
        {
            if(Environment.OSVersion.Platform!=PlatformID.Win32NT)throw new PlatformNotSupportedException("Windows QPC capture required.");
            if(!QueryPerformanceFrequency(out var frequency) || frequency<=0)throw new InvalidOperationException("Native QPC frequency failed.");
            return frequency;
        }
        public static JObject CaptureBridge()
        {
            if(Environment.OSVersion.Platform!=PlatformID.Win32NT)throw new PlatformNotSupportedException("Windows QPC capture required.");
            if(!QueryPerformanceFrequency(out var frequency) || !QueryPerformanceCounter(out var before))
                throw new InvalidOperationException("Native QPC query failed.");
            var sample=ReadCounter();
            if(!QueryPerformanceCounter(out var after))throw new InvalidOperationException("Native QPC query failed.");
            var counterFrequency=ReadFrequency();
            ValidateBridge(before,sample,after,frequency,counterFrequency,true);
            return new JObject { ["validated"]=true,["native_qpc_before"]=before,["counter_ticks"]=sample,
                ["native_qpc_after"]=after,["frequency_hz"]=frequency,["native_frequency_hz"]=frequency,
                ["counter_frequency_hz"]=counterFrequency,["high_resolution"]=true,["clock_source"]="QueryPerformanceCounter",
                ["runtime_stopwatch_ticks"]=Stopwatch.GetTimestamp(),["runtime_stopwatch_frequency_hz"]=Stopwatch.Frequency };
        }
    }

    public sealed class RawQpcStageWindow
    {
        readonly Func<long> counter;
        readonly long frequency,begin;
        long end;
        bool attempted,completed;
        public RawQpcStageWindow(Func<long> counter,long frequency)
        {
            this.counter=counter??throw new ArgumentNullException(nameof(counter));
            if(frequency<=0)throw new ArgumentOutOfRangeException(nameof(frequency));
            this.frequency=frequency;begin=counter();
            if(begin<0)throw new ArgumentOutOfRangeException(nameof(counter));
        }
        public static RawQpcStageWindow Start()
        {
            return new RawQpcStageWindow(RawQpcStageClock.ReadCounter,RawQpcStageClock.ReadFrequency());
        }
        public void Finish()
        {
            if(attempted)throw new InvalidOperationException("A completed or faulted stage cannot be finished again.");
            attempted=true;end=counter();
            if(end<begin)throw new InvalidOperationException("QPC stage clock moved backwards.");
            completed=true;
        }
        public JObject ToJson()
        {
            if(!completed)throw new InvalidOperationException("Only a completed ordered stage can be exported.");
            return new JObject { ["timestamp_unit"]="qpc_ticks",["clock_source"]="QueryPerformanceCounter",
                ["frequency_hz"]=frequency,["begin_ticks"]=begin,["end_ticks"]=end };
        }
    }
}
