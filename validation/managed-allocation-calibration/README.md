# Scoped managed allocation calibration

This standalone Editor test imports no model. It checks inclusive main-thread
managed bytes in stopped CPU profiler history (`columnGcMemory`) using one noop,
a 65,536-byte array and a 131,072-byte array. The expected payload delta is exactly
65,536 bytes. Array headers remain visible; native memory and other threads are
outside the scope.

Verified environment: Unity 6000.3.21f1, Test Framework 1.6.0 and Newtonsoft JSON
3.2.2. Copy this directory into an owned consumer's `Assets/Editor/AllocationCalibration`.
Set `EG_ALLOCATION_CALIBRATION_REPORT` to a new absolute JSON output filename, then run
the Editor with:

```text
Unity.exe -force-d3d12 -job-worker-count 4 -projectPath <consumer> -runTests -testPlatform EditMode -testFilter OwnedAllocation.Tests.ManagedAllocationCalibrationTests.KnownByteArrayAllocationsHaveDistinctScopedByteTotals -assemblyNames OwnedAllocation.Tests -testResults <new-results.xml> -logFile <new-log.txt>
```

Use interactive Editor mode for this measurement. In the verified environment,
`-batchmode -nographics` returned no profiler frames even though PlayMode frames
advanced. Missing frames/markers fail the test; they never establish zero allocation.
Profiler flags and background execution are restored afterwards. No SDK behavior,
inference arithmetic or model data is modified.

The prior `GC.GetAllocatedBytesForCurrentThread` probe returned 0 for both known
allocations and failed the same numerical allocation contract. The passing test
recorded 0 / 65,568 / 131,104 bytes and normal exit0. See
[the fixed audit](../../docs/results/managed-allocation-scoped-byte-calibration-20261018.json).
