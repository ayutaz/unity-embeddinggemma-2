# Windows DirectX event observer

This diagnostic tool records a selected, already running process. It does not run
inference or establish model/GPU acceptance. The SDK and its scheduling are unchanged.

The decoder dependency is Microsoft/DxTimingCaptureLibrary, MIT, fixed commit
`72bd2b18553493aa9850e434caf71bfa18952aac`. Keep its source and generated binaries
outside tracked source. Windows SDK10.0.26100.0 and MSVC v143 are used. The dependency's
normal build restores the pinned D3D12/DirectStorage SDK packages. It uses a stub PIX
decoder: PIX marker contents are not decoded.

Build the dependency's `third_party/googletest/googletest_main.vcxproj` and then
`OwnedGpuRecordsTests.vcxproj` and `NativeDxgObserver.vcxproj` with MSBuild, Release/x64,
`/restore`, `/p:DxTimingCaptureLibraryRoot=<external dependency directory>` and
`/p:NativeDxgBuildRoot=<ignored output directory>`. Both directories must be absolute.
Run `tests/OwnedGpuRecordsTests.exe` from the build output to verify the recorder,
callback bridge, and owned-session cleanup policy. The dedicated Actions workflow
builds this fixed source and runs40 native unit cases and8 preflight CLI cases.
Those jobs do not collect GPU events; their success is not a GPU pass.

```text
NativeDxgObserver.exe PID seconds output.jsonl
```

The target must already exist and be accessible. Duration is1..900 seconds. The output
must not exist; it is created exclusively. A successful startup prints and flushes
`capture_ready` to stdout after the trace consumer and provider rundown are ready.
A controller may launch an owned workload suspended, start this observer with its PID,
and resume the workload only after that signal. The process handle and creation
FILETIME identify the original target; the observer never reopens a recycled PID.

The capture ends at its duration or500ms after its target exits. Only the successfully
started trace handle is closed. The fixed descriptive session name is
`EmbeddingGemmaNativeDxgCapture`; an already existing session makes startup fail and
is never stopped by this invocation. Use an external bounded supervisor for process
timeouts, inspect the footer/start status before any owned-session cleanup, and keep
failed or incomplete receipts. A killed observer with no footer is not a success.

Requested ETW buffering is4..256 buffers of64KiB, at most16MiB. Record storage is capped
at200000 rows and4096 concurrent execution/resource entries, with64 stored diagnostic
codes plus the full diagnostic count. Reaching a cap invalidates the capture. Windows
and the decoder's own allocations are separate from these configured caps.

Kernel events must be processed to resolve their ownership. Only records attributed to
the chosen PID are exported. Other process payloads, object names and diagnostic message
text are not exported. Warning/error codes, loss, decoder exceptions, invalid intervals,
unfinished owned executions, stop failures and write failures are retained or cause a
nonzero exit. One such incomplete capture cannot certify complete stage coverage.

The first/last records carry raw QPC ticks, QPC frequency, bracketing QPC samples around
precise UTC FILETIME, and the initial target creation FILETIME. Data timestamps retain
the QPC epoch in integer nanoseconds. They are not rebased to a display clock. Missing
timestamp/size fields are JSON null. `gpu_work` records retain command-list/marker IDs;
`packet_envelope` records retain engine affinity and hardware-scheduling state. Packet
envelopes include time waiting after driver submission; they are not pure GPU execution.
GPU work spans can overlap and must not be summed across queues/markers without an
explicit aggregation scope. Resource destruction is an API event, not proof of driver
retirement or VRAM return. Resource `bytes` is the logical buffer Width; texture and
unknown sizes are null, and it is not an allocated-heap or residency byte count.

`capture_valid` describes the observer protocol, clocks/loss bookkeeping, and closure.
Even exit0 with an empty target sets `gpu_control_validated` and `real_model_validated`
false. Independently check positive work, ownership, external clock anchors, reference
vectors and normal workload exit before accepting a particular GPU/model experiment.
