import Foundation
import Metal

// This is a capability probe, not a Sentis/model benchmark or acceptance result.
func probe() -> [String: Any] {
    func failure(_ reason: String, available: Bool) -> [String: Any] {
        return ["schema": 1, "available": available, "compute_passed": false, "reason": reason]
    }
    guard let device = MTLCreateSystemDefaultDevice() else {
        return failure("no_metal_device", available: false)
    }
    do {
        let source = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void capability_probe(device uint* output [[buffer(0)]], uint i [[thread_position_in_grid]]) {
            if (i < 16) output[i] = i * 3 + 7;
        }
        """
        let library = try device.makeLibrary(source: source, options: nil)
        guard let function = library.makeFunction(name: "capability_probe"),
              let output = device.makeBuffer(length: 16 * MemoryLayout<UInt32>.stride, options: .storageModeShared),
              let queue = device.makeCommandQueue(), let buffer = queue.makeCommandBuffer(),
              let encoder = buffer.makeComputeCommandEncoder() else {
            return failure("probe_setup_failed", available: true)
        }
        let pipeline = try device.makeComputePipelineState(function: function)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(output, offset: 0, index: 0)
        encoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 16, height: 1, depth: 1))
        encoder.endEncoding()
        buffer.commit()
        buffer.waitUntilCompleted()
        guard buffer.status == .completed else {
            return failure("gpu_command_failed", available: true)
        }
        let values = output.contents().bindMemory(to: UInt32.self, capacity: 16)
        let mismatches = (0..<16).filter { values[$0] != UInt32($0 * 3 + 7) }.count
        guard mismatches == 0 else { return failure("compute_mismatch", available: true) }
        return ["schema": 1, "available": true, "compute_passed": true, "compared": 16, "mismatches": 0,
                "max_buffer_bytes": device.maxBufferLength,
                "recommended_working_set_bytes": device.recommendedMaxWorkingSetSize]
    } catch {
        return failure("probe_setup_failed", available: true)
    }
}

if CommandLine.arguments.count != 2 { exit(1) }
do {
    let data = try JSONSerialization.data(withJSONObject: probe(), options: [.sortedKeys])
    try data.write(to: URL(fileURLWithPath: CommandLine.arguments[1]), options: [.atomic])
} catch { exit(1) }
