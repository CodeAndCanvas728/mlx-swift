import Foundation
import MLX
import XCTest

/// Covers the SharpAI SSD expert-streaming ops (`MLXFast.streamedGatherMM`,
/// `MLXFast.preadInto`, `MLXFast.preadIntoOffset`) against a real safetensors file.
final class SSDStreamingTests: XCTestCase {

    private let experts = 4
    private let outDim = 8
    private let inDim = 16

    /// Writes an `[experts, outDim, inDim]` uint32 tensor whose element value is its flat index,
    /// so every expert slab is distinguishable.
    private func makeExpertFile() throws -> (url: URL, full: MLXArray) {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("ssd-streaming-\(UUID().uuidString).safetensors")
        let count = experts * outDim * inDim
        let full = MLXArray(Array(0 ..< UInt32(count)), [experts, outDim, inDim])
        try save(arrays: ["experts": full], url: url, stream: .cpu)
        return (url, full)
    }

    private func wShape() -> MLXArray {
        MLXArray.zeros([experts, outDim, inDim], dtype: .uint32)
    }

    func testStreamedGatherMMReadsExpertSlab() throws {
        let (url, full) = try makeExpertFile()
        defer { try? FileManager.default.removeItem(at: url) }

        let x = MLXArray.zeros([1, 1, inDim])
        let result = MLXFast.streamedGatherMM(
            x: x, wShape: wShape(), activeExpert: 2,
            safetensorsPath: url.path, tensorName: "experts")

        XCTAssertEqual(result.shape, [1, outDim, inDim])
        XCTAssertEqual(result.dtype, .uint32)
        assertEqual(result, full[2 ..< 3])
    }

    /// The op must run on the stream it is given. Swift streams are pooled and
    /// thread-agnostic, so a graph built on one thread has to evaluate on another
    /// (Swift concurrency moves tasks between threads between suspension points).
    func testStreamedGatherMMEvaluatesOnAnotherThread() throws {
        let (url, full) = try makeExpertFile()
        defer { try? FileManager.default.removeItem(at: url) }

        let x = MLXArray.zeros([1, 1, inDim])
        let result = MLXFast.streamedGatherMM(
            x: x, wShape: wShape(), activeExpert: 3,
            safetensorsPath: url.path, tensorName: "experts")

        let done = DispatchSemaphore(value: 0)
        let worker = Thread {
            eval(result)
            done.signal()
        }
        worker.qualityOfService = Thread.current.qualityOfService
        worker.start()
        done.wait()

        assertEqual(result, full[3 ..< 4])
    }

    func testPreadIntoOverwritesWholeArray() throws {
        let (url, full) = try makeExpertFile()
        defer { try? FileManager.default.removeItem(at: url) }

        let dst = MLXArray.zeros([1, outDim, inDim], dtype: .uint32)
        eval(dst)
        let rc = MLXFast.preadInto(
            dst, safetensorsPath: url.path, tensorName: "experts", expertIndex: 1)

        XCTAssertEqual(rc, 0)
        assertEqual(dst, full[1 ..< 2])
    }

    func testPreadIntoOffsetFillsSelectedSlots() throws {
        let (url, full) = try makeExpertFile()
        defer { try? FileManager.default.removeItem(at: url) }

        let bytesPerExpert = outDim * inDim * MemoryLayout<UInt32>.size
        let dst = MLXArray.zeros([2, outDim, inDim], dtype: .uint32)
        eval(dst)

        XCTAssertEqual(
            MLXFast.preadIntoOffset(
                dst, safetensorsPath: url.path, tensorName: "experts",
                expertIndex: 3, dstOffset: 0), 0)
        XCTAssertEqual(
            MLXFast.preadIntoOffset(
                dst, safetensorsPath: url.path, tensorName: "experts",
                expertIndex: 0, dstOffset: bytesPerExpert), 0)

        assertEqual(dst[0 ..< 1], full[3 ..< 4])
        assertEqual(dst[1 ..< 2], full[0 ..< 1])
    }
}
