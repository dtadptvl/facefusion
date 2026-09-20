import Foundation
import OnnxRuntimeBindings

public enum ORTBridgeError: LocalizedError {
    case runtimeUnavailable
    case sessionCreationFailed(String)
    case inferenceFailed(String)
    case invalidInput(String)
    case invalidOutput(String)

    public var errorDescription: String? {
        switch self {
        case .runtimeUnavailable:
            return "ONNX Runtime framework is unavailable in current target."
        case .sessionCreationFailed(let reason):
            return "Failed to initialize ONNX Runtime session: \(reason)"
        case .inferenceFailed(let reason):
            return "ONNX model inference failed: \(reason)"
        case .invalidInput(let reason):
            return "Invalid tensor input: \(reason)"
        case .invalidOutput(let reason):
            return "Invalid tensor output: \(reason)"
        }
    }
}

/// Generic tensor payload holding raw data, data type, and shape.
public struct TensorBuffer: Sendable {
    public enum DataType: Sendable {
        case float32
        case int64
        case int32
        case uint8
        case double
    }

    public var shape: [Int]
    public var dataType: DataType
    public var floatData: [Float]?
    public var int64Data: [Int64]?
    public var int32Data: [Int32]?
    public var uint8Data: [UInt8]?
    public var doubleData: [Double]?

    public init(floatData: [Float], shape: [Int]) {
        self.shape = shape
        self.dataType = .float32
        self.floatData = floatData
        self.int64Data = nil
        self.int32Data = nil
        self.uint8Data = nil
        self.doubleData = nil
    }

    public init(int64Data: [Int64], shape: [Int]) {
        self.shape = shape
        self.dataType = .int64
        self.floatData = nil
        self.int64Data = int64Data
        self.int32Data = nil
        self.uint8Data = nil
        self.doubleData = nil
    }

    public init(int32Data: [Int32], shape: [Int]) {
        self.shape = shape
        self.dataType = .int32
        self.floatData = nil
        self.int64Data = nil
        self.int32Data = int32Data
        self.uint8Data = nil
        self.doubleData = nil
    }

    public init(uint8Data: [UInt8], shape: [Int]) {
        self.shape = shape
        self.dataType = .uint8
        self.floatData = nil
        self.int64Data = nil
        self.int32Data = nil
        self.uint8Data = uint8Data
        self.doubleData = nil
    }

    public init(doubleData: [Double], shape: [Int]) {
        self.shape = shape
        self.dataType = .double
        self.floatData = nil
        self.int64Data = nil
        self.int32Data = nil
        self.uint8Data = nil
        self.doubleData = doubleData
    }
}

/// Actor enforcing sequential, memory-bounded ONNX Runtime session execution.
/// Guarantees that at most one heavy neural network session resides in memory at a time.
public actor ORTBridge {
    public static let shared = ORTBridge()

    private var currentModelPath: String?
    private var currentUseCoreML: Bool?
    private var env: ORTEnv?
    private var activeSession: ORTSession?

    public init() {
        self.env = try? ORTEnv(loggingLevel: .warning)
    }

    /// Releases any currently allocated session and frees working memory.
    public func releaseSession() {
        self.activeSession = nil
        self.currentModelPath = nil
        self.currentUseCoreML = nil
    }

    /// Queries the model's actual ordered input and output tensor names from the session metadata.
    public func describeNames(modelPath: String, useCoreML: Bool = true) throws -> (inputs: [String], outputs: [String]) {
        let session = try getOrCreateSession(modelPath: modelPath, useCoreML: useCoreML)
        let inputs = try session.inputNames()
        let outputs = try session.outputNames()
        return (inputs: inputs, outputs: outputs)
    }

    /// Prepares or reuses an ORTSession for the model at modelPath.
    private func getOrCreateSession(modelPath: String, useCoreML: Bool = true) throws -> ORTSession {
        if let current = activeSession, currentModelPath == modelPath, currentUseCoreML == useCoreML {
            return current
        }

        activeSession = nil
        currentModelPath = nil
        currentUseCoreML = nil

        let session = try createSession(modelPath: modelPath, useCoreML: useCoreML)
        self.activeSession = session
        self.currentModelPath = modelPath
        self.currentUseCoreML = useCoreML
        return session
    }

    private func createSession(modelPath: String, useCoreML: Bool) throws -> ORTSession {
        guard FileManager.default.fileExists(atPath: modelPath) else {
            throw ORTBridgeError.sessionCreationFailed("Model file does not exist at path: \(modelPath)")
        }

        let ortEnv: ORTEnv
        if let existing = self.env {
            ortEnv = existing
        } else {
            let newEnv = try ORTEnv(loggingLevel: .warning)
            self.env = newEnv
            ortEnv = newEnv
        }

        // Simulator CoreML repeatedly rejects dynamic graph partitions; validate CPU there.
        #if targetEnvironment(simulator)
        let coreMLAvailable = false
        #else
        let coreMLAvailable = ORTIsCoreMLExecutionProviderAvailable()
        #endif
        if useCoreML && coreMLAvailable {
            do {
                let sessionOptions = try ORTSessionOptions()
                try sessionOptions.setIntraOpNumThreads(2)
                try sessionOptions.setGraphOptimizationLevel(.all)
                let coreMLOptions = ORTCoreMLExecutionProviderOptions()
                coreMLOptions.useCPUAndGPU = true
                coreMLOptions.createMLProgram = true
                coreMLOptions.enableOnSubgraphs = true
                try sessionOptions.appendCoreMLExecutionProvider(with: coreMLOptions)
                return try ORTSession(env: ortEnv, modelPath: modelPath, sessionOptions: sessionOptions)
            } catch {
                // Fallback to pure CPU if CoreML initialization or session creation failed
            }
        }

        // CPU Fallback
        let cpuOptions = try ORTSessionOptions()
        try cpuOptions.setIntraOpNumThreads(2)
        try cpuOptions.setGraphOptimizationLevel(.all)
        return try ORTSession(env: ortEnv, modelPath: modelPath, sessionOptions: cpuOptions)
    }

    /// Executes inference with pre-staged tensor inputs and returns output tensors by name.
    public func run(
        modelPath: String,
        inputs: [String: TensorBuffer],
        outputNames: [String]? = nil,
        useCoreML: Bool = true
    ) throws -> [String: TensorBuffer] {
        let session = try getOrCreateSession(modelPath: modelPath, useCoreML: useCoreML)

        var ortInputs: [String: ORTValue] = [:]
        for (name, tensor) in inputs {
            guard tensor.shape.allSatisfy({ $0 > 0 }) else {
                throw ORTBridgeError.invalidInput("Invalid shape \(tensor.shape) for tensor \(name)")
            }
            let expectedElements = tensor.shape.reduce(1, *)
            let shapeNumbers = tensor.shape.map { NSNumber(value: $0) }

            switch tensor.dataType {
            case .float32:
                guard var data = tensor.floatData else { throw ORTBridgeError.invalidInput("Missing float data for \(name)") }
                guard data.count == expectedElements else {
                    throw ORTBridgeError.invalidInput("Float32 data count \(data.count) != shape elements \(expectedElements) for \(name)")
                }
                let length = data.count * MemoryLayout<Float>.stride
                let nsData = NSMutableData(bytes: &data, length: length)
                let ortVal = try ORTValue(tensorData: nsData, elementType: .float, shape: shapeNumbers)
                ortInputs[name] = ortVal

            case .int64:
                guard var data = tensor.int64Data else { throw ORTBridgeError.invalidInput("Missing int64 data for \(name)") }
                guard data.count == expectedElements else {
                    throw ORTBridgeError.invalidInput("Int64 data count \(data.count) != shape elements \(expectedElements) for \(name)")
                }
                let length = data.count * MemoryLayout<Int64>.stride
                let nsData = NSMutableData(bytes: &data, length: length)
                let ortVal = try ORTValue(tensorData: nsData, elementType: .int64, shape: shapeNumbers)
                ortInputs[name] = ortVal

            case .int32:
                guard var data = tensor.int32Data else { throw ORTBridgeError.invalidInput("Missing int32 data for \(name)") }
                guard data.count == expectedElements else {
                    throw ORTBridgeError.invalidInput("Int32 data count \(data.count) != shape elements \(expectedElements) for \(name)")
                }
                let length = data.count * MemoryLayout<Int32>.stride
                let nsData = NSMutableData(bytes: &data, length: length)
                let ortVal = try ORTValue(tensorData: nsData, elementType: .int32, shape: shapeNumbers)
                ortInputs[name] = ortVal

            case .uint8:
                guard var data = tensor.uint8Data else { throw ORTBridgeError.invalidInput("Missing uint8 data for \(name)") }
                guard data.count == expectedElements else {
                    throw ORTBridgeError.invalidInput("UInt8 data count \(data.count) != shape elements \(expectedElements) for \(name)")
                }
                let length = data.count * MemoryLayout<UInt8>.stride
                let nsData = NSMutableData(bytes: &data, length: length)
                let ortVal = try ORTValue(tensorData: nsData, elementType: .uInt8, shape: shapeNumbers)
                ortInputs[name] = ortVal

            case .double:
                // ONNX Runtime Objective-C does not support DOUBLE tensor element data type.
                // Refuse silent cast to prevent unexpected tensor corruption.
                throw ORTBridgeError.invalidInput("Double tensor data type is unsupported by ONNX Runtime Objective-C API for \(name); convert to float32 or use native C API")
            }
        }

        let requestedOutputs: Set<String>
        if let names = outputNames, !names.isEmpty {
            requestedOutputs = Set(names)
        } else {
            let allOutputNames = try session.outputNames()
            requestedOutputs = Set(allOutputNames)
        }

        let ortOutputs = try session.run(withInputs: ortInputs, outputNames: requestedOutputs, runOptions: nil)

        var result: [String: TensorBuffer] = [:]
        for (outName, outVal) in ortOutputs {
            let info = try outVal.tensorTypeAndShapeInfo()
            let shape = info.shape.map { $0.intValue }
            guard shape.allSatisfy({ $0 > 0 }) else {
                throw ORTBridgeError.invalidOutput("Negative or invalid shape \(shape) for \(outName)")
            }
            let totalElements = shape.reduce(1, *)

            guard let tensorData = try? outVal.tensorData() else {
                throw ORTBridgeError.invalidOutput("Could not read tensor data for \(outName)")
            }

            switch info.elementType {
            case .float:
                let expectedBytes = totalElements * MemoryLayout<Float>.stride
                guard tensorData.length == expectedBytes else {
                    throw ORTBridgeError.invalidOutput("Output tensor byte count (\(tensorData.length)) != expected float bytes (\(expectedBytes)) for \(outName)")
                }
                var floatArray = [Float](repeating: 0, count: totalElements)
                _ = floatArray.withUnsafeMutableBytes { destBytes in
                    tensorData.getBytes(destBytes.baseAddress!, length: expectedBytes)
                }
                result[outName] = TensorBuffer(floatData: floatArray, shape: shape)

            case .int64:
                let expectedBytes = totalElements * MemoryLayout<Int64>.stride
                guard tensorData.length == expectedBytes else {
                    throw ORTBridgeError.invalidOutput("Output tensor byte count (\(tensorData.length)) != expected int64 bytes (\(expectedBytes)) for \(outName)")
                }
                var int64Array = [Int64](repeating: 0, count: totalElements)
                _ = int64Array.withUnsafeMutableBytes { destBytes in
                    tensorData.getBytes(destBytes.baseAddress!, length: expectedBytes)
                }
                result[outName] = TensorBuffer(int64Data: int64Array, shape: shape)

            case .int32:
                let expectedBytes = totalElements * MemoryLayout<Int32>.stride
                guard tensorData.length == expectedBytes else {
                    throw ORTBridgeError.invalidOutput("Output tensor byte count (\(tensorData.length)) != expected int32 bytes (\(expectedBytes)) for \(outName)")
                }
                var int32Array = [Int32](repeating: 0, count: totalElements)
                _ = int32Array.withUnsafeMutableBytes { destBytes in
                    tensorData.getBytes(destBytes.baseAddress!, length: expectedBytes)
                }
                result[outName] = TensorBuffer(int32Data: int32Array, shape: shape)

            case .uInt8:
                let expectedBytes = totalElements * MemoryLayout<UInt8>.stride
                guard tensorData.length == expectedBytes else {
                    throw ORTBridgeError.invalidOutput("Output tensor byte count (\(tensorData.length)) != expected uint8 bytes (\(expectedBytes)) for \(outName)")
                }
                var uint8Array = [UInt8](repeating: 0, count: totalElements)
                _ = uint8Array.withUnsafeMutableBytes { destBytes in
                    tensorData.getBytes(destBytes.baseAddress!, length: expectedBytes)
                }
                result[outName] = TensorBuffer(uint8Data: uint8Array, shape: shape)

            default:
                throw ORTBridgeError.invalidOutput("Unsupported output tensor element type (\(info.elementType.rawValue)) for \(outName)")
            }
        }

        return result
    }
}
