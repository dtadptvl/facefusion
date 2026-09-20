import Foundation

#if canImport(onnxruntime_objc)
import onnxruntime_objc
#endif

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
        case double
    }

    public var shape: [Int]
    public var dataType: DataType
    public var floatData: [Float]?
    public var int64Data: [Int64]?
    public var doubleData: [Double]?

    public init(floatData: [Float], shape: [Int]) {
        self.shape = shape
        self.dataType = .float32
        self.floatData = floatData
        self.int64Data = nil
        self.doubleData = nil
    }

    public init(int64Data: [Int64], shape: [Int]) {
        self.shape = shape
        self.dataType = .int64
        self.floatData = nil
        self.int64Data = int64Data
        self.doubleData = nil
    }

    public init(doubleData: [Double], shape: [Int]) {
        self.shape = shape
        self.dataType = .double
        self.floatData = nil
        self.int64Data = nil
        self.doubleData = doubleData
    }
}

/// Actor enforcing sequential, memory-bounded ONNX Runtime session execution.
/// Guarantees that at most one heavy neural network session resides in memory at a time.
public actor ORTBridge {
    public static let shared = ORTBridge()

    private var currentModelPath: String?
    #if canImport(onnxruntime_objc)
    private var env: ORTEnv?
    private var activeSession: ORTSession?
    #endif

    public init() {
        #if canImport(onnxruntime_objc)
        self.env = try? ORTEnv(loggingLevel: .warning)
        #endif
    }

    /// Releases any currently allocated session and frees working memory.
    public func releaseSession() {
        #if canImport(onnxruntime_objc)
        self.activeSession = nil
        self.currentModelPath = nil
        #endif
    }

    /// Prepares or reuses an ORTSession for the model at modelPath.
    private func getOrCreateSession(modelPath: String, useCoreML: Bool = true) throws -> Any? {
        #if canImport(onnxruntime_objc)
        if let current = activeSession, currentModelPath == modelPath {
            return current
        }

        activeSession = nil
        currentModelPath = nil

        guard let ortEnv = self.env ?? (try? ORTEnv(loggingLevel: .warning)) else {
            throw ORTBridgeError.runtimeUnavailable
        }
        self.env = ortEnv

        let sessionOptions = try ORTSessionOptions()
        try sessionOptions.setIntraOpNumThreads(2)
        try sessionOptions.setGraphOptimizationLevel(.all)

        if useCoreML {
            // Attempt appending CoreML execution provider where supported
            _ = try? sessionOptions.appendExecutionProvider("coreml", providerOptions: [
                "MLComputeUnits": "CPUAndGPU",
                "ModelFormat": "MLProgram"
            ])
        }

        do {
            let session = try ORTSession(env: ortEnv, modelPath: modelPath, sessionOptions: sessionOptions)
            self.activeSession = session
            self.currentModelPath = modelPath
            return session
        } catch {
            // Fallback to pure CPU if CoreML initialization failed
            let fallbackOptions = try ORTSessionOptions()
            try fallbackOptions.setIntraOpNumThreads(2)
            let cpuSession = try ORTSession(env: ortEnv, modelPath: modelPath, sessionOptions: fallbackOptions)
            self.activeSession = cpuSession
            self.currentModelPath = modelPath
            return cpuSession
        }
        #else
        throw ORTBridgeError.runtimeUnavailable
        #endif
    }

    /// Executes inference with pre-staged tensor inputs and returns output tensors by name.
    public func run(
        modelPath: String,
        inputs: [String: TensorBuffer],
        outputNames: [String]? = nil,
        useCoreML: Bool = true
    ) throws -> [String: TensorBuffer] {
        #if canImport(onnxruntime_objc)
        guard let session = try getOrCreateSession(modelPath: modelPath, useCoreML: useCoreML) as? ORTSession else {
            throw ORTBridgeError.sessionCreationFailed("Session could not be instantiated")
        }

        var ortInputs: [String: ORTValue] = [:]
        for (name, tensor) in inputs {
            let shapeNumbers = tensor.shape.map { NSNumber(value: $0) }
            switch tensor.dataType {
            case .float32:
                guard var data = tensor.floatData else { throw ORTBridgeError.invalidInput("Missing float data for \(name)") }
                let length = data.count * MemoryLayout<Float>.stride
                let nsData = NSMutableData(bytes: &data, length: length)
                let ortVal = try ORTValue(tensorData: nsData, elementType: .float, shape: shapeNumbers)
                ortInputs[name] = ortVal
            case .int64:
                guard var data = tensor.int64Data else { throw ORTBridgeError.invalidInput("Missing int64 data for \(name)") }
                let length = data.count * MemoryLayout<Int64>.stride
                let nsData = NSMutableData(bytes: &data, length: length)
                let ortVal = try ORTValue(tensorData: nsData, elementType: .int64, shape: shapeNumbers)
                ortInputs[name] = ortVal
            case .double:
                guard var data = tensor.doubleData else { throw ORTBridgeError.invalidInput("Missing double data for \(name)") }
                var floatData = data.map { Float($0) }
                let length = floatData.count * MemoryLayout<Float>.stride
                let nsData = NSMutableData(bytes: &floatData, length: length)
                let ortVal = try ORTValue(tensorData: nsData, elementType: .float, shape: shapeNumbers)
                ortInputs[name] = ortVal
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
            let totalElements = shape.reduce(1, *)

            guard let tensorData = try? outVal.tensorData() else {
                throw ORTBridgeError.invalidOutput("Could not read tensor data for \(outName)")
            }

            var floatArray = [Float](repeating: 0, count: totalElements)
            let copyBytes = min(tensorData.length, totalElements * MemoryLayout<Float>.stride)
            _ = floatArray.withUnsafeMutableBytes { destBytes in
                tensorData.getBytes(destBytes.baseAddress!, length: copyBytes)
            }
            result[outName] = TensorBuffer(floatData: floatArray, shape: shape)
        }

        return result
        #else
        throw ORTBridgeError.runtimeUnavailable
        #endif
    }
}
