I'll provide the essential executor files needed:

```kotlin
// runtime/base/ModelExecutor.kt
interface ModelExecutor {
    fun loadModel(config: ModelConfig)
    fun unloadModel()
    suspend fun execute(input: ProcessedData): RawOutput
    val isInitialized: Boolean
}

// runtime/onnx/OnnxRuntimeWrapper.kt
class OnnxRuntimeWrapper @Inject constructor(
    private val context: Context
) : ModelExecutor {
    private var session: OrtSession? = null
    private val environment = OrtEnvironment.getEnvironment()

    override fun loadModel(config: ModelConfig) {
        if (isInitialized) return
        
        val sessionOptions = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(config.deviceConfig.numThreads)
            setOptimizationLevel(OrtSession.GraphOptimizationLevel.ORT_ENABLE_ALL)
            
            if (config.deviceConfig.useGpu) {
                addGPU()
            }
        }

        val modelBytes = context.assets.open(config.modelPath).use { it.readBytes() }
        session = environment.createSession(modelBytes, sessionOptions)
    }

    override fun unloadModel() {
        session?.close()
        session = null
    }

    override suspend fun execute(input: ProcessedData): RawOutput = withContext(Dispatchers.Default) {
        checkNotNull(session) { "Session not initialized" }
        
        val inputTensor = OnnxTensor.createTensor(
            environment,
            input.tensorData,
            input.shape.map { it.toLong() }.toLongArray()
        )

        val output = session!!.run(mapOf("input" to inputTensor))
        val outputData = output[0].value as FloatArray

        RawOutput.AnomalyDetection(
            data = outputData,
            shape = input.shape
        )
    }

    override val isInitialized: Boolean
        get() = session != null
}

// runtime/tflite/TensorFlowRunner.kt
class TensorFlowRunner @Inject constructor(
    private val context: Context
) : ModelExecutor {
    private var interpreter: Interpreter? = null
    private var gpuDelegate: Delegate? = null

    override fun loadModel(config: ModelConfig) {
        if (isInitialized) return
        
        val options = Interpreter.Options().apply {
            setNumThreads(config.deviceConfig.numThreads)
            
            if (config.deviceConfig.useGpu) {
                gpuDelegate = GpuDelegate()
                addDelegate(gpuDelegate)
            }
        }

        val modelBytes = context.assets.open(config.modelPath).use { it.readBytes() }
        interpreter = Interpreter(modelBytes, options)
    }

    override fun unloadModel() {
        interpreter?.close()
        interpreter = null
        gpuDelegate?.close()
        gpuDelegate = null
    }

    override suspend fun execute(input: ProcessedData): RawOutput = withContext(Dispatchers.Default) {
        checkNotNull(interpreter) { "Interpreter not initialized" }
        
        val outputShape = input.shape
        val outputData = FloatArray(outputShape.reduce { acc, i -> acc * i })
        
        interpreter!!.run(input.tensorData, outputData)

        RawOutput.AnomalyDetection(
            data = outputData,
            shape = outputShape
        )
    }

    override val isInitialized: Boolean
        get() = interpreter != null
}

// runtime/onnx/OnnxUtils.kt
object OnnxUtils {
    fun createInputTensor(
        environment: OrtEnvironment,
        data: FloatArray,
        shape: Array<Int>
    ): OnnxTensor {
        return OnnxTensor.createTensor(
            environment,
            data,
            shape.map { it.toLong() }.toLongArray()
        )
    }

    fun processOutput(output: OrtValue): FloatArray {
        return when (val value = output.value) {
            is Array<*> -> value.map { (it as FloatArray)[0] }.toFloatArray()
            is FloatArray -> value
            else -> throw IllegalArgumentException("Unsupported output type")
        }
    }
}

// runtime/tflite/TFLiteUtils.kt
object TFLiteUtils {
    fun createInputArray(data: FloatArray, shape: Array<Int>): Array<FloatArray> {
        val size = shape.reduce { acc, i -> acc * i }
        return Array(1) { data.copyOf(size) }
    }
}

```

Add these dependencies to your build.gradle:

```gradle
dependencies {
    implementation 'org.tensorflow:tensorflow-lite:2.12.0'
    implementation 'org.tensorflow:tensorflow-lite-gpu:2.12.0'
    implementation 'com.microsoft.onnxruntime:onnxruntime-android:1.14.0'
}
```
