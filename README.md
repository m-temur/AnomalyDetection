# AnomalyDetection
I'll share the complete project structure and guide you through the next implementation steps.

Complete Project Structure:
```plaintext
app/
├── src/
│   ├── main/
│   │   ├── java/com/example/aidetection/
│   │   │   ├── core/
│   │   │   │   ├── base/
│   │   │   │   │   ├── BaseModel.kt
│   │   │   │   │   ├── BaseProcessor.kt
│   │   │   │   │   └── BaseExecutor.kt
│   │   │   │   │
│   │   │   │   ├── data/
│   │   │   │   │   ├── DetectionInput.kt
│   │   │   │   │   ├── DetectionResult.kt
│   │   │   │   │   ├── ProcessedData.kt
│   │   │   │   │   ├── RawOutput.kt
│   │   │   │   │   └── ModelConfig.kt
│   │   │   │   │
│   │   │   │   └── domain/
│   │   │   │       ├── DetectedObject.kt
│   │   │   │       ├── Anomaly.kt
│   │   │   │       └── ProcessingMetrics.kt
│   │   │   │
│   │   │   ├── detection/
│   │   │   │   ├── object/
│   │   │   │   │   ├── ObjectDetector.kt
│   │   │   │   │   ├── models/
│   │   │   │   │   │   ├── YoloDetector.kt
│   │   │   │   │   │   └── MobileNetDetector.kt
│   │   │   │   │   │
│   │   │   │   │   └── processor/
│   │   │   │   │       ├── YoloProcessor.kt
│   │   │   │   │       └── MobileNetProcessor.kt
│   │   │   │   │
│   │   │   │   └── anomaly/
│   │   │   │       ├── AnomalyDetector.kt
│   │   │   │       ├── models/
│   │   │   │       │   ├── PadimDetector.kt
│   │   │   │       │   └── PatchCoreDetector.kt
│   │   │   │       │
│   │   │   │       └── processor/
│   │   │   │           ├── PadimProcessor.kt
│   │   │   │           └── PatchCoreProcessor.kt
│   │   │   │
│   │   │   ├── executor/
│   │   │   │   ├── base/
│   │   │   │   │   └── ModelExecutor.kt
│   │   │   │   │
│   │   │   │   ├── onnx/
│   │   │   │   │   ├── OnnxExecutor.kt
│   │   │   │   │   └── OnnxUtils.kt
│   │   │   │   │
│   │   │   │   └── tflite/
│   │   │   │       ├── TfliteExecutor.kt
│   │   │   │       └── TfliteUtils.kt
│   │   │   │
│   │   │   ├── di/
│   │   │   │   ├── AppModule.kt
│   │   │   │   ├── DetectionModule.kt
│   │   │   │   ├── ExecutorModule.kt
│   │   │   │   └── ProcessorModule.kt
│   │   │   │
│   │   │   ├── utils/
│   │   │   │   ├── image/
│   │   │   │   │   ├── BitmapPool.kt
│   │   │   │   │   ├── ImageProcessor.kt
│   │   │   │   │   └── ImageUtils.kt
│   │   │   │   │
│   │   │   │   ├── performance/
│   │   │   │   │   ├── PerformanceMonitor.kt
│   │   │   │   │   └── MetricsCollector.kt
│   │   │   │   │
│   │   │   │   └── extensions/
│   │   │   │       ├── BitmapExt.kt
│   │   │   │       ├── ContextExt.kt
│   │   │   │       └── ImageProxyExt.kt
│   │   │   │
│   │   │   └── ui/
│   │   │       ├── main/
│   │   │       │   ├── MainActivity.kt
│   │   │       │   └── MainViewModel.kt
│   │   │       │
│   │   │       ├── camera/
│   │   │       │   ├── CameraFragment.kt
│   │   │       │   └── CameraViewModel.kt
│   │   │       │
│   │   │       └── detection/
│   │   │           ├── DetectionFragment.kt
│   │   │           └── DetectionViewModel.kt
│   │   │
│   │   ├── cpp/
│   │   │   ├── image_processor/
│   │   │   │   ├── image_processor.cpp
│   │   │   │   └── image_processor.h
│   │   │   │
│   │   │   └── CMakeLists.txt
│   │   │
│   │   ├── res/
│   │   │   ├── layout/
│   │   │   │   ├── activity_main.xml
│   │   │   │   ├── fragment_camera.xml
│   │   │   │   └── fragment_detection.xml
│   │   │   │
│   │   │   ├── values/
│   │   │   │   ├── strings.xml
│   │   │   │   ├── colors.xml
│   │   │   │   └── dimens.xml
│   │   │   │
│   │   │   └── drawable/
│   │   │
│   │   └── assets/
│   │       └── models/
│   │           ├── onnx/
│   │           │   ├── yolo.onnx
│   │           │   └── padim.onnx
│   │           │
│   │           └── tflite/
│   │               └── mobilenet.tflite
│   │
│   └── test/
└── build.gradle
```

Next Steps Implementation Order:

1. First, let's implement the ONNX Executor:

```kotlin
// executor/onnx/OnnxExecutor.kt
class OnnxExecutor @Inject constructor(
    private val context: Context,
    private val performanceMonitor: PerformanceMonitor
) : ModelExecutor {
    private var session: OrtSession? = null
    private val environment = OrtEnvironment.getEnvironment()
    private var gpuDelegate: GpuDelegate? = null

    override fun loadModel(config: ModelConfig) {
        if (isInitialized) return

        val options = SessionOptions().apply {
            setIntraOpNumThreads(config.deviceConfig.numThreads)
            setInterOpNumThreads(config.deviceConfig.numThreads)
            setOptimizationLevel(GraphOptimizationLevel.ORT_ENABLE_ALL)

            if (config.deviceConfig.useGpu) {
                gpuDelegate = config.deviceConfig.gpuDelegate
                addGPU(GPUConfig(gpuDelegate!!))
            }
        }

        val modelBytes = context.assets.open(config.modelPath).use { it.readBytes() }
        session = environment.createSession(modelBytes, options)
    }

    override fun unloadModel() {
        session?.close()
        session = null
        gpuDelegate?.close()
        gpuDelegate = null
    }

    override suspend fun execute(input: ProcessedData): RawOutput =
        withContext(Dispatchers.Default) {
            checkNotNull(session) { "Session not initialized" }

            val inputTensor = OnnxTensor.createTensor(
                environment,
                input.tensorData,
                input.shape.toLongArray()
            )

            val startTime = System.nanoTime()
            val output = session!!.run(
                mapOf("input" to inputTensor),
                setOf("output")
            )
            val inferenceTime = (System.nanoTime() - startTime) / 1_000_000

            performanceMonitor.recordMetric(
                "inference_time",
                inferenceTime,
                MetricType.INFERENCE
            )

            val outputTensor = output[0]
            val outputData = outputTensor.value as Array<*>

            RawOutput.ObjectDetection(
                data = outputData.map { (it as FloatArray)[0] }.toFloatArray(),
                shape = input.shape
            )
        }

    override val isInitialized: Boolean
        get() = session != null
}

// executor/onnx/OnnxUtils.kt
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

    fun processOutput(tensor: OrtValue): FloatArray {
        return when (val value = tensor.value) {
            is Array<*> -> value.map { (it as FloatArray)[0] }.toFloatArray()
            is FloatArray -> value
            else -> throw IllegalArgumentException("Unsupported output type")
        }
    }
}

```

2. Next, let's implement the YOLO Detector:

```kotlin
// detection/object/models/YoloDetector.kt
class YoloDetector @Inject constructor(
    private val executor: ModelExecutor,
    private val processor: YoloProcessor,
    private val performanceMonitor: PerformanceMonitor
) : ObjectDetector {

    private val config = ModelConfig(
        modelPath = "models/yolo.onnx",
        inputShape = arrayOf(1, 3, 640, 640),
        outputShape = arrayOf(1, 85, 8400),
        framework = Framework.ONNX,
        deviceConfig = DeviceConfig(useGpu = true)
    )

    init {
        executor.loadModel(config)
    }

    override suspend fun detect(input: DetectionInput): DetectionResult {
        val startTime = System.nanoTime()
        
        return try {
            // Preprocess
            val processed = processor.preprocess(input)
            val preprocessTime = (System.nanoTime() - startTime) / 1_000_000

            // Run inference
            val output = executor.execute(processed)
            val inferenceTime = (System.nanoTime() - startTime) / 1_000_000 - preprocessTime

            // Postprocess
            val postprocessStart = System.nanoTime()
            val result = processor.postprocess(output, ProcessingConfig(
                targetSize = Size(640, 640),
                normalizationType = NormalizationType.ZERO_TO_ONE
            ))
            val postprocessTime = (System.nanoTime() - postprocessStart) / 1_000_000

            when (result) {
                is DetectionResult.ObjectDetection -> result.copy(
                    metrics = ProcessingMetrics(
                        preprocessTime = preprocessTime,
                        inferenceTime = inferenceTime,
                        postprocessTime = postprocessTime
                    )
                )
                else -> DetectionResult.Error(
                    Exception("Invalid result type"),
                    "Expected object detection result"
                )
            }
        } catch (e: Exception) {
            DetectionResult.Error(e, "Detection failed: ${e.message}")
        }
    }

    override fun release() {
        executor.unloadModel()
    }
}

```

Would you like me to continue with:
1. YOLO Processor implementation
2. TFLite Executor implementation
3. MobileNet Detector implementation
4. UI components implementation

Let me know which part you'd like to see next!
