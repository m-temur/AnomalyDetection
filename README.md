Let me provide both the BaseProcessor for anomaly and fix the ModelExecutor issue:

```kotlin
// detection/anomaly/base/AnomalyBaseProcessor.kt
interface AnomalyBaseProcessor {
    suspend fun preprocess(input: DetectionInput): ProcessedData
    suspend fun postprocess(output: RawOutput, config: ProcessingConfig): DetectionResult
    fun calculateAnomalyScore(features: FloatArray): Float
    fun generateAnomalyMask(scores: FloatArray, width: Int, height: Int): Bitmap?
}

// detection/anomaly/base/AnomalyProcessingConfig.kt
data class AnomalyProcessingConfig(
    val threshold: Float = 0.5f,
    val maskGenerationEnabled: Boolean = true,
    val featureAggregation: FeatureAggregationType = FeatureAggregationType.MEAN
)

enum class FeatureAggregationType {
    MEAN,
    MAX,
    WEIGHTED
}

```

For the ModelExecutor error, here's the interface and implementation:

```kotlin
// executor/base/ModelExecutor.kt
interface ModelExecutor {
    fun loadModel(config: ModelConfig)
    fun unloadModel()
    suspend fun execute(input: ProcessedData): RawOutput
    val isInitialized: Boolean
}

// Add this to your di/ExecutorModule.kt
@Module
@InstallIn(SingletonComponent::class)
object ExecutorModule {
    @Provides
    @Singleton
    fun provideModelExecutor(
        @ApplicationContext context: Context,
        performanceMonitor: PerformanceMonitor
    ): ModelExecutor {
        return OnnxExecutor(context, performanceMonitor)
    }
}

```

Update your AnomalyDetector implementations to use these new components.
