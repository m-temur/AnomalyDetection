I'll provide the complete capture package files based on the original structure:

```kotlin
// capture/CameraManager.kt
class CameraManager @Inject constructor(
    private val context: Context,
    private val lifecycleOwner: LifecycleOwner
) {
    private var camera: Camera? = null
    private var imageCapture: ImageCapture? = null
    private var imageAnalysis: ImageAnalysis? = null
    private var preview: Preview? = null
    private var cameraProvider: ProcessCameraProvider? = null

    private val mainExecutor = ContextCompat.getMainExecutor(context)
    private val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

    suspend fun initializeCamera(
        previewView: PreviewView,
        detectionMode: DetectionMode,
        captureMode: CaptureMode,
        imageAnalyzer: ImageAnalysis.Analyzer
    ) {
        cameraProvider = ProcessCameraProvider.getInstance(context).await()
        
        setupUseCases(
            previewView,
            detectionMode,
            captureMode,
            imageAnalyzer
        )

        try {
            cameraProvider?.unbindAll()
            camera = cameraProvider?.bindToLifecycle(
                lifecycleOwner,
                cameraSelector,
                preview,
                imageCapture,
                imageAnalysis
            )
        } catch (e: Exception) {
            throw CameraException("Failed to bind camera use cases", e)
        }
    }

    private fun setupUseCases(
        previewView: PreviewView,
        detectionMode: DetectionMode,
        captureMode: CaptureMode,
        imageAnalyzer: ImageAnalysis.Analyzer
    ) {
        val rotation = previewView.display.rotation
        val aspectRatio = aspectRatio(previewView.width, previewView.height)

        preview = Preview.Builder()
            .setTargetAspectRatio(aspectRatio)
            .setTargetRotation(rotation)
            .build()
            .also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }

        imageCapture = ImageCapture.Builder()
            .setCaptureMode(getImageCaptureMode(captureMode))
            .setTargetAspectRatio(aspectRatio)
            .setTargetRotation(rotation)
            .build()

        imageAnalysis = ImageAnalysis.Builder()
            .setTargetAspectRatio(aspectRatio)
            .setTargetRotation(rotation)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()
            .also {
                it.setAnalyzer(
                    Dispatchers.Default.asExecutor(),
                    imageAnalyzer
                )
            }
    }

    fun captureImage(
        file: File,
        onSuccess: (Uri) -> Unit,
        onError: (Exception) -> Unit
    ) {
        val outputOptions = ImageCapture.OutputFileOptions.Builder(file).build()

        imageCapture?.takePicture(
            outputOptions,
            mainExecutor,
            object : ImageCapture.OnImageSavedCallback {
                override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                    output.savedUri?.let { onSuccess(it) }
                }

                override fun onError(exception: ImageCaptureException) {
                    onError(exception)
                }
            }
        )
    }

    fun release() {
        try {
            cameraProvider?.unbindAll()
        } catch (e: Exception) {
            // Handle exception
        } finally {
            camera = null
            imageCapture = null
            imageAnalysis = null
            preview = null
            cameraProvider = null
        }
    }

    private fun aspectRatio(width: Int, height: Int): Int {
        val previewRatio = max(width, height).toDouble() / min(width, height)
        return if (abs(previewRatio - RATIO_4_3_VALUE) <= abs(previewRatio - RATIO_16_9_VALUE)) {
            AspectRatio.RATIO_4_3
        } else {
            AspectRatio.RATIO_16_9
        }
    }

    private fun getImageCaptureMode(captureMode: CaptureMode): Int {
        return when (captureMode) {
            CaptureMode.OPTIMIZE_QUALITY -> ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY
            CaptureMode.OPTIMIZE_LATENCY -> ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY
        }
    }

    companion object {
        private const val RATIO_4_3_VALUE = 4.0 / 3.0
        private const val RATIO_16_9_VALUE = 16.0 / 9.0
    }

    class CameraException(message: String, cause: Throwable? = null) : 
        Exception(message, cause)
}

// capture/CaptureMode.kt
enum class CaptureMode {
    OPTIMIZE_QUALITY,
    OPTIMIZE_LATENCY
}

// capture/DetectionMode.kt
enum class DetectionMode {
    OBJECT_DETECTION {
        override fun getConfig(): DetectionConfig = DetectionConfig(
            modelType = ModelType.YOLO,
            threshold = 0.5f,
            nmsThreshold = 0.45f,
            targetSize = Size(640, 640)
        )
    },
    ANOMALY_DETECTION {
        override fun getConfig(): DetectionConfig = DetectionConfig(
            modelType = ModelType.PADIM,
            threshold = 0.5f,
            targetSize = Size(224, 224)
        )
    };

    abstract fun getConfig(): DetectionConfig
}

// Internal helper classes
data class DetectionConfig(
    val modelType: ModelType,
    val threshold: Float,
    val nmsThreshold: Float = 0.0f,
    val targetSize: Size
)

enum class ModelType {
    YOLO,
    MOBILENET,
    PADIM,
    PATCHCORE
}

```

Required imports:
```kotlin
import android.content.Context
import android.graphics.ImageFormat
import android.net.Uri
import android.util.Size
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.asExecutor
import kotlinx.coroutines.tasks.await
import java.io.File
import java.util.concurrent.Executor
import javax.inject.Inject
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
```

To use this in your app:

```kotlin
class YourFragment : Fragment() {
    @Inject lateinit var cameraManager: CameraManager

    private fun setupCamera() {
        lifecycleScope.launch {
            cameraManager.initializeCamera(
                previewView = binding.previewView,
                detectionMode = DetectionMode.OBJECT_DETECTION,
                captureMode = CaptureMode.OPTIMIZE_LATENCY,
                imageAnalyzer = YourImageAnalyzer()
            )
        }
    }
}
```

This implementation provides:
1. Efficient camera handling
2. Support for both photo capture and real-time analysis
3. Configurable detection modes
4. Performance-optimized image capture
5. Proper lifecycle management

Let me know if you need any clarification or additional functionality!
