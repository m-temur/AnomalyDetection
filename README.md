Here's the ImageProcessor implementation that's needed:

```kotlin
// utils/image/ImageProcessor.kt
class ImageProcessor @Inject constructor(
    private val context: Context,
    private val bitmapPool: BitmapPool
) {
    suspend fun loadBitmapFromUri(uri: Uri): Bitmap = withContext(Dispatchers.IO) {
        context.contentResolver.openInputStream(uri)?.use {
            BitmapFactory.decodeStream(it)
        } ?: throw IllegalStateException("Failed to load bitmap from URI")
    }

    fun loadBitmapFromByteArray(data: ByteArray): Bitmap {
        return BitmapFactory.decodeByteArray(data, 0, data.size)
    }

    fun resizeBitmap(bitmap: Bitmap, targetSize: Size): Bitmap {
        if (bitmap.width == targetSize.width && bitmap.height == targetSize.height) {
            return bitmap
        }

        return Bitmap.createScaledBitmap(
            bitmap,
            targetSize.width,
            targetSize.height,
            true
        )
    }

    fun rotateBitmap(bitmap: Bitmap, rotation: Int): Bitmap {
        if (rotation == 0) return bitmap

        val matrix = Matrix().apply {
            postRotate(rotation.toFloat())
        }

        return Bitmap.createBitmap(
            bitmap,
            0,
            0,
            bitmap.width,
            bitmap.height,
            matrix,
            true
        )
    }

    fun normalizePixel(value: Int, mean: Float = 127.5f, std: Float = 127.5f): Float {
        return (value - mean) / std
    }

    fun preprocessForInference(bitmap: Bitmap, config: ProcessingConfig): FloatArray {
        val pixels = IntArray(bitmap.width * bitmap.height)
        bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        
        val tensorData = FloatArray(bitmap.width * bitmap.height * 3)
        for (i in pixels.indices) {
            val pixel = pixels[i]
            tensorData[i] = normalizePixel(Color.red(pixel))
            tensorData[i + pixels.size] = normalizePixel(Color.green(pixel))
            tensorData[i + 2 * pixels.size] = normalizePixel(Color.blue(pixel))
        }
        return tensorData
    }

    fun convertBitmapToByteBuffer(bitmap: Bitmap, config: ProcessingConfig): ByteBuffer {
        val byteBuffer = ByteBuffer.allocateDirect(
            4 * bitmap.width * bitmap.height * 3
        ).apply {
            order(ByteOrder.nativeOrder())
        }

        val pixels = IntArray(bitmap.width * bitmap.height)
        bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)

        pixels.forEach { pixel ->
            byteBuffer.putFloat(normalizePixel(Color.red(pixel)))
            byteBuffer.putFloat(normalizePixel(Color.green(pixel)))
            byteBuffer.putFloat(normalizePixel(Color.blue(pixel)))
        }

        return byteBuffer
    }
}

// di/ProcessingModule.kt
@Module
@InstallIn(SingletonComponent::class)
object ProcessingModule {
    @Provides
    @Singleton
    fun provideImageProcessor(
        @ApplicationContext context: Context,
        bitmapPool: BitmapPool
    ): ImageProcessor {
        return ImageProcessor(context, bitmapPool)
    }
    
    @Provides
    @Singleton
    fun provideBitmapPool(): BitmapPool {
        return BitmapPool()
    }
}

```
