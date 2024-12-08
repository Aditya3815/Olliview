package com.example.objectdetectionapp

import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.*
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.view.Surface
import android.view.TextureView
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.Toolbar
import androidx.core.content.ContextCompat
import com.example.objectdetectionapp.ml.SsdMobilenetV11Metadata1
import com.google.android.material.floatingactionbutton.FloatingActionButton
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp

class MainActivity : AppCompatActivity() {

    lateinit var labels: List<String>
    val colors = listOf(
        Color.BLUE, Color.GREEN, Color.RED, Color.CYAN, Color.GRAY, Color.BLACK,
        Color.DKGRAY, Color.MAGENTA, Color.YELLOW
    )
    val paint = Paint()
    lateinit var bitmap: Bitmap
    lateinit var cameraDevice: CameraDevice
    lateinit var handler: Handler
    lateinit var cameraManager: CameraManager
    lateinit var textureView: TextureView
    lateinit var model: SsdMobilenetV11Metadata1
    lateinit var imageProcessor: ImageProcessor
    lateinit var imageView: ImageView
    lateinit var statusTextView: TextView
    lateinit var captureButton: FloatingActionButton
    lateinit var toolbar: Toolbar

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Initialize UI elements
        toolbar = findViewById(R.id.toolbar)
        setSupportActionBar(toolbar) // Use the custom Toolbar as the ActionBar

        statusTextView = findViewById(R.id.statusTextView)
        imageView = findViewById(R.id.imageView)
        textureView = findViewById(R.id.textureview)
        captureButton = findViewById(R.id.captureButton)

        captureButton.setOnClickListener {
            captureAndProcessImage()
        }

        get_permission()

        labels = FileUtil.loadLabels(this, "mobilenet_objectdetection_labels.txt")
        imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(300, 300, ResizeOp.ResizeMethod.BILINEAR))
            .build()
        model = SsdMobilenetV11Metadata1.newInstance(this)

        val handlerThread = HandlerThread("videothread")
        handlerThread.start()
        handler = Handler(handlerThread.looper)

        statusTextView.text = "Initializing Camera..."

        textureView.surfaceTextureListener = object : TextureView.SurfaceTextureListener {
            override fun onSurfaceTextureAvailable(
                surface: android.graphics.SurfaceTexture,
                width: Int,
                height: Int
            ) {
                open_camera()
            }

            override fun onSurfaceTextureSizeChanged(
                surface: android.graphics.SurfaceTexture,
                width: Int,
                height: Int
            ) {}

            override fun onSurfaceTextureDestroyed(surface: android.graphics.SurfaceTexture): Boolean = false

            override fun onSurfaceTextureUpdated(surface: android.graphics.SurfaceTexture) {
                // No continuous detection, only capture on button press
            }
        }

        cameraManager = getSystemService(Context.CAMERA_SERVICE) as CameraManager
    }

    private fun captureAndProcessImage() {
        statusTextView.text = "Capturing Image..."
        textureView.bitmap?.let { bmp ->
            bitmap = bmp
            processFrame()
        }
    }

    private fun processFrame() {
        val image = imageProcessor.process(TensorImage.fromBitmap(bitmap))
        val outputs = model.process(image)
        val locations = outputs.locationsAsTensorBuffer.floatArray
        val classes = outputs.classesAsTensorBuffer.floatArray
        val scores = outputs.scoresAsTensorBuffer.floatArray

        val mutable = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(mutable)
        val h = mutable.height
        val w = mutable.width

        paint.textSize = h / 15f
        paint.strokeWidth = h / 85f

        scores.forEachIndexed { index, fl ->
            if (fl > 0.5) {
                val x = index * 4
                val labelIndex = classes[index].toInt()
                if (labelIndex in labels.indices) {
                    paint.color = colors[index % colors.size]
                    paint.style = Paint.Style.STROKE
                    canvas.drawRect(
                        RectF(
                            locations[x + 1] * w, locations[x] * h,
                            locations[x + 3] * w, locations[x + 2] * h
                        ), paint
                    )

                    paint.style = Paint.Style.FILL
                    canvas.drawText(
                        "${labels[labelIndex]} $fl",
                        locations[x + 1] * w, locations[x] * h, paint
                    )
                }
            }
        }

        imageView.setImageBitmap(mutable)
        statusTextView.text = "Detection Complete"
    }

    @SuppressLint("MissingPermission")
    private fun open_camera() {
        statusTextView.text = "Opening Camera..."
        cameraManager.openCamera(cameraManager.cameraIdList[0], object : CameraDevice.StateCallback() {
            override fun onOpened(camera: CameraDevice) {
                cameraDevice = camera
                val surfaceTexture = textureView.surfaceTexture
                val surface = Surface(surfaceTexture)
                val cap = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
                cap.addTarget(surface)

                cameraDevice.createCaptureSession(
                    listOf(surface),
                    object : CameraCaptureSession.StateCallback() {
                        override fun onConfigured(session: CameraCaptureSession) {
                            session.setRepeatingRequest(cap.build(), null, handler)
                            statusTextView.text = "Camera Ready"
                        }

                        override fun onConfigureFailed(session: CameraCaptureSession) {
                            statusTextView.text = "Camera Configuration Failed"
                        }
                    }, handler
                )
            }

            override fun onDisconnected(camera: CameraDevice) {
                statusTextView.text = "Camera Disconnected"
            }

            override fun onError(camera: CameraDevice, error: Int) {
                statusTextView.text = "Camera Error: $error"
            }
        }, handler)
    }

    private fun get_permission() {
        if (ContextCompat.checkSelfPermission(
                this,
                android.Manifest.permission.CAMERA
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            requestPermissions(arrayOf(android.Manifest.permission.CAMERA), 101)
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (grantResults.isNotEmpty() && grantResults[0] != PackageManager.PERMISSION_GRANTED) {
            get_permission()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        if (::model.isInitialized) {
            model.close()
        }
    }
}
