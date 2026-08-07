package com.example.child_growth_monitor_app.ar

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.opengl.GLES11Ext
import android.opengl.GLES20
import android.opengl.GLSurfaceView
import android.os.Bundle
import android.view.Gravity
import android.view.View
import android.widget.FrameLayout
import android.widget.TextView
import com.google.ar.core.ArCoreApk
import com.google.ar.core.Config
import com.google.ar.core.Coordinates2d
import com.google.ar.core.Frame
import com.google.ar.core.Plane
import com.google.ar.core.Session
import com.google.ar.core.TrackingState
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.roundToInt

/**
 * Resource-bounded ARCore depth sampler.
 *
 * It deliberately avoids RGB recording, dense point clouds, meshes, and a
 * frame queue. At most one depth image is acquired per render tick and only a
 * sparse center-body grid contributes to a maximum of eight keyframes.
 * Results are experimental and never clinical-measurement eligible.
 */
class SparseArScanActivity : Activity(), GLSurfaceView.Renderer {
    private lateinit var surface: GLSurfaceView
    private lateinit var instruction: TextView
    private var session: Session? = null
    private val finished = AtomicBoolean(false)
    private val heightCandidatesCm = mutableListOf<Double>()
    private var acceptedDepthSamples = 0
    private var attemptedDepthSamples = 0
    private var attemptedGridPoints = 0
    private var lastSampleNs = 0L
    private var textureId = 0
    private var installRequested = false
    private var program = 0
    private var positionBuffer: FloatBuffer? = null
    private var texBuffer: FloatBuffer? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val root = FrameLayout(this)
        surface = GLSurfaceView(this).apply {
            setEGLContextClientVersion(2)
            preserveEGLContextOnPause = true
            setRenderer(this@SparseArScanActivity)
            renderMode = GLSurfaceView.RENDERMODE_CONTINUOUSLY
        }
        instruction = TextView(this).apply {
            text = "Initializing depth scan…"
            textSize = 17f
            setTextColor(0xffffffff.toInt())
            setBackgroundColor(0x99000000.toInt())
            gravity = Gravity.CENTER
            setPadding(24, 18, 24, 18)
        }
        root.addView(surface)
        root.addView(
            instruction,
            FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.WRAP_CONTENT,
                Gravity.BOTTOM,
            ),
        )
        setContentView(root)
        ensureCameraPermission()
    }

    private fun ensureCameraPermission() {
        if (checkSelfPermission(Manifest.permission.CAMERA) ==
            PackageManager.PERMISSION_GRANTED
        ) {
            initializeSession()
        } else {
            requestPermissions(arrayOf(Manifest.permission.CAMERA), CAMERA_REQUEST)
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray,
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == CAMERA_REQUEST && grantResults.firstOrNull() == PackageManager.PERMISSION_GRANTED) {
            initializeSession()
        } else {
            finishCancelled("Camera permission is required")
        }
    }

    private fun initializeSession() {
        try {
            when (ArCoreApk.getInstance().requestInstall(this, !installRequested)) {
                ArCoreApk.InstallStatus.INSTALL_REQUESTED -> {
                    installRequested = true
                    instruction.text = "Finish installing Google Play Services for AR, then retry."
                    return
                }
                ArCoreApk.InstallStatus.INSTALLED -> Unit
            }
            val next = Session(this)
            if (!next.isDepthModeSupported(Config.DepthMode.AUTOMATIC)) {
                finishCancelled("Depth is not supported on this phone")
                return
            }
            val config = next.config.apply {
                depthMode = Config.DepthMode.AUTOMATIC
                planeFindingMode = Config.PlaneFindingMode.HORIZONTAL
                updateMode = Config.UpdateMode.LATEST_CAMERA_IMAGE
            }
            next.configure(config)
            session = next
            if (textureId != 0) next.setCameraTextureName(textureId)
            next.resume()
            instruction.text = "Keep the whole child centered. Move slowly in a short arc."
        } catch (error: Exception) {
            finishCancelled("AR depth could not start: ${error.javaClass.simpleName}")
        }
    }

    override fun onResume() {
        super.onResume()
        surface.onResume()
        if (session == null && checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            initializeSession()
        }
        try {
            session?.resume()
        } catch (_: Exception) {
            finishCancelled("AR session could not resume")
        }
    }

    override fun onPause() {
        session?.pause()
        surface.onPause()
        super.onPause()
    }

    override fun onDestroy() {
        session?.close()
        session = null
        super.onDestroy()
    }

    override fun onSurfaceCreated(gl: javax.microedition.khronos.opengles.GL10?, config: javax.microedition.khronos.egl.EGLConfig?) {
        textureId = createExternalTexture()
        program = createProgram(VERTEX_SHADER, FRAGMENT_SHADER)
        positionBuffer = floatBuffer(floatArrayOf(-1f, -1f, 1f, -1f, -1f, 1f, 1f, 1f))
        texBuffer = floatBuffer(floatArrayOf(0f, 1f, 1f, 1f, 0f, 0f, 1f, 0f))
        session?.setCameraTextureName(textureId)
        GLES20.glClearColor(0f, 0f, 0f, 1f)
    }

    override fun onSurfaceChanged(gl: javax.microedition.khronos.opengles.GL10?, width: Int, height: Int) {
        GLES20.glViewport(0, 0, width, height)
        session?.setDisplayGeometry(display.rotation, width, height)
    }

    override fun onDrawFrame(gl: javax.microedition.khronos.opengles.GL10?) {
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT)
        val active = session ?: return
        val frame = try {
            active.setCameraTextureName(textureId)
            active.update()
        } catch (_: Exception) {
            return
        }
        drawCamera(frame)
        if (frame.camera.trackingState != TrackingState.TRACKING) {
            showInstruction("Move the phone slowly until tracking starts")
            return
        }
        val now = frame.timestamp
        if (heightCandidatesCm.size >= MAX_KEYFRAMES) {
            finishWithResult()
            return
        }
        if (now - lastSampleNs < SAMPLE_INTERVAL_NS) return
        lastSampleNs = now
        sampleSparseHeight(active, frame)
    }

    private fun sampleSparseHeight(active: Session, frame: Frame) {
        attemptedDepthSamples++
        val floor = active.getAllTrackables(Plane::class.java)
            .filter { it.trackingState == TrackingState.TRACKING && it.type == Plane.Type.HORIZONTAL_UPWARD_FACING }
            .minByOrNull { kotlin.math.abs(frame.camera.pose.ty() - it.centerPose.ty()) }
        if (floor == null) {
            showInstruction("Point at the floor near the child's feet")
            return
        }
        val image = try {
            frame.acquireDepthImage16Bits()
        } catch (_: Exception) {
            showInstruction("Move slowly while depth becomes available")
            return
        }
        try {
            val plane = image.planes[0]
            val buffer = plane.buffer.order(ByteOrder.nativeOrder())
            val intrinsics = frame.camera.imageIntrinsics
            val dims = intrinsics.imageDimensions
            val fx = intrinsics.focalLength[0] * image.width / dims[0].toFloat()
            val fy = intrinsics.focalLength[1] * image.height / dims[1].toFloat()
            val cx = intrinsics.principalPoint[0] * image.width / dims[0].toFloat()
            val cy = intrinsics.principalPoint[1] * image.height / dims[1].toFloat()
            val heights = ArrayList<Double>(512)
            var valid = 0
            var total = 0
            val startX = (image.width * 0.25).roundToInt()
            val endX = (image.width * 0.75).roundToInt()
            val startY = (image.height * 0.08).roundToInt()
            val endY = (image.height * 0.95).roundToInt()
            for (y in startY until endY step GRID_STEP) {
                for (x in startX until endX step GRID_STEP) {
                    total++
                    val offset = y * plane.rowStride + x * plane.pixelStride
                    if (offset + 1 >= buffer.limit()) continue
                    val depthMm = buffer.getShort(offset).toInt() and 0x1fff
                    if (depthMm !in MIN_DEPTH_MM..MAX_DEPTH_MM) continue
                    valid++
                    val z = -depthMm / 1000f
                    val cameraPoint = floatArrayOf(
                        (x - cx) * -z / fx,
                        (y - cy) * z / fy,
                        z,
                    )
                    val world = frame.camera.pose.transformPoint(cameraPoint)
                    val height = (world[1] - floor.centerPose.ty()).toDouble()
                    if (height in MIN_BODY_HEIGHT_M..MAX_BODY_HEIGHT_M) heights.add(height)
                }
            }
            attemptedGridPoints += total
            if (total > 0) acceptedDepthSamples += valid
            if (heights.size < MIN_VALID_POINTS) {
                showInstruction("Center the complete child and keep the feet visible")
                return
            }
            heights.sort()
            val candidateCm = heights[(heights.size * 0.98).toInt().coerceAtMost(heights.lastIndex)] * 100.0
            heightCandidatesCm.add(candidateCm)
            showInstruction("Good depth ${heightCandidatesCm.size}/$MAX_KEYFRAMES — continue moving slowly")
        } finally {
            image.close()
        }
    }

    private fun finishWithResult() {
        if (!finished.compareAndSet(false, true)) return
        val sorted = heightCandidatesCm.sorted()
        val median = sorted[sorted.size / 2]
        val deviations = sorted.map { kotlin.math.abs(it - median) }.sorted()
        val uncertainty = deviations[deviations.size / 2].coerceAtLeast(0.1)
        val validFraction = if (attemptedDepthSamples == 0) 0.0 else {
            (acceptedDepthSamples.toDouble() / attemptedGridPoints.coerceAtLeast(1)).coerceIn(0.0, 1.0)
        }
        setResult(
            RESULT_OK,
            Intent().apply {
                putExtra("estimatedHeightCm", median)
                putExtra("uncertaintyCm", uncertainty)
                putExtra("acceptedKeyframes", sorted.size)
                putExtra("validDepthFraction", validFraction)
                putExtra("depthMode", "automatic")
            },
        )
        runOnUiThread { finish() }
    }

    private fun finishCancelled(message: String) {
        if (!finished.compareAndSet(false, true)) return
        runOnUiThread {
            setResult(RESULT_CANCELED, Intent().putExtra("reason", message))
            finish()
        }
    }

    private fun showInstruction(message: String) = runOnUiThread { instruction.text = message }

    private fun drawCamera(frame: Frame) {
        if (frame.hasDisplayGeometryChanged()) {
            val output = ByteBuffer.allocateDirect(8 * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
            frame.transformCoordinates2d(Coordinates2d.OPENGL_NORMALIZED_DEVICE_COORDINATES, positionBuffer!!, Coordinates2d.TEXTURE_NORMALIZED, output)
            texBuffer = output
        }
        GLES20.glUseProgram(program)
        val pos = GLES20.glGetAttribLocation(program, "aPosition")
        val tex = GLES20.glGetAttribLocation(program, "aTexCoord")
        GLES20.glEnableVertexAttribArray(pos)
        GLES20.glVertexAttribPointer(pos, 2, GLES20.GL_FLOAT, false, 0, positionBuffer!!)
        GLES20.glEnableVertexAttribArray(tex)
        GLES20.glVertexAttribPointer(tex, 2, GLES20.GL_FLOAT, false, 0, texBuffer!!)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, textureId)
        GLES20.glUniform1i(GLES20.glGetUniformLocation(program, "sTexture"), 0)
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4)
        GLES20.glDisableVertexAttribArray(pos)
        GLES20.glDisableVertexAttribArray(tex)
    }

    private fun floatBuffer(values: FloatArray): FloatBuffer =
        ByteBuffer.allocateDirect(values.size * 4).order(ByteOrder.nativeOrder()).asFloatBuffer().apply {
            put(values)
            position(0)
        }

    private fun createExternalTexture(): Int {
        val textures = IntArray(1)
        GLES20.glGenTextures(1, textures, 0)
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, textures[0])
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)
        return textures[0]
    }

    private fun createProgram(vertex: String, fragment: String): Int {
        fun shader(type: Int, source: String): Int = GLES20.glCreateShader(type).also {
            GLES20.glShaderSource(it, source)
            GLES20.glCompileShader(it)
        }
        return GLES20.glCreateProgram().also {
            GLES20.glAttachShader(it, shader(GLES20.GL_VERTEX_SHADER, vertex))
            GLES20.glAttachShader(it, shader(GLES20.GL_FRAGMENT_SHADER, fragment))
            GLES20.glLinkProgram(it)
        }
    }

    companion object {
        private const val CAMERA_REQUEST = 7402
        private const val MAX_KEYFRAMES = 8
        private const val SAMPLE_INTERVAL_NS = 350_000_000L
        private const val GRID_STEP = 8
        private const val MIN_VALID_POINTS = 25
        private const val MIN_DEPTH_MM = 300
        private const val MAX_DEPTH_MM = 4000
        private const val MIN_BODY_HEIGHT_M = 0.10
        private const val MAX_BODY_HEIGHT_M = 1.40
        private const val VERTEX_SHADER = "attribute vec4 aPosition; attribute vec2 aTexCoord; varying vec2 vTexCoord; void main(){ gl_Position=aPosition; vTexCoord=aTexCoord; }"
        private const val FRAGMENT_SHADER = "#extension GL_OES_EGL_image_external : require\nprecision mediump float; uniform samplerExternalOES sTexture; varying vec2 vTexCoord; void main(){ gl_FragColor=texture2D(sTexture,vTexCoord); }"
    }
}
