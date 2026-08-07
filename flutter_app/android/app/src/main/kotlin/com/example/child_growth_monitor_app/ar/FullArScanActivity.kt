package com.example.child_growth_monitor_app.ar

import android.Manifest
import android.app.Activity
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Color
import android.graphics.PointF
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.opengl.GLES11Ext
import android.opengl.GLES20
import android.opengl.GLSurfaceView
import android.os.Bundle
import android.os.SystemClock
import android.view.Gravity
import android.view.Surface
import android.widget.Button
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView
import com.google.ar.core.ArCoreApk
import com.google.ar.core.Config
import com.google.ar.core.Coordinates2d
import com.google.ar.core.Frame
import com.google.ar.core.Plane
import com.google.ar.core.Session
import com.google.ar.core.TrackingState
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.pose.Pose
import com.google.mlkit.vision.pose.PoseDetection
import com.google.mlkit.vision.pose.PoseDetector
import com.google.mlkit.vision.pose.PoseLandmark
import com.google.mlkit.vision.pose.defaults.PoseDetectorOptions
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.ceil
import kotlin.math.hypot
import kotlin.math.sqrt

/**
 * Guided, bounded multi-view ARCore raw-depth capture.
 *
 * Each unique raw-depth frame is paired with its confidence image, reduced to
 * body/floor evidence, and closed immediately. No RGB frame, depth image,
 * point cloud, or mesh is written to disk or returned to Flutter.
 */
class FullArScanActivity : Activity(), GLSurfaceView.Renderer {
    private lateinit var surface: GLSurfaceView
    private lateinit var instruction: TextView
    private lateinit var progress: ProgressBar
    private var session: Session? = null
    private val finished = AtomicBoolean(false)
    private val estimator = FullArScanEstimator()
    private var lastSampleNs = 0L
    private var lastPoseSampleNs = 0L
    private var lastRawDepthTimestampNs = Long.MIN_VALUE
    private val poseInFlight = AtomicBoolean(false)
    @Volatile private var latestPoseEvidence: PoseFrameEvidence? = null
    private val poseDetector: PoseDetector by lazy {
        PoseDetection.getClient(
            PoseDetectorOptions.Builder()
                .setDetectorMode(PoseDetectorOptions.STREAM_MODE)
                .build(),
        )
    }
    private var scanStartedElapsedMs = 0L
    private var textureId = 0
    private var installRequested = false
    private var program = 0
    private var surfaceWidth = 0
    private var surfaceHeight = 0
    private var positionBuffer: FloatBuffer? = null
    private var texBuffer: FloatBuffer? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val root = FrameLayout(this)
        surface = GLSurfaceView(this).apply {
            setEGLContextClientVersion(2)
            preserveEGLContextOnPause = true
            setRenderer(this@FullArScanActivity)
            renderMode = GLSurfaceView.RENDERMODE_CONTINUOUSLY
        }

        val controls = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER_HORIZONTAL
            setPadding(24, 18, 24, 18)
            setBackgroundColor(0xbb000000.toInt())
        }
        instruction = TextView(this).apply {
            text = "Initializing guided depth scan…"
            textSize = 17f
            setTextColor(Color.WHITE)
            gravity = Gravity.CENTER
        }
        progress = ProgressBar(
            this,
            null,
            android.R.attr.progressBarStyleHorizontal,
        ).apply {
            max = 100
            progress = 0
        }
        val cancel = Button(this).apply {
            text = "Cancel and use guided photos"
            setOnClickListener { finishCancelled("Depth scan cancelled by operator") }
        }
        controls.addView(
            instruction,
            LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT,
            ),
        )
        controls.addView(
            progress,
            LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT,
            ).apply { topMargin = 12 },
        )
        controls.addView(
            cancel,
            LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT,
            ).apply { topMargin = 10 },
        )
        root.addView(surface)
        root.addView(
            controls,
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
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
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
        if (session != null || finished.get()) return
        try {
            when (ArCoreApk.getInstance().requestInstall(this, !installRequested)) {
                ArCoreApk.InstallStatus.INSTALL_REQUESTED -> {
                    installRequested = true
                    showInstruction("Finish installing Google Play Services for AR, then return")
                    return
                }
                ArCoreApk.InstallStatus.INSTALLED -> Unit
            }
        } catch (error: Exception) {
            finishCancelled("AR depth could not start: ${error.javaClass.simpleName}")
            return
        }

        val next = try {
            Session(this)
        } catch (error: Exception) {
            finishCancelled("AR session could not start: ${error.javaClass.simpleName}")
            return
        }
        try {
            if (!next.isDepthModeSupported(Config.DepthMode.AUTOMATIC)) {
                next.close()
                finishCancelled("Depth is not supported on this phone")
                return
            }
            val config = next.config.apply {
                depthMode = Config.DepthMode.AUTOMATIC
                planeFindingMode = Config.PlaneFindingMode.HORIZONTAL
                focusMode = Config.FocusMode.AUTO
                updateMode = Config.UpdateMode.LATEST_CAMERA_IMAGE
            }
            next.configure(config)
            if (textureId != 0) next.setCameraTextureName(textureId)
            if (surfaceWidth > 0 && surfaceHeight > 0) {
                next.setDisplayGeometry(display.rotation, surfaceWidth, surfaceHeight)
            }
            next.resume()
            session = next
            scanStartedElapsedMs = SystemClock.elapsedRealtime()
            showInstruction(
                "Keep the whole standing child centered with arms slightly away. Move slowly toward the side",
            )
        } catch (error: Exception) {
            next.close()
            finishCancelled("AR depth could not start: ${error.javaClass.simpleName}")
        }
    }

    override fun onResume() {
        super.onResume()
        surface.onResume()
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) return
        if (session == null) {
            initializeSession()
        } else {
            try {
                session?.resume()
            } catch (_: Exception) {
                finishCancelled("AR session could not resume")
            }
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
        if (::surface.isInitialized) {
            try {
                poseDetector.close()
            } catch (_: Exception) {
                // The detector may not have been initialized on early exit.
            }
        }
        super.onDestroy()
    }

    @Suppress("DEPRECATION", "OVERRIDE_DEPRECATION")
    override fun onBackPressed() {
        finishCancelled("Depth scan cancelled by operator")
    }

    override fun onSurfaceCreated(
        gl: javax.microedition.khronos.opengles.GL10?,
        config: javax.microedition.khronos.egl.EGLConfig?,
    ) {
        textureId = createExternalTexture()
        program = createProgram(VERTEX_SHADER, FRAGMENT_SHADER)
        positionBuffer = floatBuffer(floatArrayOf(-1f, -1f, 1f, -1f, -1f, 1f, 1f, 1f))
        texBuffer = floatBuffer(floatArrayOf(0f, 1f, 1f, 1f, 0f, 0f, 1f, 0f))
        session?.setCameraTextureName(textureId)
        GLES20.glClearColor(0f, 0f, 0f, 1f)
    }

    override fun onSurfaceChanged(
        gl: javax.microedition.khronos.opengles.GL10?,
        width: Int,
        height: Int,
    ) {
        surfaceWidth = width
        surfaceHeight = height
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
        val elapsedMs = SystemClock.elapsedRealtime() - scanStartedElapsedMs
        if (elapsedMs >= SCAN_TIMEOUT_MS || estimator.hasReachedFrameLimit()) {
            finishWithAvailableSummary(elapsedMs)
            return
        }
        if (frame.camera.trackingState != TrackingState.TRACKING) {
            showInstruction("Move the phone slowly until tracking starts")
            return
        }
        if (estimator.readyToFinish()) {
            finishWithAvailableSummary(elapsedMs)
            return
        }

        val now = frame.timestamp
        if (now <= 0L) return
        samplePose(frame, now)
        if (now - lastSampleNs < SAMPLE_INTERVAL_NS) return
        lastSampleNs = now
        sampleFullDepth(active, frame)
    }

    private fun samplePose(frame: Frame, timestampNs: Long) {
        if (timestampNs - lastPoseSampleNs < POSE_SAMPLE_INTERVAL_NS ||
            !poseInFlight.compareAndSet(false, true)
        ) {
            return
        }
        val cameraImage = try {
            frame.acquireCameraImage()
        } catch (_: Exception) {
            poseInFlight.set(false)
            return
        }
        lastPoseSampleNs = timestampNs
        val input = try {
            InputImage.fromMediaImage(cameraImage, cameraImageRotationDegrees())
        } catch (_: Exception) {
            cameraImage.close()
            poseInFlight.set(false)
            return
        }
        try {
            poseDetector.process(input)
                .addOnSuccessListener { pose ->
                    extractPoseEvidence(pose, timestampNs)?.let { latestPoseEvidence = it }
                }
                .addOnCompleteListener {
                    cameraImage.close()
                    poseInFlight.set(false)
                }
        } catch (_: Exception) {
            cameraImage.close()
            poseInFlight.set(false)
        }
    }

    private fun extractPoseEvidence(pose: Pose, timestampNs: Long): PoseFrameEvidence? {
        fun landmark(type: Int): PoseLandmark? = pose.getPoseLandmark(type)?.takeIf {
            it.inFrameLikelihood >= MIN_POSE_LIKELIHOOD
        }
        val nose = landmark(PoseLandmark.NOSE) ?: return null
        val leftShoulder = landmark(PoseLandmark.LEFT_SHOULDER) ?: return null
        val rightShoulder = landmark(PoseLandmark.RIGHT_SHOULDER) ?: return null
        val leftHip = landmark(PoseLandmark.LEFT_HIP) ?: return null
        val rightHip = landmark(PoseLandmark.RIGHT_HIP) ?: return null
        val leftElbow = landmark(PoseLandmark.LEFT_ELBOW) ?: return null
        val rightElbow = landmark(PoseLandmark.RIGHT_ELBOW) ?: return null
        val leftHeel = landmark(PoseLandmark.LEFT_HEEL) ?: return null
        val rightHeel = landmark(PoseLandmark.RIGHT_HEEL) ?: return null

        fun midpoint(first: PointF, second: PointF) = PointF(
            (first.x + second.x) / 2f,
            (first.y + second.y) / 2f,
        )
        fun distance(first: PointF, second: PointF): Double = hypot(
            (second.x - first.x).toDouble(),
            (second.y - first.y).toDouble(),
        )
        val head = nose.position
        val heels = midpoint(leftHeel.position, rightHeel.position)
        val shoulders = midpoint(leftShoulder.position, rightShoulder.position)
        val hips = midpoint(leftHip.position, rightHip.position)
        val bodyAxisLength = distance(head, heels)
        if (!bodyAxisLength.isFinite() || bodyAxisLength < MIN_POSE_AXIS_PIXELS) return null
        val axisX = heels.x - head.x
        val axisY = heels.y - head.y
        val axisSquared = axisX * axisX + axisY * axisY
        if (axisSquared <= 0f) return null

        fun heightRatio(point: PointF): Double {
            val t = ((point.x - head.x) * axisX + (point.y - head.y) * axisY) / axisSquared
            return 1.0 - t.toDouble()
        }
        val shoulderHeightRatio = heightRatio(shoulders)
        val hipHeightRatio = heightRatio(hips)
        val armMidpoint = midpoint(
            midpoint(leftShoulder.position, leftElbow.position),
            midpoint(rightShoulder.position, rightElbow.position),
        )
        val armMidHeightRatio = heightRatio(armMidpoint)
        val torsoLengthRatio = distance(shoulders, hips) / bodyAxisLength
        val upperArmLengthRatio = listOf(
            distance(leftShoulder.position, leftElbow.position),
            distance(rightShoulder.position, rightElbow.position),
        ).average() / bodyAxisLength
        if (shoulderHeightRatio !in MIN_SHOULDER_HEIGHT_RATIO..MAX_SHOULDER_HEIGHT_RATIO ||
            hipHeightRatio !in MIN_HIP_HEIGHT_RATIO..MAX_HIP_HEIGHT_RATIO ||
            shoulderHeightRatio <= hipHeightRatio ||
            armMidHeightRatio !in MIN_ARM_MID_HEIGHT_RATIO..MAX_ARM_MID_HEIGHT_RATIO ||
            torsoLengthRatio !in MIN_TORSO_LENGTH_RATIO..MAX_TORSO_LENGTH_RATIO ||
            upperArmLengthRatio !in MIN_UPPER_ARM_LENGTH_RATIO..MAX_UPPER_ARM_LENGTH_RATIO
        ) {
            return null
        }
        val quality = listOf(
            nose,
            leftShoulder,
            rightShoulder,
            leftHip,
            rightHip,
            leftElbow,
            rightElbow,
            leftHeel,
            rightHeel,
        ).map { it.inFrameLikelihood.toDouble() }.average()
        return PoseFrameEvidence(
            timestampNs = timestampNs,
            shoulderHeightRatio = shoulderHeightRatio,
            hipHeightRatio = hipHeightRatio,
            armMidHeightRatio = armMidHeightRatio,
            torsoLengthRatio = torsoLengthRatio,
            upperArmLengthRatio = upperArmLengthRatio,
            qualityScore = quality.coerceIn(0.0, 1.0),
        )
    }

    private fun cameraImageRotationDegrees(): Int {
        val cameraManager = getSystemService(Context.CAMERA_SERVICE) as CameraManager
        val sensorOrientation = cameraManager.cameraIdList.asSequence()
            .map { cameraManager.getCameraCharacteristics(it) }
            .firstOrNull {
                it.get(CameraCharacteristics.LENS_FACING) == CameraCharacteristics.LENS_FACING_BACK
            }
            ?.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 90
        val displayDegrees = when (display.rotation) {
            Surface.ROTATION_90 -> 90
            Surface.ROTATION_180 -> 180
            Surface.ROTATION_270 -> 270
            else -> 0
        }
        return (sensorOrientation - displayDegrees + 360) % 360
    }

    private fun sampleFullDepth(active: Session, frame: Frame) {
        val floor = findFloor(active, frame)
        if (floor == null) {
            showInstruction("Point at the open floor beside the child's feet")
            return
        }
        val depth = try {
            frame.acquireRawDepthImage16Bits()
        } catch (_: Exception) {
            showInstruction("Move slowly while raw depth becomes available")
            return
        }
        try {
            if (depth.timestamp == lastRawDepthTimestampNs || depth.timestamp != frame.timestamp) {
                showInstruction("Continue moving to collect a new depth frame")
                return
            }
            val confidence = try {
                frame.acquireRawDepthConfidenceImage()
            } catch (_: Exception) {
                showInstruction("Waiting for depth confidence")
                return
            }
            try {
                if (confidence.width != depth.width || confidence.height != depth.height) {
                    showInstruction("Depth confidence did not align; continue with photos")
                    return
                }
                val evidence = analyzeDepth(frame, depth, confidence, floor.centerPose.ty().toDouble())
                lastRawDepthTimestampNs = depth.timestamp
                if (evidence == null) return
                val acceptance = estimator.tryAdd(evidence)
                showProgress(acceptance.guidance)
            } finally {
                confidence.close()
            }
        } finally {
            depth.close()
        }
    }

    private fun analyzeDepth(
        frame: Frame,
        depth: android.media.Image,
        confidence: android.media.Image,
        floorY: Double,
    ): DepthFrameEvidence? {
        // Raw depth is aligned to the CPU camera image, not the GL texture.
        // Scale image intrinsics into the lower-resolution depth plane.
        val intrinsics = frame.camera.imageIntrinsics
        val dimensions = intrinsics.imageDimensions
        val fx = intrinsics.focalLength[0] * depth.width / dimensions[0].toFloat()
        val fy = intrinsics.focalLength[1] * depth.height / dimensions[1].toFloat()
        val cx = intrinsics.principalPoint[0] * depth.width / dimensions[0].toFloat()
        val cy = intrinsics.principalPoint[1] * depth.height / dimensions[1].toFloat()
        if (fx <= 0f || fy <= 0f) {
            showInstruction("Camera calibration was unavailable")
            return null
        }

        val depthPlane = depth.planes[0]
        val confidencePlane = confidence.planes[0]
        val depthBuffer = depthPlane.buffer.duplicate().order(ByteOrder.LITTLE_ENDIAN)
        val confidenceBuffer = confidencePlane.buffer.duplicate()
        val step = ceil(sqrt(depth.width * depth.height / MAX_GRID_POINTS.toDouble()))
            .toInt()
            .coerceAtLeast(1)
        val startX = (depth.width * BODY_ROI_LEFT).toInt()
        val endX = (depth.width * BODY_ROI_RIGHT).toInt()
        val startY = (depth.height * BODY_ROI_TOP).toInt()
        val endY = (depth.height * BODY_ROI_BOTTOM).toInt()
        val rawPixels = ArrayList<RawDepthPixel>()
        val seedDepths = ArrayList<Double>()
        var attemptedPoints = 0

        for (y in startY until endY step step) {
            for (x in startX until endX step step) {
                attemptedPoints++
                val depthOffset = y * depthPlane.rowStride + x * depthPlane.pixelStride
                val confidenceOffset =
                    y * confidencePlane.rowStride + x * confidencePlane.pixelStride
                if (depthOffset + 1 >= depthBuffer.limit() || confidenceOffset >= confidenceBuffer.limit()) {
                    continue
                }
                val depthMm = depthBuffer.getShort(depthOffset).toInt() and 0xffff
                val confidenceValue = confidenceBuffer.get(confidenceOffset).toInt() and 0xff
                if (depthMm !in MIN_DEPTH_MM..MAX_DEPTH_MM || confidenceValue < MIN_CONFIDENCE) {
                    continue
                }
                val depthMeters = depthMm / 1000.0
                val confidenceNormalized = confidenceValue / 255.0
                rawPixels += RawDepthPixel(x, y, depthMeters, confidenceNormalized)
                if (
                    x in (depth.width * SEED_LEFT).toInt()..(depth.width * SEED_RIGHT).toInt() &&
                    y in (depth.height * SEED_TOP).toInt()..(depth.height * SEED_BOTTOM).toInt()
                ) {
                    seedDepths += depthMeters
                }
            }
        }

        if (seedDepths.size < MIN_SEED_POINTS) {
            showInstruction("Center the child's torso and keep the full body visible")
            return null
        }
        val targetDepth = FullArScanEstimator.median(seedDepths)
        val cameraPose = frame.camera.pose
        val initialBody = ArrayList<BodyPoint>()
        for (pixel in rawPixels) {
            if (abs(pixel.depthMeters - targetDepth) > BODY_DEPTH_BAND_M) continue
            val cameraPoint = floatArrayOf(
                (pixel.depthMeters * (pixel.x - cx) / fx).toFloat(),
                (pixel.depthMeters * (cy - pixel.y) / fy).toFloat(),
                -pixel.depthMeters.toFloat(),
            )
            val world = cameraPose.transformPoint(cameraPoint)
            val heightAboveFloor = world[1].toDouble() - floorY
            if (heightAboveFloor !in MIN_BODY_POINT_HEIGHT_M..MAX_BODY_HEIGHT_M) continue
            initialBody += BodyPoint(
                x = world[0].toDouble(),
                yAboveFloor = heightAboveFloor,
                z = world[2].toDouble(),
                cameraX = cameraPoint[0].toDouble(),
                confidence = pixel.confidence,
            )
        }

        if (initialBody.size < FullArScanEstimator.MIN_BODY_POINTS) {
            showInstruction("Keep the child centered and clear the background")
            return null
        }
        val centerX = FullArScanEstimator.median(initialBody.map { it.x })
        val centerZ = FullArScanEstimator.median(initialBody.map { it.z })
        val body = initialBody.filter {
            hypot(it.x - centerX, it.z - centerZ) <= MAX_BODY_RADIUS_M
        }
        if (body.size < FullArScanEstimator.MIN_BODY_POINTS) {
            showInstruction("Move closer while keeping the entire child visible")
            return null
        }
        val sortedHeights = body.map { it.yAboveFloor }.sorted()
        if (FullArScanEstimator.percentile(sortedHeights, 0.02) > MAX_LOWEST_BODY_POINT_M) {
            showInstruction("Include both feet and the floor in the frame")
            return null
        }
        val heightCm = FullArScanEstimator.percentile(sortedHeights, 0.99) * 100.0
        val poseEvidence = latestPoseEvidence?.takeIf {
            abs(it.timestampNs - depth.timestamp) <= MAX_POSE_EVIDENCE_AGE_NS
        }
        val profile = buildBodyProfile(body, heightCm / 100.0, poseEvidence)
        val bodyCenterX = FullArScanEstimator.median(body.map { it.x })
        val bodyCenterZ = FullArScanEstimator.median(body.map { it.z })
        val forward = FloatArray(3)
        cameraPose.getTransformedAxis(2, -1.0f, forward, 0)
        val yawDegrees = atan2(forward[0].toDouble(), forward[2].toDouble()) * 180.0 / PI
        return DepthFrameEvidence(
            heightCm = heightCm,
            validDepthFraction =
                (rawPixels.size.toDouble() / attemptedPoints.coerceAtLeast(1)).coerceIn(0.0, 1.0),
            meanConfidence = body.map { it.confidence }.average(),
            floorY = floorY,
            bodyCenterX = bodyCenterX,
            bodyCenterZ = bodyCenterZ,
            bodyPointCount = body.size,
            profile = profile,
            pose = ScanPose(
                x = cameraPose.tx().toDouble(),
                y = cameraPose.ty().toDouble(),
                z = cameraPose.tz().toDouble(),
                yawDegrees = yawDegrees,
                timestampNs = depth.timestamp,
            ),
        )
    }

    private fun buildBodyProfile(
        body: List<BodyPoint>,
        heightMeters: Double,
        pose: PoseFrameEvidence?,
    ): BodyProfileEvidence? {
        if (!heightMeters.isFinite() || heightMeters <= 0.0) return null

        fun spanAt(lowerRatio: Double, upperRatio: Double): Double? {
            val values = body.asSequence()
                .filter {
                    it.yAboveFloor >= heightMeters * lowerRatio &&
                        it.yAboveFloor <= heightMeters * upperRatio
                }
                .map { it.cameraX }
                .sorted()
                .toList()
            if (values.size < MIN_PROFILE_POINTS) return null
            val lower = FullArScanEstimator.percentile(values, 0.08)
            val upper = FullArScanEstimator.percentile(values, 0.92)
            return ((upper - lower) * 100.0).takeIf { it.isFinite() && it > 0.0 }
        }

        val shoulderRatio = pose?.shoulderHeightRatio ?: DEFAULT_SHOULDER_HEIGHT_RATIO
        val hipRatio = pose?.hipHeightRatio ?: DEFAULT_HIP_HEIGHT_RATIO
        val chestRatio = hipRatio + (shoulderRatio - hipRatio) * 0.62
        val abdomenRatio = hipRatio + (shoulderRatio - hipRatio) * 0.28
        val armRatio = pose?.armMidHeightRatio ?: DEFAULT_ARM_MID_HEIGHT_RATIO
        val shoulderSpan = spanAt(
            shoulderRatio - PROFILE_HALF_BAND_RATIO,
            shoulderRatio + PROFILE_HALF_BAND_RATIO,
        )
        val hipSpan = spanAt(
            hipRatio - PROFILE_HALF_BAND_RATIO,
            hipRatio + PROFILE_HALF_BAND_RATIO,
        )
        val chestSpan = spanAt(
            chestRatio - PROFILE_HALF_BAND_RATIO,
            chestRatio + PROFILE_HALF_BAND_RATIO,
        )
        val abdomenSpan = spanAt(
            abdomenRatio - PROFILE_HALF_BAND_RATIO,
            abdomenRatio + PROFILE_HALF_BAND_RATIO,
        )
        val armBand = body.filter {
            it.yAboveFloor >= heightMeters * (armRatio - ARM_HALF_BAND_RATIO) &&
                it.yAboveFloor <= heightMeters * (armRatio + ARM_HALF_BAND_RATIO)
        }
        val armSpan = armClusterSpan(armBand, shoulderSpan)
        if (listOf(shoulderSpan, hipSpan, chestSpan, abdomenSpan, armSpan).all { it == null }) {
            return null
        }
        return BodyProfileEvidence(
            shoulderSpanCm = shoulderSpan,
            hipSpanCm = hipSpan,
            chestSpanCm = chestSpan,
            abdomenSpanCm = abdomenSpan,
            armSpanCm = armSpan,
            torsoLengthCm = pose?.let { heightMeters * it.torsoLengthRatio * 100.0 },
            upperArmLengthCm = pose?.let { heightMeters * it.upperArmLengthRatio * 100.0 },
            poseQualityScore = pose?.qualityScore,
        )
    }

    private fun armClusterSpan(
        armBand: List<BodyPoint>,
        shoulderSpanCm: Double?,
    ): Double? {
        if (armBand.size < MIN_PROFILE_POINTS || shoulderSpanCm == null) return null
        val center = FullArScanEstimator.median(armBand.map { it.cameraX })
        val cutoffMeters = shoulderSpanCm / 100.0 * ARM_CLUSTER_INNER_RATIO
        val left = armBand.map { it.cameraX }.filter { it < center - cutoffMeters }.sorted()
        val right = armBand.map { it.cameraX }.filter { it > center + cutoffMeters }.sorted()
        val selected = listOf(left, right)
            .filter { it.size >= MIN_ARM_PROFILE_POINTS }
            .minByOrNull { values ->
                FullArScanEstimator.percentile(values, 0.90) -
                    FullArScanEstimator.percentile(values, 0.10)
            } ?: return null
        val span = (
            FullArScanEstimator.percentile(selected, 0.90) -
                FullArScanEstimator.percentile(selected, 0.10)
            ) * 100.0
        return span.takeIf { it.isFinite() && it in MIN_ARM_SPAN_CM..MAX_ARM_SPAN_CM }
    }

    private fun findFloor(active: Session, frame: Frame): Plane? {
        val cameraY = frame.camera.pose.ty()
        return active.getAllTrackables(Plane::class.java)
            .asSequence()
            .filter {
                it.trackingState == TrackingState.TRACKING &&
                    it.subsumedBy == null &&
                    it.type == Plane.Type.HORIZONTAL_UPWARD_FACING
            }
            .filter { cameraY - it.centerPose.ty() in MIN_CAMERA_FLOOR_M..MAX_CAMERA_FLOOR_M }
            .maxByOrNull { it.extentX * it.extentZ }
    }

    private fun finishWithAvailableSummary(durationMs: Long) {
        if (finished.get()) return
        val summary = estimator.summarize(durationMs)
        if (summary == null) {
            finishCancelled("Full depth quality gate failed: ${estimator.failureGuidance()}")
            return
        }
        if (!finished.compareAndSet(false, true)) return
        setResult(
            RESULT_OK,
            Intent().apply {
                putExtra("estimatedHeightCm", summary.estimatedHeightCm)
                putExtra("uncertaintyCm", summary.uncertaintyCm)
                putExtra("acceptedKeyframes", summary.acceptedKeyframes)
                putExtra("validDepthFraction", summary.validDepthFraction)
                putExtra("meanDepthConfidence", summary.meanDepthConfidence)
                putExtra("scanCoverageDegrees", summary.scanCoverageDegrees)
                putExtra("cameraTravelMeters", summary.cameraTravelMeters)
                putExtra("floorStabilityCm", summary.floorStabilityCm)
                putExtra("capturedBodyPoints", summary.capturedBodyPoints)
                putExtra("durationMs", summary.durationMs)
                putExtra("qualityScore", summary.qualityScore)
                putExtra("depthMode", "raw_depth_with_confidence")
                summary.geometry?.let { geometry ->
                    putExtra("shoulderWidthCm", geometry.shoulderWidthCm)
                    putExtra("hipWidthCm", geometry.hipWidthCm)
                    putExtra("torsoLengthCm", geometry.torsoLengthCm)
                    putExtra("upperArmLengthCm", geometry.upperArmLengthCm)
                    putExtra("chestDepthCm", geometry.chestDepthCm)
                    putExtra("abdomenDepthCm", geometry.abdomenDepthCm)
                    putExtra("estimatedMuacCm", geometry.estimatedMuacCm ?: Double.NaN)
                    putExtra("muacUncertaintyCm", geometry.muacUncertaintyCm ?: Double.NaN)
                    putExtra("poseQualityScore", geometry.poseQualityScore)
                    putExtra("geometryQualityScore", geometry.qualityScore)
                }
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

    private fun showProgress(message: String) = runOnUiThread {
        instruction.text = message
        progress.progress = estimator.progressPercent
    }

    private fun drawCamera(frame: Frame) {
        if (frame.hasDisplayGeometryChanged()) {
            val output = ByteBuffer.allocateDirect(8 * 4)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer()
            frame.transformCoordinates2d(
                Coordinates2d.OPENGL_NORMALIZED_DEVICE_COORDINATES,
                positionBuffer!!,
                Coordinates2d.TEXTURE_NORMALIZED,
                output,
            )
            texBuffer = output
        }
        GLES20.glUseProgram(program)
        val position = GLES20.glGetAttribLocation(program, "aPosition")
        val texture = GLES20.glGetAttribLocation(program, "aTexCoord")
        GLES20.glEnableVertexAttribArray(position)
        GLES20.glVertexAttribPointer(position, 2, GLES20.GL_FLOAT, false, 0, positionBuffer!!)
        GLES20.glEnableVertexAttribArray(texture)
        GLES20.glVertexAttribPointer(texture, 2, GLES20.GL_FLOAT, false, 0, texBuffer!!)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, textureId)
        GLES20.glUniform1i(GLES20.glGetUniformLocation(program, "sTexture"), 0)
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4)
        GLES20.glDisableVertexAttribArray(position)
        GLES20.glDisableVertexAttribArray(texture)
    }

    private fun floatBuffer(values: FloatArray): FloatBuffer =
        ByteBuffer.allocateDirect(values.size * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
            .apply {
                put(values)
                position(0)
            }

    private fun createExternalTexture(): Int {
        val textures = IntArray(1)
        GLES20.glGenTextures(1, textures, 0)
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, textures[0])
        GLES20.glTexParameteri(
            GLES11Ext.GL_TEXTURE_EXTERNAL_OES,
            GLES20.GL_TEXTURE_MIN_FILTER,
            GLES20.GL_LINEAR,
        )
        GLES20.glTexParameteri(
            GLES11Ext.GL_TEXTURE_EXTERNAL_OES,
            GLES20.GL_TEXTURE_MAG_FILTER,
            GLES20.GL_LINEAR,
        )
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

    private data class RawDepthPixel(
        val x: Int,
        val y: Int,
        val depthMeters: Double,
        val confidence: Double,
    )

    private data class BodyPoint(
        val x: Double,
        val yAboveFloor: Double,
        val z: Double,
        val cameraX: Double,
        val confidence: Double,
    )

    private data class PoseFrameEvidence(
        val timestampNs: Long,
        val shoulderHeightRatio: Double,
        val hipHeightRatio: Double,
        val armMidHeightRatio: Double,
        val torsoLengthRatio: Double,
        val upperArmLengthRatio: Double,
        val qualityScore: Double,
    )

    companion object {
        private const val CAMERA_REQUEST = 7402
        private const val SAMPLE_INTERVAL_NS = 250_000_000L
        private const val POSE_SAMPLE_INTERVAL_NS = 350_000_000L
        private const val MAX_POSE_EVIDENCE_AGE_NS = 1_200_000_000L
        private const val SCAN_TIMEOUT_MS = 45_000L
        private const val MAX_GRID_POINTS = 6_000
        private const val MIN_CONFIDENCE = 128
        private const val MIN_SEED_POINTS = 18
        private const val MIN_PROFILE_POINTS = 10
        private const val MIN_ARM_PROFILE_POINTS = 5
        private const val MIN_POSE_AXIS_PIXELS = 120.0
        private const val MIN_POSE_LIKELIHOOD = 0.55f
        private const val MIN_SHOULDER_HEIGHT_RATIO = 0.62
        private const val MAX_SHOULDER_HEIGHT_RATIO = 0.94
        private const val MIN_HIP_HEIGHT_RATIO = 0.35
        private const val MAX_HIP_HEIGHT_RATIO = 0.72
        private const val MIN_ARM_MID_HEIGHT_RATIO = 0.42
        private const val MAX_ARM_MID_HEIGHT_RATIO = 0.88
        private const val MIN_TORSO_LENGTH_RATIO = 0.14
        private const val MAX_TORSO_LENGTH_RATIO = 0.42
        private const val MIN_UPPER_ARM_LENGTH_RATIO = 0.08
        private const val MAX_UPPER_ARM_LENGTH_RATIO = 0.28
        private const val DEFAULT_SHOULDER_HEIGHT_RATIO = 0.78
        private const val DEFAULT_HIP_HEIGHT_RATIO = 0.53
        private const val DEFAULT_ARM_MID_HEIGHT_RATIO = 0.70
        private const val PROFILE_HALF_BAND_RATIO = 0.035
        private const val ARM_HALF_BAND_RATIO = 0.045
        private const val MIN_DEPTH_MM = 500
        private const val MAX_DEPTH_MM = 5_000
        private const val MIN_CAMERA_FLOOR_M = 0.40f
        private const val MAX_CAMERA_FLOOR_M = 2.20f
        private const val MIN_BODY_POINT_HEIGHT_M = 0.015
        private const val MAX_BODY_HEIGHT_M = 1.50
        private const val MAX_LOWEST_BODY_POINT_M = 0.20
        private const val MAX_BODY_RADIUS_M = 0.55
        private const val ARM_CLUSTER_INNER_RATIO = 0.30
        private const val MIN_ARM_SPAN_CM = 2.0
        private const val MAX_ARM_SPAN_CM = 10.0
        private const val BODY_DEPTH_BAND_M = 0.40
        private const val BODY_ROI_LEFT = 0.10
        private const val BODY_ROI_RIGHT = 0.90
        private const val BODY_ROI_TOP = 0.03
        private const val BODY_ROI_BOTTOM = 0.98
        private const val SEED_LEFT = 0.35
        private const val SEED_RIGHT = 0.65
        private const val SEED_TOP = 0.15
        private const val SEED_BOTTOM = 0.88
        private const val VERTEX_SHADER =
            "attribute vec4 aPosition; attribute vec2 aTexCoord; varying vec2 vTexCoord; " +
                "void main(){ gl_Position=aPosition; vTexCoord=aTexCoord; }"
        private const val FRAGMENT_SHADER =
            "#extension GL_OES_EGL_image_external : require\nprecision mediump float; " +
                "uniform samplerExternalOES sTexture; varying vec2 vTexCoord; " +
                "void main(){ gl_FragColor=texture2D(sTexture,vTexCoord); }"
    }
}
