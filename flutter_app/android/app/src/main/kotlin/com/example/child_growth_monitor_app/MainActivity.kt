package com.example.child_growth_monitor_app

import android.app.Activity
import android.app.ActivityManager
import android.content.Intent
import com.example.child_growth_monitor_app.ar.FullArScanActivity
import com.google.ar.core.ArCoreApk
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel

class MainActivity : FlutterActivity() {
    private var pendingScanResult: MethodChannel.Result? = null

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        MethodChannel(
            flutterEngine.dartExecutor.binaryMessenger,
            AR_CHANNEL,
        ).setMethodCallHandler { call, result ->
            when (call.method) {
                "checkCapability" -> result.success(capability())
                "startContactlessScan", "startFullScan", "startSparseScan" -> {
                    if (pendingScanResult != null) {
                        result.error("scan_active", "An AR depth scan is already active", null)
                    } else {
                        pendingScanResult = result
                        startActivityForResult(
                            Intent(this, FullArScanActivity::class.java).apply {
                                call.argument<Double>("ageMonths")?.let {
                                    putExtra("ageMonths", it)
                                }
                                call.argument<String>("sex")?.let { putExtra("sex", it) }
                            },
                            AR_SCAN_REQUEST,
                        )
                    }
                }
                else -> result.notImplemented()
            }
        }
    }

    private fun capability(): Map<String, Any?> {
        val availability = ArCoreApk.getInstance().checkAvailability(this)
        val memoryClassMb =
            (getSystemService(ACTIVITY_SERVICE) as ActivityManager).memoryClass
        return mapOf(
            "availability" to availability.name.lowercase(),
            "arSupported" to availability.isSupported,
            "transient" to availability.isTransient,
            "ramMb" to memoryClassMb,
            "method" to "arcore_contactless_anthropometry_v3",
        )
    }

    @Deprecated("Legacy result bridge retained for FlutterActivity compatibility")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode != AR_SCAN_REQUEST) return
        val pending = pendingScanResult ?: return
        pendingScanResult = null
        if (resultCode != Activity.RESULT_OK || data == null) {
            pending.error(
                "scan_unavailable",
                data?.getStringExtra("reason")
                    ?: "Depth scan was cancelled; use guided photos",
                null,
            )
            return
        }
        pending.success(
            mapOf(
                "method" to "arcore_contactless_anthropometry_v3",
                "estimatedHeightCm" to data.getDoubleExtra("estimatedHeightCm", Double.NaN)
                    .takeIf { it.isFinite() },
                "uncertaintyCm" to data.getDoubleExtra("uncertaintyCm", Double.NaN)
                    .takeIf { it.isFinite() },
                "acceptedKeyframes" to data.getIntExtra("acceptedKeyframes", 0),
                "validDepthFraction" to data.getDoubleExtra("validDepthFraction", 0.0),
                "meanDepthConfidence" to data.getDoubleExtra("meanDepthConfidence", 0.0),
                "scanCoverageDegrees" to data.getDoubleExtra("scanCoverageDegrees", 0.0),
                "cameraTravelMeters" to data.getDoubleExtra("cameraTravelMeters", 0.0),
                "floorStabilityCm" to data.getDoubleExtra("floorStabilityCm", Double.NaN)
                    .takeIf { it.isFinite() },
                "capturedBodyPoints" to data.getIntExtra("capturedBodyPoints", 0),
                "durationMs" to data.getLongExtra("durationMs", 0L),
                "qualityScore" to data.getDoubleExtra("qualityScore", 0.0),
                "depthMode" to data.getStringExtra("depthMode"),
                "shoulderWidthCm" to finiteExtra(data, "shoulderWidthCm"),
                "hipWidthCm" to finiteExtra(data, "hipWidthCm"),
                "torsoLengthCm" to finiteExtra(data, "torsoLengthCm"),
                "upperArmLengthCm" to finiteExtra(data, "upperArmLengthCm"),
                "chestDepthCm" to finiteExtra(data, "chestDepthCm"),
                "abdomenDepthCm" to finiteExtra(data, "abdomenDepthCm"),
                "estimatedMuacCm" to finiteExtra(data, "estimatedMuacCm"),
                "muacUncertaintyCm" to finiteExtra(data, "muacUncertaintyCm"),
                "poseQualityScore" to finiteExtra(data, "poseQualityScore"),
                "geometryQualityScore" to finiteExtra(data, "geometryQualityScore"),
                "clinicalMeasurementEligible" to false,
                "isEstimate" to true,
            ),
        )
    }

    private fun finiteExtra(data: Intent, name: String): Double? =
        data.getDoubleExtra(name, Double.NaN).takeIf { it.isFinite() }

    companion object {
        private const val AR_CHANNEL = "org.childgrowthmonitor/ar_scan"
        private const val AR_SCAN_REQUEST = 7401
    }
}
