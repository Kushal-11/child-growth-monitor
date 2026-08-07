package com.example.child_growth_monitor_app

import android.app.Activity
import android.content.Intent
import com.example.child_growth_monitor_app.ar.SparseArScanActivity
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
                "startSparseScan" -> {
                    if (pendingScanResult != null) {
                        result.error("scan_active", "A sparse AR scan is already active", null)
                    } else {
                        pendingScanResult = result
                        startActivityForResult(
                            Intent(this, SparseArScanActivity::class.java),
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
        return mapOf(
            "availability" to availability.name.lowercase(),
            "arSupported" to availability.isSupported,
            "transient" to availability.isTransient,
            "ramMb" to (Runtime.getRuntime().maxMemory() / (1024L * 1024L)),
            "method" to "arcore_sparse_depth_v1",
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
                "depth_unavailable",
                data?.getStringExtra("reason")
                    ?: "Depth scan was cancelled; use guided photos",
                null,
            )
            return
        }
        pending.success(
            mapOf(
                "method" to "arcore_sparse_depth_v1",
                "estimatedHeightCm" to data.getDoubleExtra("estimatedHeightCm", Double.NaN)
                    .takeIf { it.isFinite() },
                "uncertaintyCm" to data.getDoubleExtra("uncertaintyCm", Double.NaN)
                    .takeIf { it.isFinite() },
                "acceptedKeyframes" to data.getIntExtra("acceptedKeyframes", 0),
                "validDepthFraction" to data.getDoubleExtra("validDepthFraction", 0.0),
                "depthMode" to data.getStringExtra("depthMode"),
                "clinicalMeasurementEligible" to false,
            ),
        )
    }

    companion object {
        private const val AR_CHANNEL = "org.childgrowthmonitor/ar_scan"
        private const val AR_SCAN_REQUEST = 7401
    }
}
