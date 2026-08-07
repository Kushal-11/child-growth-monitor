package com.example.child_growth_monitor_app.ar

import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.min
import kotlin.math.sqrt

internal data class ScanPose(
    val x: Double,
    val y: Double,
    val z: Double,
    val yawDegrees: Double,
    val timestampNs: Long,
)

internal data class DepthFrameEvidence(
    val heightCm: Double,
    val validDepthFraction: Double,
    val meanConfidence: Double,
    val floorY: Double,
    val bodyCenterX: Double,
    val bodyCenterZ: Double,
    val bodyPointCount: Int,
    val pose: ScanPose,
)

internal data class FrameAcceptance(
    val accepted: Boolean,
    val guidance: String,
)

internal data class FullArScanSummary(
    val estimatedHeightCm: Double,
    val uncertaintyCm: Double,
    val acceptedKeyframes: Int,
    val validDepthFraction: Double,
    val meanDepthConfidence: Double,
    val scanCoverageDegrees: Double,
    val cameraTravelMeters: Double,
    val floorStabilityCm: Double,
    val capturedBodyPoints: Int,
    val durationMs: Long,
    val qualityScore: Double,
)

/**
 * Pure quality gate and robust aggregator for the guided ARCore scan.
 *
 * Raw images and body points are intentionally not retained here. Native frames
 * are reduced to bounded evidence before crossing this boundary so the same
 * deterministic rules can be exercised by host-side unit tests.
 */
internal class FullArScanEstimator {
    private val frames = mutableListOf<DepthFrameEvidence>()
    private var cameraTravelMeters = 0.0

    val acceptedKeyframes: Int
        get() = frames.size

    val progressPercent: Int
        get() {
            val frameProgress = (acceptedKeyframes.toDouble() / TARGET_KEYFRAMES).coerceIn(0.0, 1.0)
            val travelProgress = (cameraTravelMeters / TARGET_CAMERA_TRAVEL_M).coerceIn(0.0, 1.0)
            val coverageProgress = (coverageDegrees() / TARGET_COVERAGE_DEGREES).coerceIn(0.0, 1.0)
            return (100.0 * (0.50 * frameProgress + 0.25 * travelProgress + 0.25 * coverageProgress))
                .toInt()
                .coerceIn(0, 100)
        }

    fun tryAdd(frame: DepthFrameEvidence): FrameAcceptance {
        if (!frame.heightCm.isFinite() || frame.heightCm !in MIN_HEIGHT_CM..MAX_HEIGHT_CM) {
            return FrameAcceptance(false, "Keep the full standing child in view")
        }
        if (frame.bodyPointCount < MIN_BODY_POINTS) {
            return FrameAcceptance(false, "Move slowly and keep the child centered")
        }
        if (frame.meanConfidence < MIN_MEAN_CONFIDENCE) {
            return FrameAcceptance(false, "Find brighter, more textured surroundings")
        }

        if (frames.size >= FLOOR_WARMUP_FRAMES) {
            val floorMedian = median(frames.map { it.floorY })
            if (abs(frame.floorY - floorMedian) > MAX_FLOOR_DEVIATION_M) {
                return FrameAcceptance(false, "Point at the same floor beside the child's feet")
            }
            val centerX = median(frames.map { it.bodyCenterX })
            val centerZ = median(frames.map { it.bodyCenterZ })
            if (hypot(frame.bodyCenterX - centerX, frame.bodyCenterZ - centerZ) > MAX_BODY_DRIFT_M) {
                return FrameAcceptance(false, "Ask the child to stay still while you move")
            }
        }

        val previous = frames.lastOrNull()
        if (previous != null) {
            val translation = distance(previous.pose, frame.pose)
            val yawDelta = angularDistance(previous.pose.yawDegrees, frame.pose.yawDegrees)
            if (translation < MIN_KEYFRAME_TRANSLATION_M && yawDelta < MIN_KEYFRAME_YAW_DEGREES) {
                return FrameAcceptance(false, "Move one small step around the child")
            }
            cameraTravelMeters += translation
        }

        frames += frame
        return FrameAcceptance(
            true,
            if (hasMinimumCoverage()) {
                "Good coverage. Continue slowly for a stronger scan"
            } else {
                "Good depth ${frames.size}/$TARGET_KEYFRAMES. Continue around the child"
            },
        )
    }

    fun readyToFinish(): Boolean =
        frames.size >= TARGET_KEYFRAMES &&
            cameraTravelMeters >= TARGET_CAMERA_TRAVEL_M &&
            coverageDegrees() >= TARGET_COVERAGE_DEGREES &&
            floorStabilityCm() <= MAX_FLOOR_STABILITY_CM

    fun hasReachedFrameLimit(): Boolean = frames.size >= MAX_KEYFRAMES

    fun summarize(durationMs: Long): FullArScanSummary? {
        if (!hasMinimumCoverage()) return null

        val heights = frames.map { it.heightCm }
        val heightMedian = median(heights)
        val absoluteDeviations = heights.map { abs(it - heightMedian) }
        val uncertainty = (median(absoluteDeviations) * MAD_TO_SIGMA).coerceAtLeast(0.2)
        if (uncertainty > MAX_HEIGHT_UNCERTAINTY_CM) return null

        val confidence = frames.map { it.meanConfidence }.average()
        val validFraction = median(frames.map { it.validDepthFraction })
        val coverage = coverageDegrees()
        val floorStability = floorStabilityCm()
        val qualityScore = (
            0.20 * (frames.size.toDouble() / TARGET_KEYFRAMES).coerceIn(0.0, 1.0) +
                0.20 * ((confidence - MIN_MEAN_CONFIDENCE) / 0.35).coerceIn(0.0, 1.0) +
                0.20 * (coverage / TARGET_COVERAGE_DEGREES).coerceIn(0.0, 1.0) +
                0.15 * (cameraTravelMeters / TARGET_CAMERA_TRAVEL_M).coerceIn(0.0, 1.0) +
                0.15 * (1.0 - floorStability / MAX_FLOOR_STABILITY_CM).coerceIn(0.0, 1.0) +
                0.10 * (1.0 - uncertainty / MAX_HEIGHT_UNCERTAINTY_CM).coerceIn(0.0, 1.0)
            ).coerceIn(0.0, 1.0)

        return FullArScanSummary(
            estimatedHeightCm = heightMedian,
            uncertaintyCm = uncertainty,
            acceptedKeyframes = frames.size,
            validDepthFraction = validFraction,
            meanDepthConfidence = confidence,
            scanCoverageDegrees = coverage,
            cameraTravelMeters = cameraTravelMeters,
            floorStabilityCm = floorStability,
            capturedBodyPoints = frames.sumOf { it.bodyPointCount },
            durationMs = durationMs.coerceAtLeast(1L),
            qualityScore = qualityScore,
        )
    }

    fun failureGuidance(): String {
        if (frames.size < MIN_KEYFRAMES) return "Not enough stable depth frames"
        if (cameraTravelMeters < MIN_CAMERA_TRAVEL_M) return "Move farther around the child"
        if (coverageDegrees() < MIN_COVERAGE_DEGREES) return "Capture a wider arc around the child"
        if (floorStabilityCm() > MAX_FLOOR_STABILITY_CM) return "Floor tracking was unstable"
        return "Depth measurements were inconsistent"
    }

    private fun hasMinimumCoverage(): Boolean =
        frames.size >= MIN_KEYFRAMES &&
            cameraTravelMeters >= MIN_CAMERA_TRAVEL_M &&
            coverageDegrees() >= MIN_COVERAGE_DEGREES &&
            floorStabilityCm() <= MAX_FLOOR_STABILITY_CM

    private fun coverageDegrees(): Double = angularSpan(frames.map { it.pose.yawDegrees })

    private fun floorStabilityCm(): Double {
        if (frames.size < 2) return 0.0
        val floors = frames.map { it.floorY }.sorted()
        return (percentile(floors, 0.90) - percentile(floors, 0.10)) * 100.0
    }

    private fun distance(first: ScanPose, second: ScanPose): Double {
        val dx = second.x - first.x
        val dy = second.y - first.y
        val dz = second.z - first.z
        return sqrt(dx * dx + dy * dy + dz * dz)
    }

    companion object {
        const val MIN_KEYFRAMES = 12
        const val TARGET_KEYFRAMES = 20
        const val MAX_KEYFRAMES = 32
        const val MIN_BODY_POINTS = 60
        const val MIN_MEAN_CONFIDENCE = 0.50
        const val MIN_CAMERA_TRAVEL_M = 0.25
        const val TARGET_CAMERA_TRAVEL_M = 0.40
        const val MIN_COVERAGE_DEGREES = 20.0
        const val TARGET_COVERAGE_DEGREES = 35.0
        const val MAX_FLOOR_STABILITY_CM = 5.0
        const val MAX_HEIGHT_UNCERTAINTY_CM = 6.0

        private const val MIN_HEIGHT_CM = 35.0
        private const val MAX_HEIGHT_CM = 145.0
        private const val FLOOR_WARMUP_FRAMES = 3
        private const val MAX_FLOOR_DEVIATION_M = 0.06
        private const val MAX_BODY_DRIFT_M = 0.35
        private const val MIN_KEYFRAME_TRANSLATION_M = 0.045
        private const val MIN_KEYFRAME_YAW_DEGREES = 4.0
        private const val MAD_TO_SIGMA = 1.4826

        internal fun median(values: List<Double>): Double = percentile(values.sorted(), 0.50)

        internal fun percentile(sortedValues: List<Double>, fraction: Double): Double {
            require(sortedValues.isNotEmpty())
            if (sortedValues.size == 1) return sortedValues.first()
            val position = fraction.coerceIn(0.0, 1.0) * (sortedValues.size - 1)
            val lower = position.toInt()
            val upper = min(lower + 1, sortedValues.lastIndex)
            val weight = position - lower
            return sortedValues[lower] * (1.0 - weight) + sortedValues[upper] * weight
        }

        internal fun angularDistance(first: Double, second: Double): Double {
            val raw = abs(normalizeDegrees(first) - normalizeDegrees(second))
            return min(raw, 360.0 - raw)
        }

        internal fun angularSpan(values: List<Double>): Double {
            if (values.size < 2) return 0.0
            val normalized = values.map(::normalizeDegrees).sorted()
            var largestGap = 0.0
            for (index in 0 until normalized.lastIndex) {
                largestGap = maxOf(largestGap, normalized[index + 1] - normalized[index])
            }
            largestGap = maxOf(largestGap, normalized.first() + 360.0 - normalized.last())
            return 360.0 - largestGap
        }

        private fun normalizeDegrees(value: Double): Double = ((value % 360.0) + 360.0) % 360.0
    }
}
