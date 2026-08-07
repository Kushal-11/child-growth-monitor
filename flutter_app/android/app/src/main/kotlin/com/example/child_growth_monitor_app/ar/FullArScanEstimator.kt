package com.example.child_growth_monitor_app.ar

import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.min
import kotlin.math.PI
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
    val profile: BodyProfileEvidence? = null,
)

internal data class BodyProfileEvidence(
    val shoulderSpanCm: Double?,
    val hipSpanCm: Double?,
    val chestSpanCm: Double?,
    val abdomenSpanCm: Double?,
    val armSpanCm: Double?,
    val torsoLengthCm: Double? = null,
    val upperArmLengthCm: Double? = null,
    val poseQualityScore: Double? = null,
)

internal data class ContactlessBodyGeometry(
    val shoulderWidthCm: Double,
    val hipWidthCm: Double,
    val torsoLengthCm: Double,
    val upperArmLengthCm: Double,
    val chestDepthCm: Double,
    val abdomenDepthCm: Double,
    val estimatedMuacCm: Double?,
    val muacUncertaintyCm: Double?,
    val poseQualityScore: Double,
    val qualityScore: Double,
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
    val geometry: ContactlessBodyGeometry?,
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
        val geometry = summarizeGeometry(heightMedian)

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
            geometry = geometry,
        )
    }

    private fun summarizeGeometry(heightCm: Double): ContactlessBodyGeometry? {
        if (frames.size < MIN_KEYFRAMES) return null
        val firstYaw = frames.first().pose.yawDegrees
        val front = frames.filter {
            angularDistance(firstYaw, it.pose.yawDegrees) <= MAX_FRONT_VIEW_DEGREES
        }
        val side = frames.filter {
            angularDistance(firstYaw, it.pose.yawDegrees) >= MIN_SIDE_VIEW_DEGREES
        }

        fun robust(values: List<Double?>, minimum: Int = MIN_PROFILE_SAMPLES): Double? {
            val finite = values.filterNotNull().filter { it.isFinite() && it > 0 }.sorted()
            if (finite.size < minimum) return null
            return median(finite)
        }

        val shoulderWidth = robust(front.map { it.profile?.shoulderSpanCm }) ?: return null
        val hipWidth = robust(front.map { it.profile?.hipSpanCm }) ?: return null
        val chestDepth = robust(side.map { it.profile?.chestSpanCm }) ?: return null
        val abdomenDepth = robust(side.map { it.profile?.abdomenSpanCm }) ?: return null
        val torsoLength = robust(frames.map { it.profile?.torsoLengthCm }) ?: return null
        val upperArmLength = robust(frames.map { it.profile?.upperArmLengthCm }) ?: return null
        val poseQuality = robust(frames.map { it.profile?.poseQualityScore }) ?: return null

        if (shoulderWidth !in MIN_TORSO_SPAN_CM..MAX_TORSO_SPAN_CM ||
            hipWidth !in MIN_TORSO_SPAN_CM..MAX_TORSO_SPAN_CM ||
            chestDepth !in MIN_DEPTH_SPAN_CM..MAX_DEPTH_SPAN_CM ||
            abdomenDepth !in MIN_DEPTH_SPAN_CM..MAX_DEPTH_SPAN_CM ||
            torsoLength !in MIN_TORSO_LENGTH_CM..MAX_TORSO_LENGTH_CM ||
            upperArmLength !in MIN_UPPER_ARM_LENGTH_CM..MAX_UPPER_ARM_LENGTH_CM ||
            poseQuality !in MIN_POSE_QUALITY..1.0
        ) {
            return null
        }

        val frontArm = robust(front.map { it.profile?.armSpanCm })
        val sideArm = robust(side.map { it.profile?.armSpanCm })
        val muac = ellipsePerimeter(frontArm, sideArm)
        val muacSamples = frames.mapNotNull { frame ->
            val angle = angularDistance(firstYaw, frame.pose.yawDegrees)
            val transverse = frame.profile?.armSpanCm ?: return@mapNotNull null
            when {
                angle <= MAX_FRONT_VIEW_DEGREES && sideArm != null ->
                    ellipsePerimeter(transverse, sideArm)
                angle >= MIN_SIDE_VIEW_DEGREES && frontArm != null ->
                    ellipsePerimeter(frontArm, transverse)
                else -> null
            }
        }
        val muacUncertainty = if (muac != null && muacSamples.size >= MIN_PROFILE_SAMPLES) {
            val deviations = muacSamples.map { abs(it - muac) }
            (median(deviations) * MAD_TO_SIGMA).coerceAtLeast(MIN_MUAC_UNCERTAINTY_CM)
        } else {
            null
        }

        val frontCoverage = (front.size.toDouble() / TARGET_PROFILE_SAMPLES).coerceIn(0.0, 1.0)
        val sideCoverage = (side.size.toDouble() / TARGET_PROFILE_SAMPLES).coerceIn(0.0, 1.0)
        val armQuality = if (muac != null) 1.0 else 0.0
        val geometryQuality = (
            0.30 * frontCoverage +
                0.30 * sideCoverage +
                0.25 * poseQuality +
                0.15 * armQuality
            )
            .coerceIn(0.0, 1.0)

        return ContactlessBodyGeometry(
            shoulderWidthCm = shoulderWidth,
            hipWidthCm = hipWidth,
            torsoLengthCm = torsoLength,
            upperArmLengthCm = upperArmLength,
            chestDepthCm = chestDepth,
            abdomenDepthCm = abdomenDepth,
            estimatedMuacCm = muac,
            muacUncertaintyCm = muacUncertainty,
            poseQualityScore = poseQuality,
            qualityScore = geometryQuality,
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
        const val MIN_CAMERA_TRAVEL_M = 0.50
        const val TARGET_CAMERA_TRAVEL_M = 0.80
        const val MIN_COVERAGE_DEGREES = 70.0
        const val TARGET_COVERAGE_DEGREES = 90.0
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
        private const val MAX_FRONT_VIEW_DEGREES = 30.0
        private const val MIN_SIDE_VIEW_DEGREES = 55.0
        private const val MIN_PROFILE_SAMPLES = 2
        private const val TARGET_PROFILE_SAMPLES = 5.0
        private const val MIN_TORSO_SPAN_CM = 5.0
        private const val MAX_TORSO_SPAN_CM = 45.0
        private const val MIN_DEPTH_SPAN_CM = 3.0
        private const val MAX_DEPTH_SPAN_CM = 35.0
        private const val MIN_TORSO_LENGTH_CM = 8.0
        private const val MAX_TORSO_LENGTH_CM = 60.0
        private const val MIN_UPPER_ARM_LENGTH_CM = 5.0
        private const val MAX_UPPER_ARM_LENGTH_CM = 35.0
        private const val MIN_POSE_QUALITY = 0.45
        private const val MIN_ARM_DIAMETER_CM = 2.0
        private const val MAX_ARM_DIAMETER_CM = 10.0
        private const val MIN_MUAC_CM = 7.0
        private const val MAX_MUAC_CM = 24.0
        private const val MIN_MUAC_UNCERTAINTY_CM = 0.3
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

        internal fun ellipsePerimeter(
            firstDiameterCm: Double?,
            secondDiameterCm: Double?,
        ): Double? {
            if (firstDiameterCm == null || secondDiameterCm == null ||
                !firstDiameterCm.isFinite() || !secondDiameterCm.isFinite() ||
                firstDiameterCm !in MIN_ARM_DIAMETER_CM..MAX_ARM_DIAMETER_CM ||
                secondDiameterCm !in MIN_ARM_DIAMETER_CM..MAX_ARM_DIAMETER_CM
            ) {
                return null
            }
            val a = firstDiameterCm / 2.0
            val b = secondDiameterCm / 2.0
            val perimeter = PI * (3.0 * (a + b) - sqrt((3.0 * a + b) * (a + 3.0 * b)))
            return perimeter.takeIf { it in MIN_MUAC_CM..MAX_MUAC_CM }
        }

        private fun normalizeDegrees(value: Double): Double = ((value % 360.0) + 360.0) % 360.0
    }
}
