package com.example.child_growth_monitor_app.ar

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

class FullArScanEstimatorTest {
    @Test
    fun `stable multi-view evidence produces a bounded summary`() {
        val estimator = FullArScanEstimator()
        repeat(20) { index ->
            val acceptance = estimator.tryAdd(evidence(index))
            assertTrue("frame $index should be accepted", acceptance.accepted)
        }

        assertTrue(estimator.readyToFinish())
        val summary = estimator.summarize(durationMs = 14_000)
        assertNotNull(summary)
        assertEquals(20, summary!!.acceptedKeyframes)
        assertEquals(90.0, summary.estimatedHeightCm, 0.6)
        assertTrue(summary.uncertaintyCm >= 0.2)
        assertTrue(summary.scanCoverageDegrees >= 35.0)
        assertTrue(summary.cameraTravelMeters >= 0.8)
        assertTrue(summary.qualityScore in 0.0..1.0)
        assertNotNull(summary.geometry)
        assertEquals(20.0, summary.geometry!!.shoulderWidthCm, 0.1)
        assertEquals(8.0, summary.geometry!!.chestDepthCm, 0.1)
        assertEquals(27.0, summary.geometry!!.torsoLengthCm, 0.1)
        assertEquals(14.0, summary.geometry!!.upperArmLengthCm, 0.1)
        assertEquals(0.9, summary.geometry!!.poseQualityScore, 0.01)
        assertNotNull(summary.geometry!!.estimatedMuacCm)
    }

    @Test
    fun `stationary duplicate view is rejected`() {
        val estimator = FullArScanEstimator()
        val first = evidence(0)
        assertTrue(estimator.tryAdd(first).accepted)
        val stationary = first.copy(
            pose = first.pose.copy(timestampNs = first.pose.timestampNs + 300_000_000L),
        )

        assertFalse(estimator.tryAdd(stationary).accepted)
        assertEquals(1, estimator.acceptedKeyframes)
    }

    @Test
    fun `unstable floor is rejected after warmup`() {
        val estimator = FullArScanEstimator()
        repeat(3) { assertTrue(estimator.tryAdd(evidence(it)).accepted) }
        val unstable = evidence(3).copy(floorY = 0.20)

        val acceptance = estimator.tryAdd(unstable)
        assertFalse(acceptance.accepted)
        assertTrue(acceptance.guidance.contains("same floor"))
    }

    @Test
    fun `angular span handles wraparound`() {
        assertEquals(
            20.0,
            FullArScanEstimator.angularSpan(listOf(350.0, 0.0, 10.0)),
            0.001,
        )
    }

    @Test
    fun `ellipse perimeter rejects missing and implausible arm evidence`() {
        assertEquals(12.6, FullArScanEstimator.ellipsePerimeter(4.0, 4.0)!!, 0.1)
        assertEquals(null, FullArScanEstimator.ellipsePerimeter(null, 4.0))
        assertEquals(null, FullArScanEstimator.ellipsePerimeter(1.0, 4.0))
    }

    private fun evidence(index: Int): DepthFrameEvidence = DepthFrameEvidence(
        heightCm = if (index % 2 == 0) 89.6 else 90.4,
        validDepthFraction = 0.45,
        meanConfidence = 0.82,
        floorY = if (index % 2 == 0) -0.005 else 0.005,
        bodyCenterX = 0.0,
        bodyCenterZ = -1.5,
        bodyPointCount = 220,
        pose = ScanPose(
            x = index * 0.06,
            y = 1.2,
            z = 0.0,
            yawDegrees = index * 5.0,
            timestampNs = index * 300_000_000L,
        ),
        profile = BodyProfileEvidence(
            shoulderSpanCm = if (index <= 6) 20.0 else null,
            hipSpanCm = if (index <= 6) 17.0 else null,
            chestSpanCm = if (index >= 11) 8.0 else null,
            abdomenSpanCm = if (index >= 11) 8.5 else null,
            armSpanCm = when {
                index <= 6 -> 4.0
                index >= 11 -> 3.8
                else -> null
            },
            torsoLengthCm = 27.0,
            upperArmLengthCm = 14.0,
            poseQualityScore = 0.9,
        ),
    )
}
