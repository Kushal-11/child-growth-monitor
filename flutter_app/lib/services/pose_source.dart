import '../models/body_measurements.dart';

/// Adapter interface used by AssessmentService to obtain body segments.
/// Both the real PoseService and test stubs implement this.
abstract class PoseSource {
  /// Returns front-view body segments derived from the image at [path].
  /// Returns a `BodySegments` whose `totalHeightPx` is null when pose
  /// detection failed.
  Future<BodySegments> segmentsFor(String path);

  /// Returns side-view depth segments, or null if unavailable.
  Future<SideViewSegments?> sideSegmentsFor(String path);

  /// Returns a 0..1 confidence score for the front-view detection.
  Future<double> confidenceFor(String path);
}
