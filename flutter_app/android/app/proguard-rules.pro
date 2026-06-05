# TensorFlow Lite keep rules.
#
# tensorflow-lite references the optional GPU and NNAPI delegate classes at compile
# time even when they are not all present on the classpath, so R8 aborts the release
# build ("Missing class org.tensorflow.lite.gpu.GpuDelegateFactory$Options"). TFLite
# also loads delegates/interpreters via JNI + reflection, so the runtime classes must
# survive minification or inference fails at runtime.
-keep class org.tensorflow.lite.** { *; }
-keep class org.tensorflow.lite.gpu.** { *; }
-dontwarn org.tensorflow.lite.gpu.**
-dontwarn org.tensorflow.lite.gpu.GpuDelegateFactory$Options

# Google ML Kit pose detection (google_mlkit_pose_detection) loads native models and
# uses reflection; keep its classes to avoid stripping the pose pipeline in release.
-keep class com.google.mlkit.** { *; }
-keep class com.google.android.gms.internal.mlkit_vision_pose.** { *; }
-dontwarn com.google.mlkit.**
