plugins {
    id("com.android.application")
    id("kotlin-android")
    // The Flutter Gradle Plugin must be applied after the Android and Kotlin Gradle plugins.
    id("dev.flutter.flutter-gradle-plugin")
}

android {
    namespace = "com.example.child_growth_monitor_app"
    compileSdk = flutter.compileSdkVersion
    ndkVersion = flutter.ndkVersion

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = JavaVersion.VERSION_17.toString()
    }

    defaultConfig {
        // TODO: Specify your own unique Application ID (https://developer.android.com/studio/build/application-id.html).
        applicationId = "com.example.child_growth_monitor_app"
        // You can update the following values to match your application needs.
        // For more information, see: https://flutter.dev/to/review-gradle-config.
        minSdk = flutter.minSdkVersion
        targetSdk = flutter.targetSdkVersion
        versionCode = flutter.versionCode
        versionName = flutter.versionName

        // tflite_flutter 0.12.1 / LiteRT 1.4.0 ships pre-compiled native .so
        // files for these four ABIs. Keep every supported ABI in universal APKs.
        ndk {
            abiFilters += listOf("armeabi-v7a", "arm64-v8a", "x86", "x86_64")
        }
    }

    // tflite_flutter uses dart:ffi DynamicLibrary.open() to load
    // libtensorflowlite_jni.so at runtime. AGP 8.x release builds
    // DEFLATE-compress AAR-sourced .so files when minSdk < 23 by default,
    // making them unloadable via dlopen. useLegacyPackaging = false forces
    // uncompressed storage regardless of minSdk, matching debug-build
    // behaviour so extraction and direct-mmap both work.
    packagingOptions {
        jniLibs {
            useLegacyPackaging = false
        }
    }

    buildTypes {
        release {
            // TODO: Add your own signing config for the release build.
            // Signing with the debug keys for now, so `flutter run --release` works.
            signingConfig = signingConfigs.getByName("debug")
            // Keep rules for TensorFlow Lite / ML Kit so R8 minification doesn't strip
            // reflectively-loaded inference classes or abort on optional GPU delegates.
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro",
            )
        }
    }
}

flutter {
    source = "../.."
}

dependencies {
    // Optional at runtime: unsupported devices keep the existing lightweight
    // guided-camera workflow instead of being excluded from installation.
    implementation("com.google.ar:core:1.48.0")
    testImplementation("junit:junit:4.13.2")
}
