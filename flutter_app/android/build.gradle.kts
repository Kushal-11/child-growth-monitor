allprojects {
    repositories {
        google()
        mavenCentral()
    }
}

val newBuildDir: Directory =
    rootProject.layout.buildDirectory
        .dir("../../build")
        .get()
rootProject.layout.buildDirectory.value(newBuildDir)

subprojects {
    val newSubprojectBuildDir: Directory = newBuildDir.dir(project.name)
    project.layout.buildDirectory.value(newSubprojectBuildDir)
}
subprojects {
    project.evaluationDependsOn(":app")
}

// tflite_flutter 0.11.0 pins its Java compileOptions to 1.8 while its Kotlin compile
// task inherits the JDK 17 toolchain default. Under Gradle 8.14 + AGP 8.11 + Kotlin
// 2.2 that Java(1.8)/Kotlin(17) split fails the build with "Inconsistent JVM-target
// compatibility". We can't edit the plugin's build.gradle, and its AGP compileOptions
// are already finalized by the time this root block runs (evaluationDependsOn(":app")
// + Flutter's plugin loader force early evaluation), so we relax the Kotlin JVM-target
// validation to a warning. Our own app module already sets both Java and Kotlin to 17,
// so this only affects the third-party mismatch we don't control.
subprojects {
    tasks.withType<org.jetbrains.kotlin.gradle.tasks.KotlinCompile>().configureEach {
        compilerOptions {
            jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
        }
        jvmTargetValidationMode.set(org.jetbrains.kotlin.gradle.dsl.jvm.JvmTargetValidationMode.WARNING)
    }
}

tasks.register<Delete>("clean") {
    delete(rootProject.layout.buildDirectory)
}
