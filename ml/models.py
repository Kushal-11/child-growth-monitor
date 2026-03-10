"""
TensorFlow/Keras model architectures for the wasting detection pipeline.

Models use a 14-feature input vector:
  [age_months, sex_binary, height_cm, shoulder_width_cm, hip_width_cm,
   torso_length_cm, upper_arm_length_cm, shoulder_height_ratio,
   hip_height_ratio, body_build_score,
   chest_depth_cm, abd_depth_cm, chest_depth_ratio, abd_depth_ratio]

Features 10-13 are AP (anterior-posterior) depth measurements from a side-view
photo. When no side view is provided, they are imputed from lateral widths using
Snyder 1975 AP/lateral ratios. A single model handles both cases.

All are small MLPs (< 500 KB each) exportable to TFLite for mobile deployment.
"""

FEATURE_NAMES = [
    "age_months",
    "sex_binary",
    "height_cm",
    "shoulder_width_cm",
    "hip_width_cm",
    "torso_length_cm",
    "upper_arm_length_cm",
    "shoulder_height_ratio",
    "hip_height_ratio",
    "body_build_score",
    # Side-view depth features (measured or imputed)
    "chest_depth_cm",    # AP diameter at chest/shoulder level
    "abd_depth_cm",      # AP diameter at abdomen/hip level
    "chest_depth_ratio", # chest_depth / height
    "abd_depth_ratio",   # abd_depth / height
]

N_FEATURES = len(FEATURE_NAMES)

# Wasting classifier labels (must match generate_synthetic_data.py)
WASTING_LABELS = ["MAM", "Normal", "Overweight", "Risk_Overweight", "SAM"]
N_CLASSES = len(WASTING_LABELS)

# Clinical priority order for ensemble (most severe first)
LABEL_PRIORITY = {"SAM": 0, "MAM": 1, "Normal": 2, "Risk_Overweight": 3, "Overweight": 4}


def build_weight_estimator():
    """
    Regression MLP: predicts child weight (kg) from body proportions.
    Predicted weight → WHZ via WHO LMS → wasting classification.
    ~64 KB when compiled.
    """
    import tensorflow as tf
    inp = tf.keras.Input(shape=(N_FEATURES,), name="features")
    x = tf.keras.layers.Dense(64, activation="relu")(inp)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    out = tf.keras.layers.Dense(1, name="weight_kg")(x)
    model = tf.keras.Model(inp, out, name="weight_estimator")
    model.compile(optimizer="adam", loss="huber", metrics=["mae"])
    return model


def build_wasting_classifier():
    """
    5-class MLP: predicts SAM / MAM / Normal / Risk_Overweight / Overweight.
    Primary deployment model. ~200 KB when compiled.
    """
    import tensorflow as tf
    inp = tf.keras.Input(shape=(N_FEATURES,), name="features")
    x = tf.keras.layers.Dense(128, activation="relu")(inp)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    out = tf.keras.layers.Dense(N_CLASSES, activation="softmax", name="class_probs")(x)
    model = tf.keras.Model(inp, out, name="wasting_classifier")
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
