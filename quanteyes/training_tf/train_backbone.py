import os

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow import keras

from quanteyes.dataloader.dataset_tf import get_zipped_dataset

# Define your data loading and preprocessing
# base_path = "/data/openEDS2020-GazePrediction-2bit"
# base_path = "/data/openEDS2020-GazePrediction-2bit-octree"
base_path = "/data/openEDS2020-GazePrediction-1bit-otsu"
base_path = "/data/openEDS2020-GazePrediction-1bit-edge"


train_dataset = get_zipped_dataset(f"{base_path}/train").shuffle(1000)
val_dataset = get_zipped_dataset(f"{base_path}/validation").shuffle(1000)

# Define your model
model = keras.Sequential(
    [
        keras.layers.Conv2D(
            16, (3, 3), activation="relu", input_shape=(200, 320, 1), padding="same"
        ),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(3),
    ]
)

quantize_model = tfmot.quantization.keras.quantize_model

# q_aware stands for for quantization aware.
q_aware_model = quantize_model(model)

# `quantize_model` requires a recompile.
q_aware_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="mean_squared_error",
    metrics=["mse"],
)

q_aware_model.summary()

q_aware_model.fit(
    train_dataset.batch(32),
    epochs=10,
    steps_per_epoch=100,
    validation_data=val_dataset.batch(32).take(100),
)

q_aware_model.evaluate(val_dataset.batch(32).take(100))

converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8


def representative_dataset():
    for data in train_dataset.batch(1).take(100):
        yield [data[0]]


converter.representative_dataset = representative_dataset

quantized_tflite_model = converter.convert()


# Save the FLOAT16 quantized model to disk

SAVED_MODEL_FILENAME = "./model_export/int8_quantized.tflite"
open(SAVED_MODEL_FILENAME, "wb").write(quantized_tflite_model)

# Calculate size
size = os.path.getsize(SAVED_MODEL_FILENAME)
print(f"Size of {SAVED_MODEL_FILENAME} is {size} bytes")
