import os

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow import keras
from utils import DATA_PATHS, QUANTIZATIONS

from quanteyes.dataloader.dataset_tf import get_zipped_dataset

for d_name, base_path in DATA_PATHS.items():
    train_dataset = get_zipped_dataset(f"{base_path}/train", train=True).shuffle(10000)
    val_dataset = get_zipped_dataset(f"{base_path}/validation", train=False).shuffle(
        10000
    )

    # Define your model with dropout.
    model = keras.Sequential(
        [
            keras.layers.Conv2D(
                32,
                (3, 3),
                input_shape=(100, 160, 1),
                padding="same",
            ),
            keras.layers.BatchNormalization(),
            keras.layers.Activation("relu"),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(
                32,
                (3, 3),
                padding="same",
            ),
            keras.layers.BatchNormalization(),
            keras.layers.Activation("relu"),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Dropout(0.3),
    
            keras.layers.Conv2D(64, (3, 3), padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.Activation("relu"),
            keras.layers.Conv2D(64, (3, 3), padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.Activation("relu"),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Dropout(0.5),
    
            keras.layers.Conv2D(128, (3, 3), padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.Activation("relu"),
            keras.layers.Conv2D(128, (3, 3), padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.Activation("relu"),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Dropout(0.5),

            keras.layers.Flatten(),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(3),
        ]
    )

    quantize_model = tfmot.quantization.keras.quantize_model

    # q_aware stands for for quantization aware.
    # q_aware_model = quantize_model(model)
    q_aware_model = quantize_model(model)

    # `quantize_model` requires a recompile.
    q_aware_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        # loss=tf.keras.losses.CosineSimilarity(
        #     axis=1, reduction=tf.keras.losses.Reduction.SUM
        # ),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=["mse", "cosine_similarity"],
    )

    # q_aware_model.summary()

    q_aware_model.fit(
        train_dataset.batch(32).repeat(20),
        epochs=40,
        steps_per_epoch=2000,
        validation_data=val_dataset.batch(32).take(100),
    )
    
    # Take first element of val_dataset and perform prediction.
    for input_data, target in val_dataset.batch(1).take(5):
        print("input_data", input_data)
        print("target", target)
        pred = q_aware_model.predict(input_data)
        print("pred", pred)

    for q_name, quantization in QUANTIZATIONS.items():
        converter = quantization(q_aware_model, train_dataset)
        quantized_tflite_model = converter.convert()

        SAVED_MODEL_FILENAME = f"./model_export/q-{q_name}_d-{d_name}.tflite"
        open(SAVED_MODEL_FILENAME, "wb").write(quantized_tflite_model)
