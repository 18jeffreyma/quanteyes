import tensorflow as tf


def float32_quantize(q_aware_model, train_dataset, n=100):
    converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_dataset():
        for data in train_dataset.batch(1).take(n):
            yield [data[0]]

    converter.representative_dataset = representative_dataset
    return converter


def float16_quantize(q_aware_model, train_dataset, n=100):
    converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]

    def representative_dataset():
        for data in train_dataset.batch(1).take(n):
            yield [data[0]]

    converter.representative_dataset = representative_dataset
    return converter


def int8_quantize(q_aware_model, train_dataset, n=100):
    converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # Ensure that if any ops can't be quantized, the converter throws an error
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    # Set the input and output tensors to uint8 (APIs added in r2.3)
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    def representative_dataset():
        for data in train_dataset.batch(1).take(n):
            yield [data[0]]

    converter.representative_dataset = representative_dataset
    return converter


DATA_PATHS = {
    # "original": "/data/openEDS2020-GazePrediction",
    # "2bit": "/data/openEDS2020-GazePrediction-2bit",
    # "2bit-octree": "/data/openEDS2020-GazePrediction-2bit-octree",
    # "1bit-edge": "/data/openEDS2020-GazePrediction-1bit-edge",
    "1bit-otsu": "/data/openEDS2020-GazePrediction-1bit-otsu",
}

QUANTIZATIONS = {
    "float32": float32_quantize,
    "float16": float16_quantize,
    "int8": int8_quantize,
}
