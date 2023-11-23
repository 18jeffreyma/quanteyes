# List all files in model_export.
import os
import re

import numpy as np
import tensorflow as tf

from quanteyes.dataloader.dataset_tf import get_zipped_dataset
from quanteyes.training_tf.utils import DATA_PATHS

data = {}

for dirname, _, filenames in os.walk("./model_export"):
    for filename in filenames:
        # Split out quantization type and dataset bit width from filename.
        quantization, data_width = [
            item.split("-")[1] for item in filename.split(".")[0].split("_")
        ]

        matches = re.search(r"q-(\w+)_d-(.+)\.", filename)
        if matches:
            quantization = matches.group(1)
            data_width = matches.group(2)
        else:
            raise Exception(f"Could not parse filename: {filename}")

        val_dataset = get_zipped_dataset(
            f"{DATA_PATHS[data_width]}/validation", train=False
        ).shuffle(1000)

        interpreter = tf.lite.Interpreter(model_path=os.path.join(dirname, filename))
        interpreter.allocate_tensors()

        # Get input and output tensors
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]

        input_tensor_index = input_details["index"]
        output = interpreter.get_tensor(output_details["index"])

        mse = 0.0
        avg_cosine_similarity = 0.0
        num_examples = 100

        # Perform inference on the validation dataset
        for input_data, target in val_dataset.batch(1).take(num_examples):
            # If required, quantize the input layer (from float to integer)
            img_array = input_data.numpy()
            print("sum of input: ", np.sum(img_array))

            input_scale, input_zero_point = input_details["quantization"]
            if (input_scale, input_zero_point) != (0.0, 0):
                img_array = np.multiply(img_array, 1.0 / input_scale) + input_zero_point
            img_array = img_array.astype(input_details["dtype"])

            interpreter.set_tensor(input_tensor_index, img_array)
            interpreter.invoke()
            pred = output[0]
            # print("pred", pred)

            # If required, dequantized the output layer (from integer to float)
            output_scale, output_zero_point = output_details["quantization"]
            if (output_scale, output_zero_point) != (0.0, 0):
                pred = pred.astype(np.float32)
                pred = np.multiply((pred - output_zero_point), output_scale)

            # Compute cosine similarity.
            y_pred = pred.astype(np.float32)
            y_true = np.squeeze(target.numpy().T).astype(np.float32)
            print(f"true: {y_true}, pred: {y_pred} \n")

            avg_cosine_similarity += -1 * tf.keras.losses.cosine_similarity(
                y_true, y_pred
            )
            mse += np.sum(np.square(y_true - y_pred))

        mse /= num_examples
        avg_cosine_similarity /= num_examples

        print(
            f"Quantization: {quantization}, Data width: {data_width}, MSE: {mse}, Cosine similarity: {avg_cosine_similarity}"
        )
        data.setdefault(
            quantization, {"mse": mse, "avg_cosine_similarity": avg_cosine_similarity}
        )

print(data)
