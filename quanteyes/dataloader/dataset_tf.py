import tensorflow as tf


def load_and_decode_image(file_path):
    # Read the raw image data
    image = tf.io.read_file(file_path)
    
    # Log the file path and image shape (optional)
    # tf.print(file_path, tf.shape(image))
    
    # Decode the image (supports various image formats)
    image = tf.image.decode_png(image, channels=1)
    image = tf.reshape(image, [1, 400, 640, 1])
    # You may need to adjust the 'channels' parameter based on your images
    # Perform any additional preprocessing as needed
    # For example, you might want to resize the image
    # image = tf.image.resize(image, [640, 400])
    # Normalize pixel values to be in the range [0, 1]
    image = tf.cast(image, tf.float32) / 255.0

    # For image of [height, width, channels], perform 4x4 max pooling,
    # resulting in an output image of [height/4, width/4, channels]
    print("image shape", image.shape)
    
    pool = tf.keras.layers.MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')
    
    pooled_image = tf.reshape(pool(image), [100, 160, 1])
    
    return pooled_image


def read_label_for_sequence(label_file):
    file_contents = tf.strings.split(
        tf.strings.strip(tf.io.read_file(label_file)), "\r\n"
    )
    vectors_string = tf.map_fn(lambda x: tf.strings.split(x, ","), file_contents)
    vector_string_first_removed = tf.map_fn(lambda x: x[1:4], vectors_string)
    return tf.strings.to_number(vector_string_first_removed, tf.float32)


def get_zipped_dataset(path, train=True):
    image_paths = tf.data.Dataset.list_files(
        f"{path}/sequences/*/*.png", shuffle=False
    ).map(load_and_decode_image)
    labels = (
        # tf.data.Dataset.list_files(f"{path}/labels/*.txt", shuffle=False)
        # NOTE: This is a hack since the label file isn't properly copied over.
        tf.data.Dataset.list_files(
            f"/data/openEDS2020-GazePrediction-2bit-octree/{'train' if train else 'validation'}/labels/*.txt",
            shuffle=False,
        )
        .map(read_label_for_sequence)
        .flat_map(tf.data.Dataset.from_tensor_slices)
    )

    return tf.data.Dataset.zip((image_paths, labels))
