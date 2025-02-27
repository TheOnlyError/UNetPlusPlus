from typing import Tuple

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def load_data(buffer_size=400, **kwargs) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    dataset = loadDataset().shuffle(buffer_size)
    DATASET_SIZE = len(list(dataset))
    print(DATASET_SIZE)
    train_size = int(0.8 * DATASET_SIZE)
    val_size = int(0.2 * DATASET_SIZE)

    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size)
    return train_dataset, test_dataset


def _parse_function(example_proto):
    feature = {'image': tf.io.FixedLenFeature([], tf.string),
               'boundary': tf.io.FixedLenFeature([], tf.string)}
    example = tf.io.parse_single_example(example_proto, feature)

    image, mask = decodeAllRaw(example)
    return preprocess(image, mask)


def decodeAllRaw(x):
    image = tf.io.decode_raw(x['image'], tf.uint8)
    mask = tf.io.decode_raw(x['boundary'], tf.uint8)
    return image, mask


def preprocess(img, mask, size=512):  # 1024
    img = tf.cast(img, dtype=tf.float32)
    img = tf.reshape(img, [size, size, 3]) / 255
    mask = tf.reshape(mask, [size, size, 1])
    return img, mask


def loadDataset():
    raw_dataset = tf.data.TFRecordDataset('data.tfrecords')
    parsed_dataset = raw_dataset.map(_parse_function)
    return parsed_dataset


if __name__ == "__main__":
    train_dataset, validation_dataset = load_data()
    # counts = [0, 0, 0]
    # classes = 3
    # for (image, mask) in train_dataset.batch(1):
    #     for c in range(classes):
    #         counts[c] += np.count_nonzero(mask == c)
    # print(counts)
    # weights = sum(counts) / np.array(counts)
    # print(weights)

    rows = 10
    fig, axs = plt.subplots(rows, 2, figsize=(8, 30))
    for ax, (image, mask) in zip(axs, validation_dataset.take(rows).batch(1)):
        ax[0].matshow(np.array(image[0]).squeeze())
        ax[1].matshow(np.array(mask[0]).squeeze())
    plt.show()
