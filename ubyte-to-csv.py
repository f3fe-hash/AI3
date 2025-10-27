import numpy as np
import struct
import os

train_images = "datasets/fashion/train-images-idx3-ubyte"
train_labels = "datasets/fashion/train-labels-idx1-ubyte"
output_csv   = "datasets/fashion/fashion_mnist_train.csv"

def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num, rows * cols)
    return images

def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

def save_to_csv(images_file, labels_file, output_file):
    images = load_mnist_images(images_file)
    labels = load_mnist_labels(labels_file)
    assert images.shape[0] == labels.shape[0], "Image and label counts do not match"

    # Prepend labels as first column
    combined = np.column_stack((labels, images))

    # Save as CSV
    np.savetxt(output_file, combined, fmt='%d', delimiter=',')
    print(f"Saved {combined.shape[0]} samples to {output_file}")

save_to_csv(train_images, train_labels, output_csv)
