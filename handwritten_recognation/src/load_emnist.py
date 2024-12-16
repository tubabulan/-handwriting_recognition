import gzip
import numpy as np

def load_emnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
    return data.reshape(-1, 28, 28)

def load_emnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
    return data

if __name__ == '__main__':
    train_images = load_emnist_images('../data/emnist-balanced-train-images-idx3-ubyte.gz')
    train_labels = load_emnist_labels('../data/emnist-balanced-train-labels-idx1-ubyte.gz')
    print(f'Yüklendi: {len(train_images)} görüntü, {len(train_labels)} etiket')
