import numpy as np
import struct

def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

# Lire les fichiers
train_images = read_idx('data\\train-images.idx3-ubyte')
train_labels = read_idx('data\\train-labels.idx1-ubyte')
test_images = read_idx('data\\t10k-images.idx3-ubyte')
test_labels = read_idx('data\\t10k-labels.idx1-ubyte')

train_images = train_images / 255.0
test_images = test_images / 255.0
#test

print(test_images[0])
print(test_labels[0])