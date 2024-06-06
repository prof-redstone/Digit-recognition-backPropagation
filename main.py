import numpy as np
import struct
import pickle

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


print(test_images[0])
print(test_labels[0])

train_images = train_images / 255.0
test_images = test_images / 255.0

def one_hot_encode(labels, num_classes=10): #vecteur de sortie de prediction
    return np.eye(num_classes)[labels]

train_labels = one_hot_encode(train_labels)
test_labels = one_hot_encode(test_labels)

input_size = 28*28
hidden_size = 70  #changer la val
output_size = 10

np.random.seed(0)
w1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros((1, hidden_size))
w2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros((1, output_size))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_der(x):
    return x * (1 - x)

def process_forward(x1):
    z1 = np.dot(x1, w1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, w2) + b2
    a2 = sigmoid(z2)
    return z1, a1, z2, a2

def calcul_loss(x, y):
    return np.mean((y-x)**2)

def backpropag(x,y,z1,a1,z2,a2):
    global w1, w2
    global b1, b2

    m = x.shape[0] #nb d'entree

    #grad calcul
    dz2 = a2 - y
    dW2 = np.dot(a1.T, dz2) / m
    db2 = np.sum(dz2, axis=0) / m
    
    dz1 = np.dot(dz2, w2.T) * sigmoid_der(a1)
    dW1 = np.dot(x.T, dz1) / m
    db1 = np.sum(dz1, axis=0) / m

    learning_rate = 0.1
    w1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    w2 -= learning_rate * dW2
    b2 -= learning_rate * db2

epochs = 1000
for epoch in range(epochs):
    z1, a1, z2, a2 = process_forward(train_images.reshape(-1, 784))
    loss = calcul_loss(train_labels, a2)
    backpropag(train_images.reshape(-1, 784), train_labels, z1, a1, z2, a2)
    if epoch % 100 == 0:
        print("epochs :", epochs, "loss :", loss)


_, _, _, a2 = process_forward(test_images.reshape(-1, 784))
accuracy = np.mean(np.argmax(a2, axis=1) == np.argmax(test_labels, axis=1))
print(f'Test Accuracy: {accuracy}')