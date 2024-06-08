import numpy as np
import struct
import pickle
import time
import tkinter as tk
import scipy.ndimage
import matplotlib.pyplot as plt
import random

load = True

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

print(test_images[100])
print(test_labels[100])
print("taille data train ", len(train_images))
print("taille data test ", len(train_images))

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

def save_parameters(filename, w1, b1, w2, b2):
    parameters = {
        'w1': w1,
        'b1': b1,
        'w2': w2,
        'b2': b2
    }
    with open(filename, 'wb') as f:
        pickle.dump(parameters, f)

def load_parameters(filename):
    with open(filename, 'rb') as f:
        parameters = pickle.load(f)
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']
    return w1, b1, w2, b2

def generate_filename():
    timestamp = int(time.time())  # Obtenir le timestamp actuel en secondes
    filename = f"parameters_{timestamp}.pkl"  # Générer le nom du fichier avec le timestamp
    return filename

if load :
    w1, b1, w2, b2 = load_parameters('weight.pkl')


def display_before_after(sample_images):
    fig, axes = plt.subplots(len(sample_images), 2, figsize=(10, len(sample_images) * 5))
    for i, img in enumerate(sample_images):
        
        axes[i, 0].imshow(img, cmap='gray')
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')
    
    plt.tight_layout()
    plt.show()

# Sélectionner un échantillon d'images (par exemple, les 5 premières)
sample_images = train_images[:5]

# Afficher avant et après augmentation pour l'échantillon d'images
#display_before_after(sample_images)

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

def train():
    epochs = 1000
    for epoch in range(epochs):
        z1, a1, z2, a2 = process_forward(train_images.reshape(-1, 784))
        loss = calcul_loss(train_labels, a2)
        backpropag(train_images.reshape(-1, 784), train_labels, z1, a1, z2, a2)
        if epoch % 100 == 0:
            save_parameters(generate_filename(), w1, b1, w2, b2)
            print("epochs :", epoch, "loss :", loss)

def test():
    _, _, _, a2 = process_forward(test_images.reshape(-1, 784))
    accuracy = np.mean(np.argmax(a2, axis=1) == np.argmax(test_labels, axis=1))
    print(f'Test Accuracy: {accuracy}')

def process_drawing(tab):
    _, _, _, a2 = process_forward(tab.reshape(-1, 784))
    printRes(a2)

def printRes(tab):

    sum = 0
    for i in tab[0]:
        sum += i
    print("\n\n")
    for i in range(len(tab[0])):
        b = ""
        for j in range(int(tab[0][i]/sum*30)):
            b += "#"
        print(str(i), b)
        pass

class DrawInterface:
    def __init__(self, root):
        self.root = root
        self.root.title("Draw Digits")
        self.pixel_size = 15
        self.grid_size = 28
        self.brush_size = 2  # Taille du crayon
        self.canvas = tk.Canvas(root, width=self.grid_size*self.pixel_size, height=self.grid_size*self.pixel_size, bg='black')
        self.canvas.grid(row=0, column=0, columnspan=self.grid_size)
        
        self.reset_button = tk.Button(root, text="Reset", command=self.reset)
        self.reset_button.grid(row=1, column=0, columnspan=self.grid_size)

        self.drawing = False
        self.grid_data = np.zeros((self.grid_size, self.grid_size))

        self.canvas.bind("<Button-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)

    def start_drawing(self, event):
        self.drawing = True
        self.draw(event)

    def draw(self, event):
        if self.drawing:
            x, y = event.x, event.y
            col, row = x // self.pixel_size, y // self.pixel_size
            if 0 <= col < self.grid_size and 0 <= row < self.grid_size:
                self.apply_brush(col, row)
        process_drawing(self.grid_data)

    def stop_drawing(self, event):
        self.drawing = False

    def apply_brush(self, col, row):
        # Appliquer le pinceau avec des nuances sur les bords
        for i in range(-self.brush_size, self.brush_size + 1):
            for j in range(-self.brush_size, self.brush_size + 1):
                new_col, new_row = col + i, row + j
                if 0 <= new_col < self.grid_size and 0 <= new_row < self.grid_size:
                    distance = np.sqrt(i**2 + j**2)
                    if distance <= self.brush_size:
                        intensity = max(0, 1 - (distance / self.brush_size))
                        self.grid_data[new_row][new_col] = min(1.0, self.grid_data[new_row][new_col] + intensity)
                        gray_value = int(255 * self.grid_data[new_row][new_col])
                        self.canvas.create_rectangle(new_col*self.pixel_size, new_row*self.pixel_size, 
                                                     (new_col+1)*self.pixel_size, (new_row+1)*self.pixel_size, 
                                                     fill=f'#{gray_value:02x}{gray_value:02x}{gray_value:02x}', outline='')

    def reset(self):
        self.canvas.delete("all")
        self.grid_data = np.zeros((self.grid_size, self.grid_size))
        self.canvas.configure(bg='black')

def gess():
    root = tk.Tk()
    app = DrawInterface(root)
    root.mainloop()

if __name__ == "__main__":
    test()
    #train()
    #gess()
    pass