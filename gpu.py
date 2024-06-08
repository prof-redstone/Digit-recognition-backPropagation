import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import struct
import pickle
import time
import tkinter as tk
import matplotlib.pyplot as plt

def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

# Read the files
train_images = read_idx('data\\augmented-train-images.idx3-ubyte')
train_labels = read_idx('data\\augmented-train-labels.idx1-ubyte')
test_images = read_idx('data\\t10k-images.idx3-ubyte')
test_labels = read_idx('data\\t10k-labels.idx1-ubyte')

train_images = torch.tensor(train_images / 255.0, dtype=torch.float32).unsqueeze(1)
train_labels = torch.tensor(train_labels, dtype=torch.long)  # Not one-hot encoded for CrossEntropyLoss
test_images = torch.tensor(test_images / 255.0, dtype=torch.float32).unsqueeze(1)
test_labels = torch.tensor(test_labels, dtype=torch.long)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.sigmoid(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

input_size = 28 * 28
hidden_size = 70
output_size = 10

model = NeuralNet(input_size, hidden_size, output_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=10.0)

def save_parameters(filename, model):
    torch.save(model.state_dict(), filename)

def load_parameters(model):
    model.load_state_dict(torch.load("param/gweight.pkl"))
    return model

def generate_filename():
    timestamp = int(time.time())
    filename = f"param/gparameters_{timestamp}.pkl"
    return filename

def train():
    epochs = 1001
    for epoch in range(epochs):
        model.train()
        inputs = train_images.view(-1, 28*28).to(device)
        labels = train_labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            test()
            save_parameters(generate_filename(), model)
            print("epochs:", epoch, "loss:", loss.item())

def test():
    model.eval()
    with torch.no_grad():
        inputs = test_images.view(-1, 28*28).to(device)
        labels = test_labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == labels).sum().item() / labels.size(0)
        print(f'Test Accuracy: {accuracy}')

def process_forward(x1):
    x1 = torch.tensor(x1, dtype=torch.float32).to(device)
    with torch.no_grad():
        outputs = model(x1)
    return outputs.cpu().numpy()

def process_drawing(tab):
    a2 = process_forward(tab.reshape(-1, 784))
    printRes(a2)

def printRes(tab):
    sum_values = np.sum(tab[0])
    print("\n\n")
    for i in range(len(tab[0])):
        bar = ""
        for j in range(int(tab[0][i] / sum_values * 30)):
            bar += "#"
        print(str(i), bar)

class DrawInterface:
    def __init__(self, root):
        self.root = root
        self.root.title("Draw Digits")
        self.pixel_size = 15
        self.grid_size = 28
        self.brush_size = 2  # Brush size
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
        # Apply the brush with shading on the edges
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
    load_parameters(model)
    #train()
    test()
    gess()
