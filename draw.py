import torch
import torch.nn as nn
import numpy as np
import tkinter as tk

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)  # Flatten the tensor
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)
        return x

model = CNN().to(device)

def load_parameters(filename, model):
    model.load_state_dict(torch.load(filename))
    return model



def process_drawing(tab):
    # Convert the NumPy array to a PyTorch tensor and move it to the correct device
    tab_tensor = torch.tensor(tab.reshape(-1, 1, 28, 28), dtype=torch.float32).to(device)
    outputs = model(tab_tensor)
    printRes(outputs.cpu().detach().numpy())

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
    load_parameters("param/g2weight.pkl", model)
    gess()
