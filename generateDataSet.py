import numpy as np
import struct
import random
import scipy.ndimage

def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)
    
def augment_image(image):
    # Rotation
    angle = random.uniform(-15, 15)  # rotation angle between -15 and 15 degrees
    rotated_image = scipy.ndimage.rotate(image, angle, reshape=False)
    
    # Zoom
    zoom_factor = random.uniform(0.9, 1.1)  # zoom factor between 0.9 and 1.1
    zoomed_image = scipy.ndimage.zoom(rotated_image, zoom_factor)
    
    # Ensure the zoomed image has the correct shape
    if zoomed_image.shape[0] > 28:
        crop_margin = (zoomed_image.shape[0] - 28) // 2
        zoomed_image = zoomed_image[crop_margin:crop_margin+28, crop_margin:crop_margin+28]
    elif zoomed_image.shape[0] < 28:
        pad_margin = (28 - zoomed_image.shape[0]) // 2
        zoomed_image = np.pad(zoomed_image, ((pad_margin, 28 - zoomed_image.shape[0] - pad_margin), 
                                             (pad_margin, 28 - zoomed_image.shape[1] - pad_margin)), mode='constant')
    
    # Offset (décentrer légèrement l'image)
    shift_x = random.uniform(-2, 2)  # décalage horizontal
    shift_y = random.uniform(-2, 2)  # décalage vertical
    shifted_image = scipy.ndimage.shift(zoomed_image, shift=[shift_x, shift_y], mode='constant', cval=0)
    
    return shifted_image

def save_idx(images, labels, filename):
    with open(filename, 'wb') as f:
        # Write header
        f.write(struct.pack('>HBB', 0, 8, 3))  # 0, datatype (8 for unsigned byte), number of dimensions (3)
        f.write(struct.pack('>I', len(images)))  # Number of images
        f.write(struct.pack('>I', 28))  # Number of rows
        f.write(struct.pack('>I', 28))  # Number of columns
        
        # Write image data
        images = images.astype(np.uint8)
        f.write(images.tobytes())
        
    with open(filename.replace('-images', '-labels'), 'wb') as f:
        # Write header
        f.write(struct.pack('>HBB', 0, 8, 1))  # 0, datatype (8 for unsigned byte), number of dimensions (1)
        f.write(struct.pack('>I', len(labels)))  # Number of labels
        
        # Write label data
        labels = labels.astype(np.uint8)
        f.write(labels.tobytes())

# Lire les fichiers
train_images = read_idx('data/train-images.idx3-ubyte')
train_labels = read_idx('data/train-labels.idx1-ubyte')

augmented_images = []
augmented_labels = []

for img, label in zip(train_images, train_labels):
    augmented_images.append(img)  # Add the original image
    augmented_labels.append(label)
    for _ in range(10):  # Generate 10 augmented images for each original image
        augmented_images.append(augment_image(img))
        augmented_labels.append(label)

augmented_images = np.array(augmented_images)
augmented_labels = np.array(augmented_labels)

# Save the augmented dataset
save_idx(augmented_images, augmented_labels, 'data/augmented-train-images.idx3-ubyte')