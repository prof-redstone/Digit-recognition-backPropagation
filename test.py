import train

def test():
    _, _, _, a2 = process_forward(test_images.reshape(-1, 784))
    accuracy = np.mean(np.argmax(a2, axis=1) == np.argmax(test_labels, axis=1))
    print(f'Test Accuracy: {accuracy}')

test()