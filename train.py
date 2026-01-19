
import numpy as np
import matplotlib.pyplot as plt
from src.layer import Dense
from src.activation import ReLU, Softmax
from src.loss import CategoricalCrossentropy
from src.optimizer import Optimizer_SGD, Optimizer_Adam
from src.network import Network

# Create dummy spiral data
# (Class 0, 1, 2)
def create_data(points, classes):
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y

print("Generating data...")
X, y = create_data(100, 3)

# Initialize Network
model = Network()

print("Building model...")
# Input: 2 features (x, y coords) -> Hidden 1: 64 neurons, ReLU
model.add(Dense(2, 64))
model.add(ReLU())

# Hidden 2: 64 neurons, ReLU
model.add(Dense(64, 64))
model.add(ReLU())

# Output: 3 classes, Softmax
model.add(Dense(64, 3))
model.add(Softmax())

# Loss and Optimizer
# Using CategoricalCrossentropy and Adam optimizer
loss_function = CategoricalCrossentropy()
optimizer = Optimizer_Adam(learning_rate=0.02, decay=5e-5)

model.set(loss=loss_function, optimizer=optimizer)

print("Starting training...")
# Train
model.train(X, y, epochs=1000, batch_size=None, print_every=100)

print("Training complete.")
