
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# 3-layer neural network with 1 neuron each
# Input -> Hidden Layer 1 -> Hidden Layer 2 -> Output
# But wait, "3 layer neural network" usually means Input Layer (not counted) -> Hidden 1 -> Hidden 2 -> Output? 
# Or Input -> Hidden -> Output (2 layers of weights, 3 layers of neurons)?
# The prompt says: "simple 3 layer neural network each having only one neuron"
# I will interpret this as 3 layers of weights:
# Input Layer (value) -> [w1] -> L1 -> [w2] -> L2 -> [w3] -> Output

# Setup
np.random.seed(42)

# Weights
# w1 connects input to L1 (1 input -> 1 neuron)
w1 = np.random.uniform(-1, 1)
b1 = np.random.uniform(-1, 1)

# w2 connects L1 to L2
w2 = np.random.uniform(-1, 1)
b2 = np.random.uniform(-1, 1)

# w3 connects L2 to Output
w3 = np.random.uniform(-1, 1)
b3 = np.random.uniform(-1, 1)

learning_rate = 0.1
epochs = 1000

# Training data (simple "1 -> 0" mapping for demo)
X = 1
Y = 0 

print(f"Initial weights: w1={w1:.4f}, w2={w2:.4f}, w3={w3:.4f}")

for i in range(epochs):
    # --- Forward Propagation ---
    # Layer 1
    z1 = w1 * X + b1
    a1 = sigmoid(z1) # activation of layer 1
    
    # Layer 2
    z2 = w2 * a1 + b2
    a2 = sigmoid(z2) # activation of layer 2
    
    # Layer 3 (Output)
    z3 = w3 * a2 + b3
    a3 = sigmoid(z3) # final output
    
    output = a3
    
    # --- Error ---
    loss = 0.5 * (Y - output) ** 2
    
    if i % 100 == 0:
        print(f"Epoch {i}, Output: {output:.4f}, Loss: {loss:.4f}")
        
    # --- Backpropagation ---
    # dLoss/dw3 = dLoss/da3 * da3/dz3 * dz3/dw3
    
    # 1. Output Layer Gradients
    d_loss_a3 = -(Y - output)
    d_a3_z3 = sigmoid_derivative(output) 
    d_z3_w3 = a2
    d_z3_b3 = 1
    
    grad_w3 = d_loss_a3 * d_a3_z3 * d_z3_w3
    grad_b3 = d_loss_a3 * d_a3_z3 * d_z3_b3
    
    # 2. Hidden Layer 2 Gradients
    # dLoss/dw2 = dLoss/da3 * da3/dz3 * dz3/da2 * da2/dz2 * dz2/dw2
    # We already have d_loss_z3 = (d_loss_a3 * d_a3_z3)
    d_loss_z3 = d_loss_a3 * d_a3_z3
    
    d_z3_a2 = w3
    d_a2_z2 = sigmoid_derivative(a2)
    d_z2_w2 = a1
    d_z2_b2 = 1
    
    d_loss_z2 = d_loss_z3 * d_z3_a2 * d_a2_z2
    grad_w2 = d_loss_z2 * d_z2_w2
    grad_b2 = d_loss_z2 * d_z2_b2
    
    # 3. Hidden Layer 1 Gradients
    # Similar chain rule
    d_z2_a1 = w2
    d_a1_z1 = sigmoid_derivative(a1)
    d_z1_w1 = X
    d_z1_b1 = 1
    
    d_loss_z1 = d_loss_z2 * d_z2_a1 * d_a1_z1
    grad_w1 = d_loss_z1 * d_z1_w1
    grad_b1 = d_loss_z1 * d_z1_b1
    
    # --- Update Weights ---
    w1 -= learning_rate * grad_w1
    b1 -= learning_rate * grad_b1
    
    w2 -= learning_rate * grad_w2
    b2 -= learning_rate * grad_b2
    
    w3 -= learning_rate * grad_w3
    b3 -= learning_rate * grad_b3

print(f"\nFinal prediction for input {X}: {output:.4f} (Target: {Y})")
