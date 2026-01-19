
import numpy as np

class Dense:
    def __init__(self, n_inputs, n_neurons, regularizer_l1=0, regularizer_l2=0):
        # Initialize weights with small random values
        # He initialization (for ReLU) usually preferred, but using standard scaling here:
        # 0.01 * randn
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        
        # Regularization strength (Not fully implemented in Loss yet but good to have)
        self.regularizer_l1 = regularizer_l1
        self.regularizer_l2 = regularizer_l2

    def forward(self, inputs):
        self.inputs = inputs
        # Y = X * W + B
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output
    
    def backward(self, grad_output):
        # dCost/dW = inputs^T * dCost/dOutput
        self.dweights = np.dot(self.inputs.T, grad_output)
        # dCost/dB = sum(dCost/dOutput)
        self.dbiases = np.sum(grad_output, axis=0, keepdims=True)
        
        # Regularization gradients
        if self.regularizer_l2 > 0:
            self.dweights += 2 * self.regularizer_l2 * self.weights
            self.dbiases += 2 * self.regularizer_l2 * self.biases
        if self.regularizer_l1 > 0:
            self.dweights += self.regularizer_l1 * np.sign(self.weights)
            self.dbiases += self.regularizer_l1 * np.sign(self.biases)

        # dCost/dInputs = dCost/dOutput * Weights^T
        self.dinputs = np.dot(grad_output, self.weights.T)
        return self.dinputs
