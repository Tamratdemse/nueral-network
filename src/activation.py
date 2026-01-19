
import numpy as np

class Activation:
    def forward(self, inputs):
        pass
    
    def backward(self, grad_output):
        pass

class ReLU(Activation):
    def forward(self, inputs):
        self.inputs = inputs
        # ReLU: max(0, x)
        self.output = np.maximum(0, inputs)
        return self.output
    
    def backward(self, grad_output):
        # Derivative is 1 where input > 0, else 0
        self.dinputs = grad_output.copy()
        self.dinputs[self.inputs <= 0] = 0
        return self.dinputs

class Softmax(Activation):
    def forward(self, inputs):
        self.inputs = inputs
        # Subtract max for numerical stability (prevents overflow)
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        return self.output
    
    def backward(self, grad_output):
        # Softmax gradient is complex when combined with CrossEntropy
        # But if we use it separately:
        # Jacobian matrix of softmax is diagonal(s) - outer_product(s, s)
        # We will handle the combined Softmax+Loss efficient gradient in the Loss class usually
        # But for general purpose:
        self.dinputs = np.empty_like(grad_output)
        for index, (single_output, single_grad) in enumerate(zip(self.output, grad_output)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_grad)
        return self.dinputs

class Sigmoid(Activation):
    def forward(self, inputs):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))
        return self.output
        
    def backward(self, grad_output):
        # f'(x) = f(x) * (1 - f(x))
        self.dinputs = grad_output * (self.output * (1 - self.output))
        return self.dinputs

class Tanh(Activation):
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.tanh(inputs)
        return self.output
        
    def backward(self, grad_output):
        # f'(x) = 1 - f(x)^2
        self.dinputs = grad_output * (1 - self.output ** 2)
        return self.dinputs
