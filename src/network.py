
import numpy as np

class InputLayer:
    def forward(self, inputs):
        self.output = inputs

class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.optimizer = None
        self.input_layer = InputLayer()

    def add(self, layer):
        self.layers.append(layer)

    def set(self, *, loss, optimizer):
        self.loss = loss
        self.optimizer = optimizer

    def forward(self, X):
        self.input_layer.forward(X)
        output = self.input_layer.output
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, output, y):
        # If we are using the combined activation/loss
        if hasattr(self.loss, 'backward'):
             # Standard backward
             self.loss.backward(output, y)
             dinputs = self.loss.dinputs
        
        for layer in reversed(self.layers):
            dinputs = layer.backward(dinputs)

    def train(self, X, y, *, epochs=1, batch_size=None, print_every=100, validation_data=None):
        self.best_loss = 9999999
        
        # Determine steps
        if batch_size is None:
            batch_size = len(X)
            
        train_steps = len(X) // batch_size
        if train_steps * batch_size < len(X):
            train_steps += 1

        for epoch in range(1, epochs+1):
            print(f'Epoch {epoch}/{epochs}')
            
            for step in range(train_steps):
                if batch_size is None:
                    batch_X = X
                    batch_y = y
                else:
                    batch_X = X[step*batch_size:(step+1)*batch_size]
                    batch_y = y[step*batch_size:(step+1)*batch_size]

                # Forward pass
                output = self.forward(batch_X)

                # Calculate loss
                if hasattr(self.loss, 'activation'): 
                    data_loss = self.loss.calculate(output, batch_y)
                else:
                    data_loss = self.loss.calculate(output, batch_y)

                # Predictions
                predictions = np.argmax(output, axis=1)
                if len(batch_y.shape) == 2:
                    y_labels = np.argmax(batch_y, axis=1)
                else:
                    y_labels = batch_y
                accuracy = np.mean(predictions == y_labels)

                # Backward pass
                self.loss.backward(output, batch_y)
                dinputs = self.loss.dinputs
                
                for layer in reversed(self.layers):
                    dinputs = layer.backward(dinputs)
                
                # Optimize
                for layer in self.layers:
                    if hasattr(layer, 'weights'):
                        self.optimizer.pre_update_params()
                        self.optimizer.update_params(layer)
                        self.optimizer.post_update_params()

                if not step % print_every or step == train_steps - 1:
                    print(f'Step: {step}, ' +
                          f'Acc: {accuracy:.3f}, ' +
                          f'Loss: {data_loss:.3f}, ' +
                          f'LR: {self.optimizer.current_learning_rate}')
