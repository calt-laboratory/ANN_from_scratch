# modules
import numpy as np

class NeuralNetwork():
    '''
    Fully connected neural network with 1 hidden layer (including 2 neurons) and 1 output layer (including 1 neuron).
    For each neuron the sigmoid function is used.
    Adjustable learning_rate and number of epochs.
    Static bias of 1.
    '''

    def __init__(self, x, y, learning_rate, epochs):
        '''
        Constructor for initializing params.
        '''
        self.x = x
        self.y = y.reshape(-1, 1)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.hidden_weights = np.random.normal(size=(3, 2))
        self.outer_weights = np.random.normal(size=(3, 1))
        self.loss_values = [] 
        self.acc = []


    def __repr__(self):
        '''
        Returns a formal representation as string of an instantiated object.
        '''
        return f'NeuralNetwork(x={self.x}, y={self.y}, learning_rate={self.learning_rate}, epochs={self.epochs})'


    def __str__(self):
        '''
        Returns an informal representation as stringf of instantiated object.
        '''
        return f'Artificial Neural Network: shape_of_x: {self.x.shape}, shape_of_y: {self.y.shape}, learning_rate={self.learning_rate}, epochs={self.epochs}'


    @staticmethod
    def sigmoid(x):
        '''
        Static method calculating the sigmoid activation function.
        '''
        return 1 / (1 + np.exp(-x))


    @staticmethod
    def bias(x):
        '''
        Static method adding a bias of 1 to the hidden and output layer.
        '''
        return np.hstack([x, np.ones((x.shape[0], 1))])

    
    def accuracy(self, y_hat):
        '''
        Returns the accuracy of the prediction.
        '''
        acc = sum(self.y.flatten() == y_hat.flatten().round().astype(np.int64))
        return acc/len(self.y)


    def forward(self, wH, wO):
        '''
        Calculates the forward pass of the artificial neural network.
        '''
        bias1 = NeuralNetwork.bias(self.x)
        dot_hidden = np.dot(bias1, wH)
        sigmoid_hidden = NeuralNetwork.sigmoid(dot_hidden)

        bias2 = NeuralNetwork.bias(sigmoid_hidden)
        dot_output = np.dot(bias2, wO)
        sigmoid_out = NeuralNetwork.sigmoid(dot_output)
        
        return sigmoid_hidden, sigmoid_out


    def log_loss(self, y_hat):
        '''
        Calculates the objective function using the log loss.
        '''
        loss = -(self.y * np.log(y_hat) + (1 - self.y) * np.log(1 - y_hat))
        return loss


    def backpropagation(self, hidden_weights, outer_weights, y_hat, hidden_output):
        '''
        Calculates the backward pass (= backpropagationen) for the artificial neural network.
        '''
        error = (y_hat - self.y) * self.log_loss(y_hat)

        sigmoid_derivative = y_hat * (1 - y_hat)
        y_gradient = sigmoid_derivative * error

        hidden_out_with_bias = NeuralNetwork.bias(hidden_output)
        delta_outer_weights = -(np.dot(y_gradient.T, hidden_out_with_bias)) * self.learning_rate
        wO_update = outer_weights + delta_outer_weights.T

        sigmoid_derivative = hidden_output * (1 - hidden_output)
        hidden_gradient = sigmoid_derivative * np.dot(y_gradient, outer_weights[:-1].T)
        
        delta_hidden_weights = -(np.dot(hidden_gradient.T, NeuralNetwork.bias(self.x))) * self.learning_rate
        wH_update = hidden_weights + delta_hidden_weights.T

        return wH_update, wO_update

    
    def train(self):
        '''
        Fitting of the artificial neural network.
        '''
        hidden_weights = self.hidden_weights
        outer_weights = self.outer_weights

        for _ in range(self.epochs):

            hidden_output, outer_output = self.forward(wH=hidden_weights, wO=outer_weights)

            self.loss_values.append(self.log_loss(outer_output).sum())

            self.acc.append(self.accuracy(y_hat = outer_output))

            new_hidden_weights, new_outer_weights = self.backpropagation(hidden_weights, outer_weights, outer_output, hidden_output)

            hidden_weights, outer_weights = new_hidden_weights, new_outer_weights

        return hidden_weights, outer_weights