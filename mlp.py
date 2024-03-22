import numpy as np

#save activations and derivatives
#implement backpropagation
#implement gradient descent
#implement train
#train our net with dummy dataset
#make some predictions

class MLP:
    def __init__(self, num_inputs = 3, num_hidden = [3, 5], num_outputs = 2):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        layers = [self.num_inputs] + self.num_hidden + [self.num_outputs]

        #initiate random weights
        self.weights = []
        for i in range(len(layers)-1): #need two weight matrices
            W = np.random.rand(layers[i], layers[i+1])
            self.weights.append(W)

        activations = [] #list of arrays and each array is activation for a layer
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)

        self.activations = activations

        derivatives = [] #list of matrices and each matrix is derivative of Error with Weight matrix for a layer except first
        for i in range(len(layers)-1): #derivatives are equal to nbr of weight matrices
            d = np.zeros((layers[i], layers[i+1])) # Corrected: pass the shape as a tuple #like weight matrix shape
            derivatives.append(d)
        
        self.derivatives = derivatives



    def forward_propagate(self, inputs):
        #for a neuron, net input and then activation
        activation = inputs
        self.activations[0] = inputs

        for i, W in enumerate(self.weights):
            #calc net input for a given layer
            net_inputs = np.dot(activation, W)
            # calc activation for a given layer
            activation = self._sigmoid(net_inputs)
            self.activations[i+1] = activation
        
        return activation #return activation of a layer
    


    def back_propagate(self, error, verbose = False):
        #dE/dW_i = (y-a[i+1]) s'(h_[i+1]).a_i
        #s'(h_[i+1])=s(h_[i+1])(1-s(h_[i+1]))
        #s(h_[i+1]) = a_[i+1]

        #dE/dW_[i-1] = (y-a_[i+1])s'(h_[i+1])W_i s'(h_i) a_[i-1]

        for i in reversed(range(len(self.derivatives))):
            activation = self.activations[i+1]   #ndarray([0.1,0.2])-->ndarray([[0.1, 0.2]]) list to row vector 
            delta = error * self._sigmoid_derivative(activation)
            delta_reshaped = delta.reshape(delta.shape[0], -1).T
            cur_activation = self.activations[i]  #ndarray([0.1,0.2])-->ndarray([[0.1], [0.2]]) list to col vector
            cur_activation_reshaped = cur_activation.reshape(cur_activation.shape[0], -1)
            self.derivatives[i] = np.dot(cur_activation_reshaped, delta_reshaped)
            error = np.dot(delta, self.weights[i].T)

            if verbose:
                print('Derivatives for W{}: {}'.format(i, self.derivatives[i]))

        return error
    
    def gradient_descent(self, learning_rate): 
        for i in range(len(self.weights)):
            weight = self.weights[i]#for a layer taking its derivative matrix and weight matrix
            # print("Original W{} {}".format(i, weight))
            derivative = self.derivatives[i]
            weight += derivative*learning_rate
            # print("Updated W{} {}".format(i, weight))

    def train(self, inputs, targets, epochs, learning_rate):

        for i in range(epochs):
            sum_error = 0
            for input, target in zip(inputs, targets):
                    #perform forward prop
                    output = mlp.forward_propagate(input)
                    #calc erro
                    error = target - output
                    #perform back prop
                    self.back_propagate(error)
                    #apply gradient descent
                    self.gradient_descent(learning_rate=1)

                    sum_error += self._mse(target, output)
            #report error for each epoch
            print("Error: {} at epoch {}".format(sum_error/len(input), i))

    def _mse(self, target, output):
        return np.average((target-output)**2)


    def _sigmoid_derivative(self, X):
        return X*(1-X)


    def _sigmoid(self, X):
        return 1/(1+np.exp(-X))
    
import random

if __name__=="__main__":
    items = np.array([[random.random()/2 for _ in range(2)] for _ in range(1000)]) #[[0.1, 0.2], [0.3, 0.4]]
    targets = np.array([[i[0]+i[1]] for i in items]) #[[0.3], [0.7]]
    #create MLP
    mlp = MLP(2, [5], 1)
    #train mlp
    mlp.train(items, targets, 50, 0.1)

    #create dummy data
    input = np.array([0.3, 0.1])
    target = np.array([0.4])
    output = mlp.forward_propagate(input)
    print()
    print()
    print("Our network believes that {} + {} is equal to {}".format(input[0], input[1], output[0]))