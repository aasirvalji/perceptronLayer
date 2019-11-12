import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# defining our sigmoid function 


def sigmoid_derivative(x):
    return x * (1 - x)
#sigmoid (dy/dx)

training_inputs = np.array([[0,0,1],
                            [1,1,1],
                            [1,0,1],
                            [0,1,1]])


training_outputs = np.array([[0,1,1,0]]).T
#expected outputs

np.random.seed(1)
#random seeds for calculations

synaptic_weights = 2 * np.random.random((3,1)) - 1
# initialize weights randomly with mean 0 to create weight matrix, synaptic weights

print('Random starting synaptic weights: ')
print(synaptic_weights)

#Repeat process 50,000 times to produce accurate high or low outputs (1/0)
for iteration in range(50000):

    input_layer = training_inputs

    
    outputs = sigmoid(np.dot(input_layer, synaptic_weights))
    # Normalize product of the input layer with the synaptic weights

    
    error = training_outputs - outputs
    #for every iteration, check our margin of error

    adjustments = error * sigmoid_derivative(outputs)


    synaptic_weights += np.dot(input_layer.T, adjustments)
    #update our synaptic weighings

print('Synaptic weights after training: ')
print(synaptic_weights)

print("Output After Training:")
print(outputs)