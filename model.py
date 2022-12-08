import numpy as np
from utilities import sigmoid

class model():
    def __init__(self, layers: list):
        self.num_layers = len(layers)
        self.desc = "Nerual network model"
        if self.num_layers == 1:
            self.desc = "perceptron"
        self.layers = layers
        self.input_layer = layers[0]
        self.output_layer = layers[:-1]

        self.hidden_layers = layers[1:len(layers)-2]

        self.weights = np.array([np.random.randn(layers[i], layers[i+1]) for i in range(self.num_layers - 1)])

    def feedForward(self, train):
        cp_train = train.copy()
        for i in range(1,len(self.layers)-1):
            t = np.dot(cp_train, self.weights[i-1])
            print(len(cp_train), " ", self.weights[i-1].shape)
            print(t.shape)
            for j in t:
                for p in range(len(j)):
                    j[p] = sigmoid(j[p])
            cp_train = t
        return t

mdl = model([2,3,4,1])
w = mdl.weights

print(mdl.feedForward(train=np.array([1,2])))