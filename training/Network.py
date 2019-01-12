from collections import OrderedDict
import sys, os
sys.path.append(os.pardir)

from layers import Convolution, LeakyRelu, BatchNormalization


# Generator class
# it creates an image from input image.
class Network:
    def __init__(self, name=""):
        self.layers = OrderedDict()
        self.name = name

    # add a layer
    def add(self, layer):
        self.layers[layer.name] = layer

    
    # predict this model.
    # x: ndarray => data
    # batch_size: int => batch size
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
    
    def learn(self, x, t):
        # forward
        self.predict(x)
    
    def set_batch_size(self, batch_size):
        for layer in self.layers:
            layer.set_batch_size(batch_size)
