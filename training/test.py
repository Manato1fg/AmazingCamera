from layers import *
from Network import Network
import numpy as np

def main():
    N = 64
    channel = 3
    w = 4
    h = 4
    x = np.random.uniform(0, 1, (N, channel, h, w))

    print(x.shape)

    """
    net = Network(name="TestNetwork")

    net.add(Convolution((channel, w, h), 64, filter_size=4, stride=1, pad=1, name="Convolution"))
    net.add(BatchNormalization(name="BatchNorm"))
    net.add(LeakyRelu(0.2, name="LeakyRelu"))
    """

    conv = Convolution((channel, h, w), 64, filter_size=4, stride=2, pad=1, name="Convolution")

    y = conv.forward(x)

    print(y.shape)

    bn = BatchNormalization()

    y = bn.forward(y)

    print(y.shape)

    lr = LeakyRelu(0.2)

    y = lr.forward(y)

    print(y.shape)

    tc = TransposedConvolution((64, 2, 2), 3, 4, stride=2, pad=1, name="Transoposed Convolution")

    y = tc.forward(y)

    print(y.shape)

    dy = tc.backward(np.ones_like(y))

    print(dy.shape)

    dy = lr.backward(dy)

    print(dy.shape)

    dy = bn.backward(dy)

    print(dy.shape)

    dy = conv.backward(dy)

    print(dy.shape)






if __name__ == "__main__":
    main()