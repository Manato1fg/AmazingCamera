import numpy
from Network import Network
from layers import Convolution, LeakyRelu, BatchNormalization, TransposedConvolution, Relu, Dropout, Tanh

def build_generator():

    net = Network(name="Generator")

    #input shape: (3, 128, 128)
    net.add(Convolution((3, 128, 128), 64, filter_size=4, stride=2, pad=1, name="Convolution1"))
    net.add(LeakyRelu(0.2, name="LeakyRelu1"))
    net.add(BatchNormalization(name="BatchNorm1"))

    #input shape: (64, 64, 64)
    net.add(Convolution((64, 64, 64), 128, filter_size=4, stride=2, pad=1, name="Convolution2"))
    net.add(LeakyRelu(0.2, name="LeakyRelu2"))
    net.add(BatchNormalization(name="BatchNorm2"))

    #input shape: (128, 32, 32)
    net.add(Convolution((128, 32, 32), 256, filter_size=4, stride=2, pad=1, name="Convolution3"))
    net.add(LeakyRelu(0.2, name="LeakyRelu3"))
    net.add(BatchNormalization(name="BatchNorm3"))

    #input shape: (256, 16, 16)
    net.add(Convolution((256, 16, 16), 512, filter_size=4, stride=2, pad=1, name="Convolution4"))
    net.add(LeakyRelu(0.2, name="LeakyRelu4"))
    net.add(BatchNormalization(name="BatchNorm4"))

    #input shape: (512, 8, 8)
    net.add(Convolution((512, 8, 8), 512, filter_size=4, stride=2, pad=1, name="Convolution5"))
    net.add(LeakyRelu(0.2, name="LeakyRelu5"))
    net.add(BatchNormalization(name="BatchNorm5"))

    #input shape: (512, 4, 4)
    net.add(Convolution((512, 4, 4), 512, filter_size=4, stride=2, pad=1, name="Convolution6"))
    net.add(LeakyRelu(0.2, name="LeakyRelu6"))
    net.add(BatchNormalization(name="BatchNorm6"))

    #input shape: (512, 2, 2)
    net.add(Convolution((512, 2, 2), 512, filter_size=4, stride=2, pad=1, name="Convolution7"))
    net.add(LeakyRelu(0.2, name="LeakyRelu7"))
    net.add(BatchNormalization(name="BatchNorm7"))

    #input shape: (512, 1, 1)
    net.add(Relu(name="Relu1"))
    net.add(TransposedConvolution((512, 1, 1), 512, filter_size=4, stride=2, pad=1, name="TransposedConvolution1"))
    net.add(BatchNormalization(name="BatchNorm8"))
    net.add(Dropout(dropout_ratio=0.5, name="Dropout1"))

    #input shape: (512, 2, 2)
    net.add(Relu(name="Relu2"))
    net.add(TransposedConvolution((512, 2, 2), 512, filter_size=4, stride=2, pad=1, name="TransposedConvolution2"))
    net.add(BatchNormalization(name="BatchNorm9"))
    net.add(Dropout(dropout_ratio=0.5, name="Dropout2"))

    #input shape: (512, 4, 4)
    net.add(Relu(name="Relu3"))
    net.add(TransposedConvolution((512, 4, 4), 512, filter_size=4, stride=2, pad=1, name="TransposedConvolution3"))
    net.add(BatchNormalization(name="BatchNorm10"))
    net.add(Dropout(dropout_ratio=0.5, name="Dropout3"))

    #input shape: (512, 8, 8)
    net.add(Relu(name="Relu4"))
    net.add(TransposedConvolution((512, 8, 8), 256, filter_size=4, stride=2, pad=1, name="TransposedConvolution4"))
    net.add(BatchNormalization(name="BatchNorm11"))

    #input shape: (256, 16, 16)
    net.add(Relu(name="Relu5"))
    net.add(TransposedConvolution((256, 16, 16), 128, filter_size=4, stride=2, pad=1, name="TransposedConvolution5"))
    net.add(BatchNormalization(name="BatchNorm12"))

    #input shape: (128, 32, 32)
    net.add(Relu(name="Relu6"))
    net.add(TransposedConvolution((128, 32, 32), 64, filter_size=4, stride=2, pad=1, name="TransposedConvolution6"))
    net.add(BatchNormalization(name="BatchNorm13"))

    #input shape: (64, 64, 64)
    net.add(Relu(name="Relu7"))
    net.add(TransposedConvolution((64, 64, 64), 3, filter_size=4, stride=2, pad=1, name="TransposedConvolution7"))

    #output shape: (3, 128, 128)

    net.add(Tanh(name="Activation"))

    return net





if __name__ == "__main__":
    build_generator()
