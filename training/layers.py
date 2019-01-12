import numpy as np
import sys

from util.utils import *

class Layer:
    def __init__(self, name):
        if name == "":
            name = random_str(16)
        self.name = name
        self.batch_size = None
    
    def forward(self, x):
        pass
    
    def backward(self, dout):
        pass
    
    def optimize(self, lr):
        pass

    def calc_output_shape(self):
        pass

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
    
    def print(self):
        pass
    
    def canSwitchTrain(self):
        return False


class Convolution(Layer):

    def __init__(self, input_shape, filter_num, filter_size=4, stride=1, pad=0, weight_init_std=0.01, name=""):
        super().__init__(name)
        self.W = weight_init_std * np.random.randn(filter_num, input_shape[0], filter_size, filter_size)
        self.b = np.zeros(filter_num)
        self.stride = stride
        self.pad = pad

        self.x = None
        self.col = None
        self.col_W = None

        self.dW = None
        self.db = None

        self.input_shape = input_shape

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape

        out_h = int(1 + (H + 2*self.pad - FH) / self.stride)
        out_w = int(1 + (W + 2*self.pad - FW) / self.stride)

        self.col = im2col(x, FH, FW, self.stride, self.pad)
        self.col_W = self.W.reshape(FN, -1).T
        out = np.dot(self.col, self.col_W) + self.b

        self.x = x

        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        return out
    
    def backward(self, dout):
        N, _, H, W = self.x.shape
        FN, C, FH, FW = self.W.shape
        FN, out_h, out_w = self.calc_output_shape()
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dcol = dcol.reshape(N, C, FH, FW, out_h, out_w)
        dx = col2im(dcol, H, W, stride=self.stride, pad=self.pad)

        return dx
    
    def optimize(self, lr):
        self.b -= self.db * lr
        self.W -= self.dW * lr
    
    def calc_output_shape(self):
        FN, _, FH, FW = self.W.shape
        C, H, W = self.input_shape
        out_h, out_w = calc_conv_size(H, W, FH, FW, self.stride, self.pad)
        
        return (FN, out_h, out_w)
    
    def print(self):
        message = self.name + "=> output_shape:" +  self.calc_output_shape()
        print(message)
    
class TransposedConvolution(Layer):

    def __init__(self, input_shape, filter_num, filter_size=4, stride=1, pad=0, weight_init_std=0.01, name=""):
        super().__init__(name)
        self.W = weight_init_std * np.random.randn(input_shape[0], filter_num, filter_size, filter_size)
        self.b = np.zeros(filter_num)
        self.stride = stride
        self.pad = pad

        self.x = None

        self.dW = None
        self.db = None

        self.input_shape = input_shape
    
    def forward(self, x):
        self.x = x
        gcol = np.tensordot(self.W, x, (0, 1)).astype(x.dtype, copy=False)
        gcol = np.rollaxis(gcol, 3)
        _, OH, OW = self.calc_output_shape()
        y = col2im(gcol, OH, OW, stride=self.stride, pad=self.pad)
        y += self.b.reshape((1, self.b.size, 1, 1))
        return y
    
    def backward(self, dout):
        N, _, _, _= self.x.shape
        _, _, FH, FW = self.W.shape
        self.db = np.sum(dout, axis=(0, 2, 3))

        col = im2col(dout, FH, FW, stride=self.stride, pad=self.pad)
        x_col = self.x.reshape(N, -1)
        self.dW = np.dot(x_col, col).reshape(self.W.shape)

        dy = self._backward(dout, self.W)
        return dy
    
    def _backward(self, x, W):
        FN, C, FH, FW = W.shape
        N, C, h, w = x.shape

        out_h = int(1 + (h + 2*self.pad - FH) / self.stride)
        out_w = int(1 + (w + 2*self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = W.reshape(FN, -1).T
        out = np.dot(col, col_W)

        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        return out
    
    def optimize(self, lr):
        self.W -= self.dW * lr
        self.b -= self.db * lr
    
    def calc_output_shape(self):
        C, FN, FH, FW = self.W.shape
        _, H, W = self.input_shape
        out_h, out_w = calc_transposed_conv_size(H, W, FH, FW, self.stride, self.pad)

        return (FN, out_h, out_w)
    
    def print(self):
        message = self.name + "=> output_shape:" +  self.calc_output_shape()
        print(message)
        

    
class LeakyRelu(Layer):

    def __init__(self, alpha, name=""):
        super().__init__(name)
        self.alpha = alpha
        self.x = None
    
    def forward(self, x):
        x[x >= 0] = x[x >= 0]
        x[x < 0] = x[x < 0] * self.alpha
        self.x = x
        return x

    def backward(self, dout):
        dout = np.zeros_like(self.x)
        dout[self.x >= 0] = 1
        dout[self.x < 0] = self.alpha
        return dout
    
    def optimize(self, lr):
        pass
    
    def calc_output_shape(self):
        return "return input_shape"
    
    def print(self):
        message = self.name + "=> output_shape:" +  self.calc_output_shape()
        print(message)
    
class Relu(Layer):

    def __init__(self, name=""):
        super().__init__(name)
        self.x = None
    
    def forward(self, x):
        self.x = x
        x[x < 0] = 0
        return x
    
    def backward(self, dout):
        dout = np.zeros_like(self.x)
        dout[self.x >= 0] = 1
        dout[self.x < 0] = 0
        return dout

class BatchNormalization(Layer):

    def __init__(self, gamma=0.99, beta=0.1, momentum=0.99, eplison=10e-5, name=""):
        super().__init__(name)
        self.gamma = gamma
        self.beta = beta
        self.db = None
        self.dg = None

        self.eplison = eplison
        self.momentum = momentum

        self.inv_std = None
        self.x_hat = None
        
        self.x = None

        self.input_shape = None

        self.running_mean = None
        self.running_var = None

        self.m = None

    
    def forward(self, x, isTrain=True):
        self.input_shape = x.shape
        if x.ndim != 2:
            self.input_shape = x.shape
            x = x.reshape(x.shape[0], -1)

        if isTrain:
            mean = x.mean(axis=0)
            variance = x.var(axis=0)

            self.inv_std = np.reciprocal(np.sqrt(variance + self.eplison), dtype=x.dtype)

            self.x_hat = x - mean
            self.x_hat *= self.inv_std


            y = self.gamma * self.x_hat
            y += self.beta

            if self.running_mean == None:
                self.running_mean = mean
            if self.running_var == None:
                self.running_var = variance

            if self.m == None:
                self.m = np.prod(x.shape) // np.prod(mean.shape)
        
            self.running_mean = self.running_mean * (1 - self.momentum) + mean * self.momentum
            self.running_var = self.running_var * (1 - self.momentum) + (self.m / max(self.m - 1., 1.)) * variance * self.momentum
        
        else:
            inv_std = np.reciprocal(np.sqrt(self.running_var + self.eplison), dtype=x.dtype)
            x_hat = x - self.running_mean
            x_hat *= inv_std
            y = self.gamma * self.x_hat + self.beta

        return y.reshape(self.input_shape)
    
    def backward(self, dout):
        if dout.ndim != 2:
            dout = dout.reshape(dout.shape[0], -1)

        self.db = np.sum(dout, axis=0)
        self.dg = (self.x_hat * dout).sum(axis=0)

        x_diff = self.x_hat / self.inv_std
        m_x_hat = np.mean(dout * x_diff, axis=0, keepdims=True)

        c = (dout * self.inv_std) - (x_diff * m_x_hat * (self.inv_std ** 3))

        dx = self.gamma * (c - np.mean(c, axis=0, keepdims=True))

        return dx.reshape(self.input_shape)
    
    def optimize(self, lr):
        self.beta -= self.db * lr
        self.gamma -= self.dg + lr
    
    def calc_output_shape(self):
        return "return input_shape"
    
    def print(self):
        message = self.name + "=> output_shape:" +  self.calc_output_shape()
        print(message)
    
    def canSwitchTrain(self):
        return True

class Dropout(Layer):

    def __init__(self, dropout_ratio=0.50, name=""):
        super().__init__(name)
        self.dropout_ratio = dropout_ratio
        self.mask = None
    
    def forward(self, x, isTrain=True):
        if self.mask is None:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio

        if isTrain:
            return x * self.mask
        
        else:
            return x * (1 - self.dropout_ratio)
        
    def backward(self, dout):
        return dout * self.mask
    
    def calc_output_shape(self):
        return "return input_shape"
    
    def print(self):
        message = self.name + "=> output_shape:" +  self.calc_output_shape()
        print(message)
    
class Tanh(Layer):
    def __init__(self, name=""):
        super().__init__(name)
        self.y = None

    
    def forward(self, x):
        self.y = np.tanh(x)
        return self.y
    
    def backward(self, dout):
        one = np.ones_like(self.y)
        return dout * (one - np.power(self.y, 2))