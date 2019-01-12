import numpy as np
import random

def random_str(length):
    s = ""
    key = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"
    for _ in range(length):
        s += key[random.randint(0, length - 1)]
    return s


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h, out_w = calc_conv_size(H, W, filter_h, filter_w, stride, pad)

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col


def col2im(col, h, w, stride=1, pad=0):
    N, C, FH, FW, out_h, out_w  = col.shape
    H, W = h, w
    col = col.reshape(N, out_h, out_w, C, FH, FW).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(FH):
        y_max = y + stride*out_h
        for x in range(FW):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]

def calc_conv_size(h, w, fh, fw, stride, pad):
    out_h = int(1 + (h + 2 * pad - fh) / stride)
    out_w = int(1 + (w + 2 * pad - fw) / stride)
    return out_h, out_w

def calc_transposed_conv_size(h, w, fh, fw, stride, pad):
    out_h = stride * (h - 1) + fh - 2 * pad
    out_w = stride * (w - 1) + fw - 2 * pad
    return out_h, out_w