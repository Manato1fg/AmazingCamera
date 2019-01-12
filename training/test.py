from layers import *
from Network import Network
import numpy as np
from train import build_generator

def main():
    print("initializing x")
    N = 32
    channel = 3
    w = 128
    h = 128
    x = np.random.uniform(0, 1, (N, channel, h, w))
    print("Done!")

    print("initializing Generator")
    G = build_generator()
    print("Done!")

    print(G.predict(x, isTrain=True).shape)



if __name__ == "__main__":
    main()