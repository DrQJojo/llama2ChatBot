import argparse
import numpy as np
import matplotlib.pyplot as plt
from args import device

def main(args):
    load = args.load
    model_type = args.model_type
    print(load)
    print(model_type)
    print(device)
    np.save(r'test.npy', np.array([1, 2, 3]))
    plt.figure()
    plt.plot(range(len([1,2,3])),[1,2,3],label='loss',marker='o')
    plt.xlabel('training process')
    plt.ylabel('loss')
    plt.title('loss over training process')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("model_type", type=str, help="Name of the model to train")
    parser.add_argument("--load", action='store_true', help="Load previous weights")
    args = parser.parse_args()
    main(args)
