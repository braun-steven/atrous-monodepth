from monolab.networks.resnet import Resnet50
import numpy as np
import torch
import time


def test(output_stride):
    img = np.random.randn(1, 3, 256, 512)
    img = torch.Tensor(img)

    model = Resnet50(3, output_stride=output_stride)

    for i in range(10):
        model.forward(img)


def time_test(output_stride):
    start_time = time.time()
    test(output_stride=output_stride)
    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    print("Output stride 64")
    time_test(64)

    print("Output stride 16")
    time_test(16)
