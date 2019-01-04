from monolab.networks.resnet import Resnet50
import numpy as np
import torch


def test(output_stride):
    img = np.random.randn(1, 3, 32, 64)
    img = torch.Tensor(img)

    model = Resnet50(3, output_stride=output_stride)

    model.forward(img)


if __name__ == "__main__":
    import timeit

    print("Output stride = 64")
    print(
        timeit.timeit(stmt="test(output_stride=64)", setup="from __main__ import test"),
        number=10,
    )

    print("Output stride = 16")
    print(
        timeit.timeit(stmt="test(output_stride=16)", setup="from __main__ import test"),
        number=10,
    )
