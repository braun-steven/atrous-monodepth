from args import parse_args
from experiment import Experiment, setup_logging
import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    args = parse_args()
    setup_logging(level=args.log, filename=args.log_file)
    if args.mode == "train":
        model = Experiment(args)
        model.train()

    elif args.mode == "test":
        model = Experiment(args)
        model.test()

        disps = np.load(os.path.join(model.output_directory, "disparities.npy"))

        f, ax = plt.subplots(2)
        image = model.loader[0].cpu().detach().numpy()
        ax[0].imshow(image)
        ax[1].imshow(disps[0])
        plt.show()


if __name__ == "__main__":
    main()
