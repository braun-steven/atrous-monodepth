from args import parse_args
from experiment import Experiment, setup_logging
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np
import os


def main():
    args = parse_args()
    setup_logging(level=args.log_level, filename=args.log_file)
    args.model = "testnet"
    if args.mode == "train":
        model = Experiment(args)
        model.train()

    elif args.mode == "test":
        model = Experiment(args)
        model.test()

        disps = np.load(os.path.join(model.output_dir, "disparities.npy"))

        # setup figure
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)

        # set up list of images for animation
        ims = []
        for i, data in enumerate(iter(model.loader)):
            image = np.transpose(data.cpu().detach().numpy()[0, 0], (1, 2, 0))
            im = ax1.imshow(image)

            im2 = ax2.imshow(disps[i], cmap="plasma")

            ims.append([im, im2])

        # run animation
        ani = anim.ArtistAnimation(fig, ims, interval=500)
        plt.show()


if __name__ == "__main__":
    main()
