from args import parse_args
from experiment import Experiment
from monolab.networks.utils import setup_logging


def main():
    args = parse_args()
    setup_logging(level=args.log_level, filename=args.log_file)
    if args.mode == "train":
        model = Experiment(args)
        model.train()

    elif args.mode == "test":
        model = Experiment(args)
        model.test()


if __name__ == "__main__":
    main()
