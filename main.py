from args import parse_args
from experiment import Experiment


def main():
    args = parse_args()

    if args.mode == "train":
        model = Experiment(args)
        model.train()

    elif args.mode == "test":
        model = Experiment(args)
        model.test()


if __name__ == "__main__":
    main()
