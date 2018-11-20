from args import parse_args
from experiment import Experiment, setup_logging

def main():
    args = parse_args()
    setup_logging(level=args.log_level, filename=args.log_file)

    model = Experiment(args)
    model.train()



if __name__ == "__main__":
    main()
