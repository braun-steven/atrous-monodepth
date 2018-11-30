import datetime
import os
import logging

from args import parse_args
from experiment import Experiment, setup_logging
from utils import notify_mail


def main():
    args = parse_args()

    # Generate base path: ".../$(args.output_dir)/run-$(date)-$(tag)"
    base_dir = generate_run_base_dir(args.tag, args.output_dir)
    log_file = os.path.join(base_dir, "log.txt")

    # Setup logging in base_dir/log.txt
    setup_logging(level=args.log_level, filename=log_file)

    # Wrap experiment in try-catch block to send error via email if it occurs
    try:
        # Run experiment
        experiment = Experiment(args, base_dir=base_dir)
        experiment.train()

        # Notify the user via e-mail and send the log file
        if args.notify is not None:
            subject = f"[MONOLAB {args.tag}] Training Finished!"
            message = (
                f"The experiment in {base_dir} has finished training and "
                f"took {experiment.time_str}. Best loss: {experiment.best_val_loss}"
            )

            notify_mail(
                address=args.notify, subject=subject, message=message, filename=log_file
            )
    except Exception as e:
        # Log error message
        logging.warning(str(e))

        # Notify exception
        if args.notify:
            subject = f"[MONOLAB {args.tag}] Training Error!"
            message = (
                f"The experiment in {base_dir} has failed. An error occurred "
                f"during training:\n\n{str(e)}"
            )
            notify_mail(
                address=args.notify, subject=subject, message=message, filename=log_file
            )


def generate_run_base_dir(tag: str, output_dir: str) -> str:
    """
    Generate a base directory for each experiment run.
    Args:
        tag (str): Experiment tag
        output_dir (str): Experiment output directory

    Returns:
        Directory name
    """
    _date_str = datetime.datetime.today().strftime("%y-%m-%d_%Hh:%Mm")
    tagstr = tag if tag == "" else "_" + tag
    base_dir = os.path.join(output_dir, f"run_{_date_str}{tagstr}")
    os.makedirs(base_dir)
    return base_dir


if __name__ == "__main__":
    main()
