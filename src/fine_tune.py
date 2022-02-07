import argparse
import os

from job.classify import FineTuneTrainJob, FineTuneValidJob, FineTuneTestJob


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("config_dir", help="Directory of job config file")

    parser.add_argument(
        "--action",
        nargs="+",
        default=["train", "test"],
        choices=[
            "train",
            "valid",
            "test",
        ],
        help=(
            "Type of classification job to run. "
            "Note that the model is validated automatically after each epoch of "
            "training, so only pass 'valid' as an action if you want to separately "
            "evaluate the model on the validation set. "
            "Default: ['train', 'test']",
        ),
    )

    parser.add_argument(
        "--config-file",
        default="config.yaml",
        help=(
            "Configuration filename in the specified config directory."
            "Default: 'config.yaml'"
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config_path = os.path.join(args.config_dir, args.config_file)

    print(f"Configuration filepath: {config_path}")
    actions = ("train", "valid", "test")
    jobs = (FineTuneTrainJob, FineTuneValidJob, FineTuneTestJob)

    for action, job_type in zip(actions, jobs):
        if action in args.action:
            job = job_type(config_path)
            job.run()
