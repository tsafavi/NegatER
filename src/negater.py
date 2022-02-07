import argparse
import os

from job.generate_negatives import (
    NegativeGenerationFullJob,
    NegativeGenerationByRelationJob,
)
from job.rank_negatives import (
    ThresholdsRankingJob,
    FullGradientsRankingJob,
    ProxyGradientsRankingJob,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("config_dir", help="Directory of job config file")

    parser.add_argument(
        "--type",
        choices=["thresholds", "full-gradients", "proxy-gradients"],
        default="thresholds",
        help=(
            "Type of NegatER job to run: 'thresholds' for NegatER-$\\theta_r$, "
            "'full-gradients' for NegatER-$\\nabla$ without the proxy, "
            "and 'proxy-gradients' for NegatER-$\\nabla$ with the proxy. "
            "Default: 'thresholds'"
        ),
    )

    parser.add_argument(
        "--config-file",
        default="config.yaml",
        help=(
            "Configuration filename in the specified config directory. "
            "Default: 'config.yaml'"
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config_path = os.path.join(args.config_dir, args.config_file)
    print(f"Configuration filepath: {config_path}")

    negater_type = args.type

    if negater_type == "thresholds":
        generation_job = NegativeGenerationByRelationJob(config_path)
        datasets = generation_job.run()

        ranking_job = ThresholdsRankingJob(config_path, datasets)
    else:
        generation_job = NegativeGenerationFullJob(config_path)
        dataset = generation_job.run()

        if negater_type == "full-gradients":
            ranking_job = FullGradientsRankingJob(config_path, dataset)
        else:
            ranking_job = ProxyGradientsRankingJob(config_path, dataset)

    ranking_job.run()
