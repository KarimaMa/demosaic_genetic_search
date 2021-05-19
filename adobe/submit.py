#!/usr/bin/env python3
"""Submit a job to Condor."""
import argparse
from clusterlib import SubmitArgs, submit

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("scripts", nargs="+", help="path to the scripts to run.")
    parser.add_argument("--job_size", default=1, type=int, help="number of instances to launch")
    parser.add_argument("--gpus", default=1, type=int, help="number of gpus to launch")
    parser.add_argument("--docker_image", type=int, default=14, choices=[10, 14])  # 10 has older nvidia drivers
    parser.add_argument("--machine")
    parser.add_argument("--name")
    parser.add_argument("--extra_args", nargs="*", default=[])
    args = parser.parse_args()
    args = SubmitArgs.from_cli(args)
    submit(args)
