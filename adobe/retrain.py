#!/usr/bin/env python3
"""Progressively submit retrains jobs to Condor."""
import os
import argparse
import time
from clusterlib import SubmitArgs, submit
import subprocess

def main(args):
    # Batch the jobs
    jobs = []
    for script in args.scripts:
        print(script)
        for start_idx in range(args.start_idx, args.num_models, args.batch_size):
            end_idx = min(start_idx + args.batch_size, args.num_models)
            count = end_idx - start_idx
            jobs.append((script, start_idx, count))
        pass

    job = None
    while True:
        cmd1 = subprocess.Popen(['condor_q'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        cmd2 = subprocess.Popen(['tail', '-n', '1'], stdin=cmd1.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = cmd2.communicate()
        num_jobs = int(stdout.split()[0])

        if num_jobs + args.batch_size <= args.queue_size:
            print("%d of %d jobs in queue, submitting new batch" % (num_jobs, args.queue_size))
            job = jobs.pop(0)

            while True:
                print("  Attempting to schedule: ", job)

                name = os.path.splitext(os.path.basename(job[0]))[0]
                if "retrain-" in name:
                    name = name.split("retrain")[-1]
                name = "retrain-" + name

                job_args = SubmitArgs(
                    scripts=[job[0]],
                    job_size=job[2],
                    docker_image=args.docker_image,
                    machine=None,
                    name=name + "__%d" % (job[1]),
                    gpus=1,
                    extra_args=[str(job[1])],
                )

                success = False
                try:
                    submit(job_args)
                    print("  Successfully submitted", job)
                    break
                except RuntimeError as e:
                    print("  Failed to submit", job)
                    print("    Sleeping for %d seconds" % args.sleep)
                    time.sleep(args.sleep)
    # python3 cluster/submit.py cluster/check_drive_space.sh --name ssd_test --gpus 1 --docker_image 10 --job_size $BATCH_SIZE
            print("  success")
        else:
            print("%d of %d jobs in queue, waiting for free slots" % (num_jobs, args.queue_size))
            print("  Sleeping for %d seconds" % args.sleep)
            time.sleep(args.sleep)

        if len(jobs) == 0:
            print("All jobs scheduled, returning.")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("scripts", nargs="+", help="path to the scripts to run.")
    parser.add_argument("--batch_size", default=10, type=int, help="")
    parser.add_argument("--queue_size", default=30, type=int, help="condor MAX_JOBS_PER_OWNER")
    # parser.add_argument("--gpus", default=1, type=int, help="number of gpus to launch")
    parser.add_argument("--docker_image", type=int, default=14, choices=[10, 14])  # 10 has older nvidia drivers
    parser.add_argument("--sleep", default=1, type=int, help="sleep time in seconds")
    parser.add_argument("--num_models", default=100, type=int, help="number of models to retrain.")
    parser.add_argument("--start_idx", default=0, type=int, help="index of the first model to retrain (0-based)")
    # parser.add_argument("--extra_args", nargs="*", default=[])
    args = parser.parse_args()
    print(args)
    # args = SubmitArgs.from_cli(args)
    main(args)
