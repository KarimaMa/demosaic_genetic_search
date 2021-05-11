#!/usr/bin/env python3
"""Submit a job to Condor."""
import argparse
import os
import stat
import htcondor

VALID_MACHINES = [
    "ilcomp21",
    "ilcomp22",
    "ilcomp27",
    "ilcomp28",
    "ilcomp29",
    "ilcomp31",
    "ilcomp36",
    "ilcomp41",
    "ilcomp46",
    "ilcomp6x",
    "ilcomp6y",
    # "ilcomp6u",
]

def main(args):
    # make script exectutable
    # os.chmod(args.script, stat.S_IEXEC)

    os.makedirs("log", exist_ok=True)

    reqs = " || ".join(['(TARGET.Machine == "{}.ilcomp")'.format(m) for m in VALID_MACHINES])

    for script in args.scripts:
        flags = {
            "universe": "docker",
            "docker_image": args.docker_image,
            "should_transfer_files": "no",
            "transfer_executable": "false",
            "arguments": "$(process)",
            "environment": '"HIGHMEM=1 MULTIGPU=0123"',  # We need this to get 64GB of /dev/shm
            "requirements": reqs
        }

        if script == "":
            print("Launching interactive job")
            name = "interactive"
        else:
            fname = os.path.basename(script)
            name = os.path.splitext(fname)[0]
            flags["executable"] = script,

        flags["output"] = os.path.join("log", "{}.$(process).out".format(name))
        flags["error"] = os.path.join("log", "{}.$(process).err".format(name))
        flags["log"] = os.path.join("log", "{}.$(process).log".format(name))

        coll = htcondor.Collector()  # create the object representing the collector
        schedd_ad = coll.locate(htcondor.DaemonTypes.Schedd) # locate the default schedd
        job = htcondor.Submit(flags)
        schedd = htcondor.Schedd(schedd_ad)
        with schedd.transaction() as txn:
            job.queue(txn, count=args.job_size)

        print(job)
        print("Submitted {} jobs".format(args.job_size))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("scripts", nargs="+", help="path to the scripts to run.")
    parser.add_argument("--job_size", default=4, type=int, help="number of instances to launch")
    parser.add_argument("--docker_image", default="docker-arcluster-dev.dr.corp.adobe.com/mgharbi/karima_genetic:v14")
    # parser.add_argument("--gpus", type=int, nargs="*", help="list of GPUs to use", choices=[0, 1, 2, 3])
    # parser.add_argument("--notify", action="store_true", dest="notify", help="if True, sends email notifications")
    # parser.add_argument("--debug", action="store_true", dest="debug", help="if True, prints the submitted command")
    # parser.add_argument("--executable", type=str, default="jobs/demo.py", help="path to your job's executable.")
    # parser.add_argument("--args", type=str, nargs="*", default=[], help="arguments to your executable (by default we pass the job_id and job_size as the first two arguments)")
    # parser.add_argument("--jobname", type=str, help="name of your job, for the log files")
    # parser.add_argument("--docker_image", type=str, default="mgharbi/base:v23", help="name of the docker image to use")
    # parser.add_argument("--machines", nargs="*", type=str, help="name of machines to submit to")
    # parser.add_argument("--exclude", nargs="*", type=str, help="name of machines to exclude")
    # parser.add_argument("--highmem", dest="highmem", action="store_true", help="increase shared memory space on docker (/dev/shm")
    # parser.set_defaults(notify=False, debug=False, highmem=False)
    args = parser.parse_args()
    main(args)
