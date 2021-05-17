#!/usr/bin/env python3
"""Submit a job to Condor."""
import argparse
import os
import stat
import htcondor

VALID_MACHINES_V10 = [
    "ilcomp4a",
    "ilcomp4c",
    "ilcomp4e",
    "ilcomp6a",
    "ilcomp6e",
    "ilcomp6f",
    "ilcomp6m",
    "ilcomp6r",
    "ilcomp6w",
    "ilcomp6x",
    "ilcomp22",
    "ilcomp23",
    "ilcomp24",
    "ilcomp25",
    "ilcomp39",
    "ilcomp44",
    "ilcomp49",
    "ilcomp64",
]
    # "ilcomp45",

# not v10
    # "ilcomp60",
    # "ilcomp4d",
    # "ilcomp6l",

# not tested
    # "ilcomp21",
    # "ilcomp27",
    # "ilcomp28",
    # "ilcomp29",
    # "ilcomp31",
    # "ilcomp36",
    # "ilcomp41",
    # "ilcomp42",
    # "ilcomp43",
    # "ilcomp46",
    # "ilcomp65",
    # "ilcomp6y",

# not v14
    # "ilcomp26",
    # "ilcomp5k",
    # "ilcomp5l",
    # "ilcomp5z",

#buggy: gather device 3
# Error from slot4@ilcomp57.ilcomp: Error running docker job: linux runtime spec devices: error gathering device information while adding custom device '/dev/nvidia3': lstat /dev/nvidia3: no such file or directory
    # "ilcomp57",
    # "ilcomp6h",


# 024 (12116473.013.000) 05/13 01:11:20 Job reconnection failed
#     Job not found at execution machine
#         Can not reconnect to slot4@ilcomp5y.ilcomp, rescheduling job
    # "ilcomp5y",


    # Failed to open log
    # "ilcomp6n",

VALID_MACHINES_V14 = [
    "ilcomp6b",
    "ilcomp6c",
    "ilcomp6d",
    "ilcomp6g",
    "ilcomp6k",
    "ilcomp6v",
    "ilcomp6t",
    "ilcomp5a",
    "ilcomp5b",
    "ilcomp5c",
    "ilcomp5d",
    "ilcomp5g",
    "ilcomp5i",
    "ilcomp5f",
    "ilcomp5n",
    "ilcomp5p",
    "ilcomp5q",
    "ilcomp5r",
    "ilcomp5v",
    "ilcomp5w",
    "ilcomp5x",
    "ilcomp61",
    "ilcomp63",
    "ilcomp6u",
    "ilcomp72",
    "ilcomp75",
    "ilcomp76",
    "ilcomp78",
]

DOCKER_IMAGE = "docker-arcluster-dev.dr.corp.adobe.com/mgharbi/karima_genetic:v%d"

def main(args):
    os.makedirs("log", exist_ok=True)

    if args.machine:
        reqs = '(TARGET.Machine == "{}.ilcomp")'.format(args.machine)
    else:
        if args.docker_image == 10:
            valid_machines = VALID_MACHINES_V10
        elif args.docker_image == 14:
            valid_machines = VALID_MACHINES_V14
        else:
            raise NotImplementedError()
        reqs = " || ".join(['(TARGET.Machine == "{}.ilcomp")'.format(m) for m in valid_machines])
        # reqs = ""

    env = "HIGHMEM=1"

    if args.gpus == 1:
        pass
    elif args.gpus == 4:
        env += " MULTIGPU=0123"
    else:
        raise NotImplementedError("GPUS")

    script_args = "$(process) " + " ".join(args.extra_args)

    docker_image = DOCKER_IMAGE % args.docker_image

    for script in args.scripts:
        flags = {
            "universe": "docker",
            "docker_image": docker_image,
            "should_transfer_files": "no",
            "transfer_executable": "false",
            "arguments": script_args,
            "environment": env,
            "requirements": reqs

        }

        if script == "":
            print("Launching interactive job")
            name = "interactive"
        else:
            fname = os.path.basename(script)
            name = os.path.splitext(fname)[0]
            flags["executable"] = script

        if args.name is not None:
            name = args.name

        flags["JobBatchName"] = name

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
    parser.add_argument("--job_size", default=1, type=int, help="number of instances to launch")
    parser.add_argument("--gpus", default=1, type=int, help="number of gpus to launch")
    parser.add_argument("--docker_image", type=int, default=14, choices=[10, 14])  # 10 has older nvidia drivers
    parser.add_argument("--machine")
    parser.add_argument("--name")
    parser.add_argument("--extra_args", nargs="*", default=[])
    args = parser.parse_args()
    main(args)
