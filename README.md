# Genetic search for demosaicking programs

### Running jobs on the Adobe cluster

##### Interactive session (for test and debug)

Start by launching the interactice job:

```shell
condor_submit cluster/interactive.condor
```

Make sure the job is in the "running" state using:

```shell
condor_q $USER
```

Find the name and slot ID for the interactive instance using:

```shell
condor_status -claimed | grep $USER
```

You can now ssh to the interactive node with:

```shell
ssh <node_name> -p <slot*2000>
```

For instance if you `condor_status` returns `slot4@ilcomp39.ilcomp`,
ssh with `ssh ilcomp39 -p 8000`.

Once you are done, you can kill *all* jobs with:

```shell
condor_rm $USER
```

##### Launching a batch job

##### Creating new jobs

Copy `cluster/test_job.sh` and `cluster/test_job.condor` to new files,
customize as needed and schedule with `condor_submit <myjobs.condor>`.
Make sure the `.condor` file points to the correct `.sh` script.

The batch job will start in the current working directory.


##### Building a new Docker image

This should rarely be needed. You need to log into
`ilcomp37` or `ilcomp38` and run

```shell
bash cluster/build_docker_image.sh
```

