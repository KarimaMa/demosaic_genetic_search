#!bin/bash

# parallel -j 8 echo "job" {%} ::: {0..9}
parallel -j 8 ./cluster/sensei-retrain-xtrans-chroma.sh {} {%} ::: {0..99}
