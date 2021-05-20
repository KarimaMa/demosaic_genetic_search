#!bin/bash

mkdir -p logs
# parallel -j 8 echo "job" {%} ::: {0..9}
parallel -j 8 ./cluster/sensei-retrain-xtrans-chroma.sh {} {%} '>' logs/xtrans_chroma_{}.log ::: {0..99}
