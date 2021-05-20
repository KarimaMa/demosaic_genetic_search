#!bin/bash

mkdir -p logs
# parallel -j 8 ./cluster/sensei-retrain-xtrans-chroma.sh {} {%} '>' logs/xtrans_chroma_{}.log ::: {0..99}
parallel -j 8 ./cluster/sensei-retrain-sr-chroma.sh {} {%} '>' logs/sr_chroma_{}.log ::: {0..99}
