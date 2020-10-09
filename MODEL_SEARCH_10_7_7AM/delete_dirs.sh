#!/bin/bash

START=601
END=648
for i in $(seq $START $END)
do
  echo $i
  rm -rf models/$i
done
