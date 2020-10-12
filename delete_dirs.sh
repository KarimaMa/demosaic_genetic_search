#!/bin/bash

START=$1
END=$2
BASEDIR=$3
for i in $(seq $START $END)
do
  echo $i
  rm -rf $BASEDIR/$i
done
