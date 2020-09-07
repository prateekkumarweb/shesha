#!/bin/bash

out_dir=$1

for f in $1/*.out
do
  a=$(python tabulate_results.py $f)
  echo $(basename "$f"),$a
done