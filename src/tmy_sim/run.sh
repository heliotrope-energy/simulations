#!/bin/bash

data=../../data/renamed/tmy
output=../../plots/all_tmy
SGE_TASK_ID=1
albedo=0.3

VENV=~/solar

#activate VENV
source $VENV/bin/activate

python3 run_job.py --tmy_file $data.$SGE_TASK_ID --albedo=$albedo --steps max --output $output
