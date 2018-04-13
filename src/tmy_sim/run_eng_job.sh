#!/bin/bash

data=../../data/renamed/tmy
albedo=0.3

python energy_stats.py --tmy_file $data.$SGE_TASK_ID --albedo $albedo --output ../../processed/TMY-pwr-$albedo-$SGE_TASK_ID.pkl --n_steps max --procs 3
