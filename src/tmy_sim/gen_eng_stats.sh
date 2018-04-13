#!/bin/bash

data=../../data/alltmy3a/690150TYA.CSV

python energy_stats.py --tmy_file $data --albedo 0.3 --output ../../processed/690150TYA.pkl --n_steps max --procs 3
