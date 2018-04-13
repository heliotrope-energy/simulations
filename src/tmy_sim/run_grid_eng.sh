#!/bin/bash


qsub -l short -t 1-100 -pe smp 4 -cwd run_eng_job.sh
