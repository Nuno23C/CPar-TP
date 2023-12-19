#!/bin/bash
#SBATCH --time=10:00
#SBATCH --cpus-per-task=40
#SBATCH --partition=cpar
perf stat -e cache-misses,instructions,cycles ./$1.exe < inputdata.txt
perf report -n --stdio > perfreport
