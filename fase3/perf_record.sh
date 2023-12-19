#!/bin/bash
#SBATCH --time=10:00
#SBATCH --cpus-per-task=40
#SBATCH --partition=cpar
perf record ./$1.exe < inputdata.txt
perf report -n --stdio > perfreport
