CC = gcc
SRC = src/
CFLAGS = -Wall -pg -O2 -lm -funroll-loops -ffast-math -ftree-vectorize -mfpmath=sse -march=x86-64

.DEFAULT_GOAL = all

all: MDseq.exe MDpar.exe

MD_seq: $(SRC)/MDseq.cpp
	module load gcc/11.2.0;
	$(CC) $(CFLAGS) $(SRC)MDseq.cpp -lm -o MD_seq.exe

MD_par: $(SRC)/MDpar.cpp
	module load gcc/11.2.0;
	$(CC) $(CFLAGS) $(SRC)MDpar.cpp -lm -fopenmp -o MD_par.exe

clean:
	rm -f ./MD*.exe
	rm -f ./gprof.txt
	rm -f ./gmon.out
	rm -f ./cp_average.txt
	rm -f ./cp_output.txt
	rm -f ./cp_traj.xyz

run_seq:
	srun --partition=cpar --cpus-per-task=8 perf stat -e cache-misses,instructions,cycles ./MD_seq.exe < inputdata.txt

run_avg_seq:
	srun --partition=cpar --cpus-per-task=8 perf stat -r 10 -e cache-references,cache-misses,instructions,cycles ./MD_seq.exe < inputdata.txt

run_par:
	srun --partition=cpar --cpus-per-task=8 perf stat -e cache-misses,instructions,cycles ./MD_par.exe < inputdata.txt

run_avg_par:
	srun --partition=cpar --cpus-per-task=8 perf stat -r 10 -e cache-references,cache-misses,instructions,cycles ./MD_par.exe < inputdata.txt

compare:
	python compare.py
