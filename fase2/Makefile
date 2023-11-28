CC = gcc
SRC = src/
CFLAGS = -Wall -pg -O2 -lm -mfpmath=sse -funroll-loops -ffast-math -ftree-vectorize -march=x86-64 -fno-omit-frame-pointer -fopenmp

.DEFAULT_GOAL = all

all: MDseq.exe MDpar.exe

MDseq: $(SRC)/MDseq.cpp
	module load gcc/11.2.0;
	$(CC) $(CFLAGS) $(SRC)MDseq.cpp -lm -o MDseq.exe

MDpar: $(SRC)/MDpar.cpp
	module load gcc/11.2.0;
	$(CC) $(CFLAGS) $(SRC)MDpar.cpp -lm -fopenmp -o MDpar.exe

clean:
	rm -f ./MD*.exe
	rm -f ./gprof.txt
	rm -f ./*.out
	rm -f ./cp_average.txt
	rm -f ./cp_output.txt
	rm -f ./cp_traj.xyz
	rm -f ./perfreport

run:
	sbatch run.sh $(file)

test:
	sbatch test.sh

perf_record:
	sbatch perf_record.sh $(file)

perf_stat:
	sbatch perf_stat.sh $(file)

gprof:
	sbatch gprof.sh

compare:
	python compare.py
