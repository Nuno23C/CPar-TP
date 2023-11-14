CC = gcc
SRC = src/
CFLAGS = -Wall -pg -O2 -funroll-loops -ffast-math -ftree-vectorize -mavx -mfpmath=sse -march=x86-64

.DEFAULT_GOAL = MD.exe

MD.exe: $(SRC)/MD.cpp
	$(CC) $(CFLAGS) $(SRC)MD.cpp -lm -o ~/code/src/MD.exe

clean:
	rm -f ~/code/src/MD.exe
	rm -f ~/code/gprof.txt
	rm -f ~/code/gmon.out
	rm -f ~/code/cp_average.txt
	rm -f ~/code/cp_output.txt
	rm -f ~/code/cp_traj.xyz

run:
	srun --partition=cpar perf stat -e cache-misses,instructions,cycles ~/code/src/MD.exe < inputdata.txt

run_avg:
	srun --partition=cpar perf stat -r 10 -e cache-references,cache-misses,instructions,cycles ~/code/src/MD.exe < inputdata.txt

compare:
	python compare.py

profile:
	gprof ~/code/src/MD.exe gmon.out > gprof.txt

