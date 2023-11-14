CC = gcc
SRC = src/
CFLAGS = -pg -O2 -funroll-loops

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
	srun --partition=cpar perf stat -e instructions,cycles ~/code/src/MD.exe < inputdata.txt

profile:
	gprof ~/code/src/MD.exe gmon.out > gprof.txt

