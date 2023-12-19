CC = nvcc
SRC = src/
INCLUDES = include/

CFLAGS = -O2 -g -std=c++11 -arch=sm_35 -Wno-deprecated-gpu-targets

THREADS = 4
CP_CLUSTERS = 4

.DEFAULT_GOAL = MDcuda

MDcuda: $(SRC)MDcuda.cu $(SRC)MDcuda.h
	module load gcc/7.2.0;\
	module load cuda/11.3.1;\
	$(CC) $(CFLAGS) $(SRC)MDcuda.cu -lm -o MDcuda.exe

clean:
	rm -f ./MD*.exe

run:
	./MDcuda < inputdata.txt
