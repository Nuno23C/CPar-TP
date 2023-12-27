#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>
#include <cstdlib>
#include <iostream>
#include <sys/time.h>
#include <chrono>
#include <cuda.h>

void checkCUDAError (const char *msg) {
	cudaError_t err = cudaGetLastError();
	if( cudaSuccess != err) {
		std::cerr << "Cuda error: " << msg << ", " << cudaGetErrorString( err) << std::endl;
		exit(-1);
	}
}
