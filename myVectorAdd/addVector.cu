// Vector addition

#include <iostream>

#define NBlocks 16
#define NThreads 64
#define N (500*1024)

void initVectors(int *a, int *b){
	int i;
	for(i = 0; i < N; i++){
		a[i] = i;
		b[i] = i;
	}
}

__global__ void kernel(int *a, int *b, int *c){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	// We will do this operation 
	while(tid < N ){
		c[tid] = a[tid] + b[tid];
		tid += blockDim.x * gridDim.x;
	}
}

int main( void ){
	int a[N], b[N], c[N];
	int *d_a, *d_b, *d_c;

	// Reserve memory in GPU
	cudaMalloc((void**) &d_a, N * sizeof(int));
	cudaMalloc((void**) &d_b, N * sizeof(int));
	cudaMalloc((void**) &d_c, N * sizeof(int));

	// Put values in vectors
	initVectors(a, b);

	// Copy values to GPU memory
	cudaMemcpy(d_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

	// Parallel function
	kernel<<<NBlocks, NThreads>>>(d_a, d_b, d_c);

	// Copy value to CPU
	cudaMemcpy(c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);

	int i;
	int result = 0;
	for(i = 0; i < N; i++){
		result += c[i];
	}

	printf("Result: %d\n", result);

	// Free memory
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	return 0;
}