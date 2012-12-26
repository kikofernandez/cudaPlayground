// Square Matrix addition

#include <iostream>
#include <math.h>

#define N (20)
#define NBlocks 2
#define NThreads 5 // rows

__global__ void kernel(int *a, int *b, int *c){
	// These ones will compute elements from [0..2047][0..2047]
	int x = threadIdx.x + blockIdx.x * blockDim.x; // [0..19]
	int y = threadIdx.y + blockIdx.y * blockDim.y; // [0..19]
	int round = 0;
	int offset_row = 0;
	int number_total_blocks_per_row = (N+NBlocks*NThreads)/(NBlocks*NThreads); // 2
	int loops = 0;

	// We want to be able to compute as well data between [2047..X]
	while(loops < (N/NThreads)){ // 0 < 4, 1 < 4
		round = 0;
		offset_row = (NThreads*NBlocks*NThreads)*(number_total_blocks_per_row)*round; // 100*0, 100*1
		while(round < number_total_blocks_per_row){ // rows ,,,,0<2
			int offset_col = (NThreads*NBlocks*NThreads)*round; // 50*0,50| 50*0, 50*1
			//c[x*NThreads+y+offset_col+offset_row]; // 4,54| 0+4+0+100,0+4+50+100
			c[x*NThreads+y+offset_col+offset_row] = a[x*NThreads+y+offset_col+offset_row] + b[x*NThreads+y+offset_col+offset_row];
			round++;
		}
		loops += 1; // 1,
	}
}

int main( void ){
	// Variables
	int *a, *b, *c;
	int *d_a, *d_b, *d_c;
	int i;
	int result = 0;
	cudaError_t cuda_ret;

	// Reserve memory CPU
	a = (int * ) malloc(N * N * sizeof(int));
	b = (int * ) malloc(N * N * sizeof(int));
	c = (int * ) malloc(N * N * sizeof(int));

	// Reserve memory GPU
	cuda_ret = cudaMalloc((void**) &d_a, sizeof(int) * N * N);
	if(cuda_ret != cudaSuccess) printf("Unable to allocate device memory");
	cuda_ret = cudaMalloc((void**) &d_b, sizeof(int) * N * N);
	if(cuda_ret != cudaSuccess) printf("Unable to allocate device memory");
	cuda_ret = cudaMalloc((void**) &d_c, sizeof(int) * N * N);
	if(cuda_ret != cudaSuccess) printf("Unable to allocate device memory");

	// Init values for the matrix
	//initMatrixes(&a[0][0], &b[0][0]);
	for(i=0; i<N; i++){
		for(int j=0; j<N; j++){
			a[i*N+j] = i*N+j;
			b[i*N+j] = 0;
		}
	}

	for(i=0; i<N; i++){
		for(int j=0; j<N; j++){
			printf("%d,", a[i+j*N]+b[i+j*N]);
		}
		printf("\n");
	}

	printf("\n\n\n");

	// Copy values to the GPU
	cuda_ret = cudaMemcpy(d_a, a, sizeof(int) * N * N, cudaMemcpyHostToDevice);
	if(cuda_ret != cudaSuccess) printf("Unable to allocate device memory");
	cuda_ret = cudaMemcpy(d_b, b, sizeof(int) * N * N, cudaMemcpyHostToDevice);
	if(cuda_ret != cudaSuccess) printf("Unable to allocate device memory");

	dim3 threads(NThreads, NThreads);
	// Process data in GPU
	kernel<<<NBlocks,threads>>>(d_a, d_b, d_c);
		
	// Data to Mem CPU
	cudaMemcpy(c, d_c, sizeof(int) * N * N, cudaMemcpyDeviceToHost);
	
	int j;
	for(i=0; i < N; i++){
		for(j=0; j<N; j++){
			printf("%d,", c[i*N+j]);
		}
		printf("\n");
	}

	// Free memory
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	return 0;
}
