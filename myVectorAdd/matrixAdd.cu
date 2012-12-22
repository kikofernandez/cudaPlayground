// Square Matrix addition

#include <iostream>

#define N (70)
#define NBlocks 64
//#define NThreads 32

__global__ void kernel(int *a, int *b, int *c){
	// These ones will compute elements from [0..2047][0..2047]
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	// We want to be able to compute as well data between [2047..X]
	while((x < N) && (y < N)){
		c[x+y*N] = a[x+y*N] + b[x+y*N];
		x += blockDim.x * gridDim.x;
		y += blockDim.x * gridDim.y;
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
			a[i*N+j] = i+j;
			b[i*N+j] = i+j;
		}
	}

	// Copy values to the GPU
	cuda_ret = cudaMemcpy(d_a, a, sizeof(int) * N * N, cudaMemcpyHostToDevice);
	if(cuda_ret != cudaSuccess) printf("Unable to allocate device memory");
	cuda_ret = cudaMemcpy(d_b, b, sizeof(int) * N * N, cudaMemcpyHostToDevice);
	if(cuda_ret != cudaSuccess) printf("Unable to allocate device memory");

	//dim3 threads(32, 32);
	dim3 threads(N, N);
	// Process data in GPU
	kernel<<<1,threads>>>(d_a, d_b, d_c);
		
	// Data to Mem CPU
	cudaMemcpy(c, d_c, sizeof(int) * N * N, cudaMemcpyDeviceToHost);
	
	int j;
	for(i=0; i < N; i++){
		result = 0;
		for(j=0; j<N; j++){
			result += c[i*N+j];
			printf("%d,", c[i*N+j]);
		}
		printf("----> %d", result);
		printf("\n");
	}

//	// Print success
	printf("Result by row: %d", result);

	// Free memory
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	return 0;
}
