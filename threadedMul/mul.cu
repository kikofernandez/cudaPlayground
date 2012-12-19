#include <iostream>

#define imin(a, b) (a < b ? a : b)

const int N = 33 * 1024;
const int threadsPerBlock = 256;

// blocksPerGrid is smart. We won't use a constant number
// of blocks 'cause it's unnecessary. We will use the right amount,
// which comes from this simple formula, meaning, the max number
// of blocks will be 32. If we have a number smaller than that,
// do not create 32 blocks, create only the right amount of blocks.
const int blocksPerGrid = 
	imin(32, (N+threadsPerBlock-1)/threadsPerBlock);


__global__ void dot(float *a, float *b, float *c){
	__shared__ float cache[threadsPerBlock];
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int cacheIndex = threadIdx.x;
	
	float temp = 0;
	while(tid < N){ // control condition
		temp += a[tid] + b[tid];
		
		// Remember: 
		// blockDim = #threadsPerBlock
		// gridDim = #blocksPerGrid
		tid += blockDim.x * gridDim.x;
	}
	
	// set cache value
	cache[cacheIndex] = temp;
	
	// synchronize threads because we are going to read afterwards
	__syncthreads();
	
	/*
	We are going to use an iterative reduction threaded technique,
	which will leave us with a small array.
	
	@constraint: threadsPerBlock must be a power of 2
	*/
	
	int i = blockDim.x / 2; // take half number of threads
	while(i != 0){
		if(cacheIndex < i){
			cache[cacheIndex] += cache[cacheIndex+1];
		}
		// syncthreads must be executed in all threads.
		// failing to comply with this rule -> program never
		// ends since it'll be waiting for all the threads to
		// execute that line.
		__syncthreads(); // we will read later on (while) the shared variable
		i /= 2;
	}
	
	if(cacheIndex == 0){ // for instance, but it could be any thread.
		// If you think about it, the sum that we do it's perform
		// at the blockIdx level, meaning the resulting array has
		// a size equivalent to the number of blocks of the grid.
		c[blockIdx.x] = cache[0];
	}
}

int main( void ){
	float *a, *b, c, *partial_c;
	float *dev_a, *dev_b, *dev_partial_c;
	
	// allocate memory on the CPU
	a = (float*) malloc(N * sizeof(float));
	b = (float*) malloc(N * sizeof(float));
	partial_c = (float*) malloc(blocksPerGrid * sizeof(float));
	
	// allocate memory on GPU
	cudaMalloc( (void**) &dev_a, 
				N * sizeof(float));
	cudaMalloc( (void**) &dev_b, N * sizeof(float));
	cudaMalloc( (void**) &dev_partial_c, blocksPerGrid * sizeof(float));
	
	// fill in the host memory with data
	for(int i=0; i<N; i++){
		a[i] = i;
		b[i] = i*2;	
	}
	
	// Copy the arrays to the GPU
	cudaMemcpy(dev_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, N * sizeof(float), cudaMemcpyHostToDevice);
	
	// Execute in GPU
	dot<<<blocksPerGrid,threadsPerBlock>>>(dev_a, dev_b, dev_partial_c);
	
	// Copy result from GPU to CPU memory
	cudaMemcpy(partial_c, dev_partial_c, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);
	
	// Finish the final result in CPU,
	// that's why we have the blocksPerGrid * sizeof(float)
	c = 0;
	for(int i=0; i<blocksPerGrid; i++){
		c += partial_c[i];
	}
	
	//#define sum_squares(x) (x * (x+1) * (2*x + 1)/6)
	//printf("Does GPU value %.6g = %.6g\n", c, 2 * sum_squares((float) (N-1) ) );
	printf("The final result is: %.6g\n", c);
	
	// free memory on GPU
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_partial_c);
	
	// free memory on CPU
	free(a);
	free(b);
	free(partial_c);
}