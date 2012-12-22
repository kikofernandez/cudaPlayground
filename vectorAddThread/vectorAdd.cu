#include <iostream>

#define N (500*1024)

__global__ void add(int *a, int *b, int *c){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while(tid < N){
		c[tid] = a[tid] + b[tid];
		tid += blockDim.x + gridDim.x;	
	}	
}

int main( void ){
  int a[N], b[N], c[N];
  int *dev_a, *dev_b, *dev_c;
  
  // allocate memory on the GPU
  cudaMalloc( (void**) &dev_a, N * sizeof(int));
  cudaMalloc( (void**) &dev_b, N * sizeof(int));
  cudaMalloc( (void**) &dev_c, N * sizeof(int));
  
  // fill the array a and b on the CPU
  for(int i=0; i<N; i++){
    a[i] = i;
    b[i] = i;	
  }
  
  // Copy the arrays a and b into GPU
  cudaMemcpy( dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy( dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);
  
  add<<<128,128>>>(dev_a, dev_b, dev_c);
  
  // Cpy the array back from memory
  cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);
  
  // Verification
  bool success = true;
  int result = 0;
  for(int i=0; i<N; i++){
    if((a[i] + b[i]) != c[i]){
      printf("Error: %d + %d = %d\n", a[i], b[i], c[i]);
      success = false;
    }	
    result += c[i];
  }
  
  if(success) printf("We did it!!! %d\n", result);
  
  // Free memory
  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);
  
  return 0;
}