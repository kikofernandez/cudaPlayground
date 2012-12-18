#include <stdio.h>

#define N (500*1024)

void add(int *a, int *b, int *c){
	int j;
	for(j=0; j<N; j++){
		c[j] = a[j] + b[j];
	}
}

int main( void ){
  int a[N], b[N], c[N];
  
  // fill the array a and b on the CPU
  int i;
  for(i=0; i<N; i++){
    a[i] = i;
    b[i] = i*i;	
  }
  
  add(a, b, c);
  
  // Verification
  int success = 1;
  for(i=0; i<N; i++){
    if((a[i] + b[i]) != c[i]){
      printf("Error: %d + %d = %d\n", a[i], b[i], c[i]);
      success = 0;
    }	
  }
  
  if(success) printf("We did it!!!");
  
  return 0;
}