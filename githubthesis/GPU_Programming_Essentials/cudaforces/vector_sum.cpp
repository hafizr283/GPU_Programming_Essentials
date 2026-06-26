
#include <stdio.h>
#include <stdlib.h>

__global__ void kernel(int *a, int *b, int *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    int n;
    scanf("%d", &n);

    size_t bytes = n * sizeof(int);
    int *a, *b,*c;
    a = (int*) malloc(bytes);
    b= (int*) malloc(bytes);
    c = (int*) malloc(bytes);
    int *da, *db, *dc;
    cudaMalloc((void **)&da,bytes);
    cudaMalloc((void **)&db,bytes);
    cudaMalloc((void **)&dc,bytes);

    for(int i=0;i<n;i++)scanf("%d",&a[i]);
    for(int i=0;i<n;i++)scanf("%d",&b[i]);

    cudaMemcpy(da,a,bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(db,b,bytes,cudaMemcpyHostToDevice);

    kernel<<<(n+255)/256,256>>>(da,db,dc,n);
    
    cudaMemcpy(c,dc,bytes,cudaMemcpyDeviceToHost);
    
    for(int i=0;i<n;i++)printf("%d ",c[i]);
    printf("\n");





    return 0;
}