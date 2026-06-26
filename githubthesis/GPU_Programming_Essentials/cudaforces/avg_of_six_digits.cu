#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void kernel(int *a, int n, int *res){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n){
         atomicAdd(&res[0], a[idx]);
    }
}

int main(){
    int n = 6;
    size_t byten = n * sizeof(int);
    
    int *a = (int*) malloc(byten);
    int *da;
    cudaMalloc((void**)&da, byten);
    
    int *res = (int*) malloc(sizeof(int));
    res[0] = 0;
    
    int *dres;
    cudaMalloc((void**)&dres, sizeof(int));
    cudaMemcpy(dres, res, sizeof(int), cudaMemcpyHostToDevice);
    
    for(int i = 0; i < n; i++) scanf("%d", &a[i]);
    
    cudaMemcpy(da, a, byten, cudaMemcpyHostToDevice);
    
    kernel<<<(5+255)/256, 256>>>(da, 5, dres);
    
    cudaMemcpy(res, dres, sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("%d %d\n", res[0], res[0]/5);
    
    cudaFree(da);
    cudaFree(dres);
    free(a);
    free(res);
    
    return 0;
}