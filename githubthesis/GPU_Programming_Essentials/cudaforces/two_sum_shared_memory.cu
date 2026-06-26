#include <stdio.h>
#include <stdlib.h>

__global__ void sum(int *a, int *res){
    __shared__ int data[2];
    int tid = threadIdx.x;
    
    data[tid] = a[tid]; 
    
    __syncthreads();
    
    if(tid == 0) {
        *res = data[0] + data[1];
    }
}

int main(){
    int *a, *da, *d_res, res_h; 
    
    a = (int*) malloc(sizeof(int) * 2);
    
    scanf("%d %d", &a[0], &a[1]);
    
    cudaMalloc((void**)&da, sizeof(int) * 2);
    cudaMalloc((void**)&d_res, sizeof(int)); 
    
    cudaMemcpy(da, a, sizeof(int) * 2, cudaMemcpyHostToDevice);
    
    sum<<<1, 2>>>(da, d_res);
    
    cudaDeviceSynchronize();
    
    cudaMemcpy(&res_h, d_res, sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("%d\n", res_h);

    free(a);
    cudaFree(da);
    cudaFree(d_res);

    return 0;
}