#include <stdio.h>
#include <stdlib.h>

__global__ void sum(int *a, int *res){
    __shared__ int data[128];
    int tid = threadIdx.x;
    
    if(tid < 100)
        data[tid] = a[tid]; 
    else 
        data[tid] = 0;
        
    __syncthreads();
    
    for(int stride = 64; stride > 0; stride /= 2){
        if(tid < stride) 
            data[tid] += data[tid + stride]; 
        __syncthreads();
    }

    if(tid == 0) {
        *res = data[0];
    }
}

int main(){
    int *a, *da, *d_res, res_h; 
    
    a = (int*) malloc(sizeof(int) * 100);
    
    for(int i = 0; i < 100; i++){
        a[i] = 1;
    }
    
    cudaMalloc((void**)&da, sizeof(int) * 100);
    cudaMalloc((void**)&d_res, sizeof(int)); 
    
    cudaMemcpy(da, a, sizeof(int) * 100, cudaMemcpyHostToDevice);
    
    sum<<<1, 128>>>(da, d_res);
    
    cudaDeviceSynchronize();
    
    cudaMemcpy(&res_h, d_res, sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("%d\n", res_h);

    free(a);
    cudaFree(da);
    cudaFree(d_res);

    return 0;
}