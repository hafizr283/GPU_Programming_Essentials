
#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
using namespace std;

int main() {
   
    int N = 3; 
    float ha[9] = {
        25.0f, 15.0f, -5.0f,
        15.0f, 18.0f,  0.0f,
        -5.0f,  0.0f, 11.0f
    };
    float hi[9] = {
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f
    };

    cusolverDnHandle_t handle;
    cusolverDnCreate(&handle);

    float *da, *di;
    int *d_info; 

    cudaMalloc((void**)&da, sizeof(float)*N*N);
    cudaMalloc((void**)&di, sizeof(float)*N*N);
    cudaMalloc((void**)&d_info, sizeof(int)); 

    cudaMemcpy(da, ha, sizeof(float)*N*N, cudaMemcpyHostToDevice);
    cudaMemcpy(di, hi, sizeof(float)*N*N, cudaMemcpyHostToDevice);

    int lworksize = 0;
    float *dwork = nullptr;

    cusolverDnSpotrf_bufferSize(handle, CUBLAS_FILL_MODE_LOWER, N, da, N, &lworksize);
    cout<<lworksize<<endl;
    cudaMalloc((void**)&dwork, lworksize * sizeof(float));

   
    cusolverDnSpotrf(handle, CUBLAS_FILL_MODE_LOWER, N, da, N, dwork, lworksize, d_info);
    cudaMemcpy(ha,da,sizeof(float)*N*N,cudaMemcpyDeviceToHost);
        for(int i=0;i<9;i++){
      cout<<ha[i]<<" ";
    }
    cout<<endl;

    cusolverDnSpotrs(handle, CUBLAS_FILL_MODE_LOWER, N, N, da, N, di, N, d_info);

    cudaMemcpy(hi, di, sizeof(float)*N*N, cudaMemcpyDeviceToHost);

    for(int i=0; i<N*N; i++){
        printf("%f ", hi[i]);
        if((i + 1) % N == 0) printf("\n");
    }

    cudaFree(da);
    cudaFree(di);
    cudaFree(dwork);
    cudaFree(d_info); 
    cusolverDnDestroy(handle);

    return 0;
}