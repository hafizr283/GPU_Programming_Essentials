
#include <cuda_runtime.h>
#include <cublas_v2.h>
int main()
{
    float a[12] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f,
        1.0f, 0.0f, 2.0f};
    cublasHandle_t handle;
    cublasCreate(&handle);
    float res[9] = {0.0};
    float *da, *dres;
    cudaMalloc((void **)&da, sizeof(float) * 12);
    cudaMalloc((void **)&dres, sizeof(float) * 9);
    cudaMemcpy(da, a, sizeof(float) * 12, cudaMemcpyHostToDevice);
    cudaMemcpy(dres, res, sizeof(float) * 9, cudaMemcpyHostToDevice);
    float alpha = 1.0f;
    float beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, 3, 3, 4, &alpha, da, 4, da, 4, &beta, dres, 3);
    cudaMemcpy(res, dres, sizeof(float) * 9, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 9; i++)
    {
        printf("%f ", res[i]);
        if ((i + 1) % 3 == 0)
            printf("\n");
    }
    cudaFree(da);
    cudaFree(dres);
    cublasDestroy(handle);

    // y*yT
}