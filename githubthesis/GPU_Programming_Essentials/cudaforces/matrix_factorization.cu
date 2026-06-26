
#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

using namespace std;

// Custom CUDA Kernel to compute RHS dynamically using CSR format
__global__ void compute_RHS_kernel(int u, int* offsets, int* indices, float* ratings, float* d_FeatureMat, float* d_RHS, int K) {
    int idx = threadIdx.x;
    int startindex = offsets[u];
    int endindex = offsets[u+1];
    float rhs=0;
    for(int i=startindex;i<endindex;i++){
      int relatedindex = indices[i];
      float rating = ratings[i];
      rhs+=d_FeatureMat[relatedindex*K+idx]*rating;

    }
    d_RHS[idx] = rhs;
}

int main() {
    int num_users = 4;
    int num_items = 5;
    int K = 3;
    float lambda = 0.1f;
    int max_iters = 3;
    //row csr
    float h_user_ratings[9] = {5.0f, 3.0f, 4.0f, 5.0f, 1.0f, 2.0f, 4.0f, 4.5f, 3.5f};
    int h_item_indices[9] = {0, 2, 1, 3, 4, 0, 4, 2, 3};
    int h_user_offsets[5] = {0, 2, 5, 7, 9};
    //col csr
    float h_item_ratings[9] = {5.0f, 2.0f, 4.0f, 3.0f, 4.5f, 5.0f, 3.5f, 1.0f, 4.0f};
    int h_user_indices[9] = {0, 2, 1, 0, 3, 1, 3, 1, 2};
    int h_item_offsets[6] = {0, 2, 3, 5, 7, 9};

    float h_X[12] = { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f };
    float h_Y[15] = { 1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 1.0f, 2.0f, 1.0f, 2.0f, 1.0f };
    float h_I[9]  = { 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f };


    cublasHandle_t cublasH;
    cusolverDnHandle_t cusolverH;
    cublasCreate(&cublasH);
    cusolverDnCreate(&cusolverH);

 
    float *d_X, *d_Y, *d_LHS, *d_I, *d_RHS;
    int *d_info;
    cudaMalloc((void**)&d_X, sizeof(float) * num_users * K);
    cudaMalloc((void**)&d_Y, sizeof(float) * num_items * K);
    cudaMalloc((void**)&d_LHS, sizeof(float) * K * K);
    cudaMalloc((void**)&d_I, sizeof(float) * K * K);
    cudaMalloc((void**)&d_RHS, sizeof(float) * K);
    cudaMalloc((void**)&d_info, sizeof(int));

    int *d_user_offsets, *d_item_indices, *d_item_offsets, *d_user_indices;
    float *d_user_ratings, *d_item_ratings;
    cudaMalloc((void**)&d_user_offsets, sizeof(int) * 5);
    cudaMalloc((void**)&d_item_indices, sizeof(int) * 9);
    cudaMalloc((void**)&d_user_ratings, sizeof(float) * 9);
    cudaMalloc((void**)&d_item_offsets, sizeof(int) * 6);
    cudaMalloc((void**)&d_user_indices, sizeof(int) * 9);
    cudaMalloc((void**)&d_item_ratings, sizeof(float) * 9);

    cudaMemcpy(d_X, h_X, sizeof(float) * 12, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, h_Y, sizeof(float) * 15, cudaMemcpyHostToDevice);
    cudaMemcpy(d_I, h_I, sizeof(float) * 9, cudaMemcpyHostToDevice);
    
    cudaMemcpy(d_user_offsets, h_user_offsets, sizeof(int) * 5, cudaMemcpyHostToDevice);
    cudaMemcpy(d_item_indices, h_item_indices, sizeof(int) * 9, cudaMemcpyHostToDevice);
    cudaMemcpy(d_user_ratings, h_user_ratings, sizeof(float) * 9, cudaMemcpyHostToDevice);
    
    cudaMemcpy(d_item_offsets, h_item_offsets, sizeof(int) * 6, cudaMemcpyHostToDevice);
    cudaMemcpy(d_user_indices, h_user_indices, sizeof(int) * 9, cudaMemcpyHostToDevice);
    cudaMemcpy(d_item_ratings, h_item_ratings, sizeof(float) * 9, cudaMemcpyHostToDevice);

    float alpha = 1.0f, beta = 0.0f;
    int lworksize = 0;
    float *d_work = nullptr;

    cout << "Starting ALS Training with CSR Data..." << endl;


    for (int iter = 0; iter < max_iters; iter++) {
        

        // 1. Compute LHS: Y^T * Y
       cublasSsyrk(cublasH, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
            K, num_items,
            &alpha, d_Y, K,
            &beta, d_LHS, K);

        // 2. LHS + lambda * I
        for(int i = 0; i < K; i++) {
            cublasSaxpy(cublasH, 1, &lambda, d_I + (i * K + i), 1, d_LHS + (i * K + i), 1);
        }

        // 3. Cholesky Factorization (L=d_LHS find as d_LHS)
        cusolverDnSpotrf_bufferSize(cusolverH, CUBLAS_FILL_MODE_LOWER, K, d_LHS, K, &lworksize);
        if (d_work == nullptr) cudaMalloc((void**)&d_work, lworksize * sizeof(float));
        cusolverDnSpotrf(cusolverH, CUBLAS_FILL_MODE_LOWER, K, d_LHS, K, d_work, lworksize, d_info);

       
        for (int u = 0; u < num_users; u++) {
           
            // finding d_RHS
            compute_RHS_kernel<<<1, K>>>(u, d_user_offsets, d_item_indices, d_user_ratings, d_Y, d_RHS, K);
            cudaDeviceSynchronize(); 
            
            // Solve (L * L^T) * X_u = RHS with forward substitution and backword substitution 
            // x_u = d_RHS
            cusolverDnSpotrs(cusolverH, CUBLAS_FILL_MODE_LOWER, K, 1, d_LHS, K, d_RHS, K, d_info);
            cudaMemcpy(d_X + (u * K), d_RHS, sizeof(float) * K, cudaMemcpyDeviceToDevice);
        }

        




        //X^t.X
        cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, K, K, num_users, &alpha, d_X, K, d_X, K, &beta, d_LHS, K);

        // 2. Add Regularization: LHS + lambda * I
        for(int i = 0; i < K; i++) {
            cublasSaxpy(cublasH, 1, &lambda, d_I + (i * K + i), 1, d_LHS + (i * K + i), 1);
        }

        // 3. Cholesky Factorization finding L
        cusolverDnSpotrf(cusolverH, CUBLAS_FILL_MODE_LOWER, K, d_LHS, K, d_work, lworksize, d_info);

        
        for (int i = 0; i < num_items; i++) {
            
            compute_RHS_kernel<<<1, K>>>(i, d_item_offsets, d_user_indices, d_item_ratings, d_X, d_RHS, K);
            cudaDeviceSynchronize();
            
            // Solve (L * L^T) * Y_i = RHS
            cusolverDnSpotrs(cusolverH, CUBLAS_FILL_MODE_LOWER, K, 1, d_LHS, K, d_RHS, K, d_info);
            cudaMemcpy(d_Y + (i * K), d_RHS, sizeof(float) * K, cudaMemcpyDeviceToDevice);
        }
    }

    cout << "\nTraining Complete!" << endl;


    cudaMemcpy(h_X, d_X, sizeof(float) * num_users * K, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Y, d_Y, sizeof(float) * num_items * K, cudaMemcpyDeviceToHost);

    
    cout << "Final Factorized User Matrix (X)" << endl;
    for (int u = 0; u < num_users; u++) {
        printf("User %d:  ", u);
        for (int k = 0; k < K; k++) {
            printf("%8.4f ", h_X[u * K + k]);
        }
        printf("\n");
    }


    cout << "Final Factorized Item Matrix (Y)" << endl;
    for (int i = 0; i < num_items; i++) {
        printf("Item %d:  ", i);
        for (int k = 0; k < K; k++) {
            printf("%8.4f ", h_Y[i * K + k]);
        }
        printf("\n");
    }

   
    cudaFree(d_X); cudaFree(d_Y); cudaFree(d_LHS); cudaFree(d_I); cudaFree(d_RHS);
    cudaFree(d_user_offsets); cudaFree(d_item_indices); cudaFree(d_user_ratings);
    cudaFree(d_item_offsets); cudaFree(d_user_indices); cudaFree(d_item_ratings);
    cudaFree(d_info); cudaFree(d_work);
    cublasDestroy(cublasH);
    cusolverDnDestroy(cusolverH);

    return 0;
}
