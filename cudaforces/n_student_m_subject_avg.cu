    #include <cuda_runtime.h>
    #include <stdio.h>
    #include <stdlib.h>

    __global__ void kernel(int *a, int n, int *res){
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx < n){
            atomicAdd(&res[idx/6], a[idx]);
        }
    }

    int main(){
        int nn;
        scanf("%d",&nn);
    
        int n = 6;
        size_t byten = n*nn * sizeof(int);
        
        int *a = (int*) malloc(byten);
        int *da;
        cudaMalloc((void**)&da, byten);
        
        int *res = (int*) malloc(sizeof(int)*nn);
        
        for(int i=0;i<nn;i++) res[i]=0;
        int *dres;
        cudaMalloc((void**)&dres, nn*sizeof(int));
        cudaMemcpy(dres, res, sizeof(int), cudaMemcpyHostToDevice);
        
        for(int i=0;i<nn;i++){
            for(int j=0;j<n;j++){
                scanf("%d",&a[i*n+j]);
            }
            a[i*n+n-1]=0;
        }
        
        cudaMemcpy(da, a, byten, cudaMemcpyHostToDevice);
        
        kernel<<<(n*nn+255)/256, 256>>>(da, n*nn, dres);
        
        cudaMemcpy(res, dres, sizeof(int)*nn, cudaMemcpyDeviceToHost);
        
        for(int i=0;i<nn;i++){
            printf("%d %d\n",res[i],res[i]/5);
        }
        
        cudaFree(da);
        cudaFree(dres);
        free(a);
        free(res);
        
        return 0;
    }