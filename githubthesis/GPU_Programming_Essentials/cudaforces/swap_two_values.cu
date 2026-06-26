    #include <cuda_runtime.h>
    #include <stdio.h>
    #include <stdlib.h>

    __global__ void kernel(int *a, int n, int *res){
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx < n){
            res[idx]=a[n-idx-1];
        }
    }

    int main(){
        int a,b;
        scanf("%d %d",&a,&b); 
        int *da;
        cudaMalloc((void**)&da,sizeof(int)*2);
        int *hb=(int*)malloc(sizeof(int)*2);
        
        hb[0]=a;
        hb[1]=b;
        cudaMemcpy(da,hb,sizeof(int)*2,cudaMemcpyHostToDevice);
        int *dres;
        cudaMalloc((void**)&dres,sizeof(int)*2);
        kernel<<<1,2>>>(da,2,dres);
        cudaMemcpy(hb,dres,sizeof(int)*2,cudaMemcpyDeviceToHost);
        printf("%d %d",hb[0],hb[1]);


        cudaFree(da);
        cudaFree(dres);
     
        
        return 0;
    }