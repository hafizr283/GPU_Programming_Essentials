    #include <cuda_runtime.h>
    #include <stdio.h>
    #include <stdlib.h>

    __global__ void kernel(int *a, int *b, int *res){
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
       if(idx==b[0]-1) res[idx]=a[b[1]-1];
       else if(idx==b[1]-1) res[idx]=a[b[0]-1];
       else res[idx]=a[idx];

    }

    int main(){
       size_t bytes = sizeof(int)*3;
       int *a = (int*) malloc(bytes);
       for(int i=0;i<3;i++) scanf("%d",&a[i]);
       int *b = (int*) malloc(sizeof(int)*2);
       scanf("%d %d",&b[0],&b[1]);
       int *da,*db,*dres;
       cudaMalloc((void**)&da,bytes);
       cudaMalloc((void**)&db, sizeof(int)*2);
       cudaMalloc((void**)&dres,bytes);
       cudaMemcpy(da,a,bytes,cudaMemcpyHostToDevice);
       cudaMemcpy(db,b,sizeof(int)*2,cudaMemcpyHostToDevice);
       kernel<<<1,3>>>(da,db,dres);
       cudaMemcpy(a,dres,bytes,cudaMemcpyDeviceToHost);
       printf("%d %d %d",a[0],a[1],a[2]);
       cudaFree(da);
       cudaFree(db);
       cudaFree(dres);
       free(a);
       free(b);
        
        return 0;
    }