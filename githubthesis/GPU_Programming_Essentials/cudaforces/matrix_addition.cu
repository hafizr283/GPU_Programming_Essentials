#include<iostream>
using namespace std;
__global__ void kernel(int *a,int *b, int *res,int n){
    int idx = blockDim.x*blockIdx.x+threadIdx.x;
    if(idx<n*n) res[idx]=a[idx]+b[idx];
}
int main(){
    int *a,*b,*da,*db,*dres;
    int n; scanf("%d",&n);
    size_t bytes = sizeof(int)*n*n;
    a = (int*) malloc(bytes);
    b = (int*) malloc(bytes);
    cudaMalloc((void**)&da,bytes);
    cudaMalloc((void**)&db,bytes);
    cudaMalloc((void**)&dres,bytes);
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            scanf("%d",&a[i*n+j]);
        }
    }
     for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            scanf("%d",&b[i*n+j]);
        }
    }
    cudaMemcpy(da,a,bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(db,b,bytes,cudaMemcpyHostToDevice);
    
    kernel<<<(n*n+255)/256,256>>>(da,db,dres,n);
    cudaMemcpy(a,dres,bytes,cudaMemcpyDeviceToHost);

    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            printf("%d ",a[i*n+j]);
        }
        printf("\n");
    }
    cudaFree(da);
    cudaFree(db);
    cudaFree(dres);
    free(a);
    free(b);

    
}