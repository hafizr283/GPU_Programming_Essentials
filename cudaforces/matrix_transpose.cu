__global__ void kernel(int* a,int n,int m,int* res){
    int row = threadIdx.x + blockIdx.x*blockDim.x;
    int col = threadIdx.y + blockIdx.y*blockDim.y;
    if(row<n && col<m){
        res[col*n+row]=a[row*m+col];
    }

}
int main(){
    int n,m;scanf("%d %d",&n,&m);
    int *a = (int*) malloc(sizeof(int)*n*m);
    for(int i=0;i<n;i++){
        for(int j=0;j<m;j++){
            scanf("%d",&a[i*m+j]);
        }
    }
    int *da,*res;
    cudaMalloc((void**)&da,sizeof(int)*n*m);
    cudaMalloc((void**)&res,sizeof(int)*n*m);
    cudaMemcpy(da,a,sizeof(int)*n*m,cudaMemcpyHostToDevice);
    dim3 blockDim(16,16);


    kernel<<<1,blockDim>>>(da,n,m,res);
    cudaDeviceSynchronize();
    cudaMemcpy(a,res,sizeof(int)*n*m,cudaMemcpyDeviceToHost);
    
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            printf("%d ",a[i*n+j]);
        }
        printf("\n");
    }
    cudaFree(da);
    free(a);
    
     

}