__global__ void transpose(int* a,int n,int m,int* res){
    int row = threadIdx.x + blockIdx.x*blockDim.x;
    int col = threadIdx.y + blockIdx.y*blockDim.y;
    if(row<n && col<m){
        res[col*n+row]=a[row*m+col];
    }

}
__global__ void multiplication(int *a,int* b,int* c,int n){
     int row = threadIdx.x+blockIdx.x*blockDim.x;
     int col = threadIdx.y+blockIdx.y*blockDim.y;
     int i = threadIdx.z+blockIdx.z*blockDim.z;
     if(row<n && col<n && i<n){
        int sum = 0;
      
        sum+=(a[row*n+i]*b[col*n+i]);
        
        atomicAdd(&c[row*n+col],sum);
        
     }

}
int main(){
   int n;scanf("%d",&n);
   int *a, *b,*da,*db,*dc,*dbt,*c;
   size_t bytes = sizeof(int)*n*n;
   a = (int*) malloc(bytes);
   b = (int*) malloc(bytes);
   c = (int*) malloc(bytes);

   cudaMalloc((void**)&da,bytes);
   cudaMalloc((void**)&db,bytes);
   cudaMalloc((void**)&dc,bytes);
   cudaMalloc((void**)&dbt,bytes);
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
   for(int i=0;i<n;i++){
    for(int j=0;j<n;j++){
        c[i*n+j]=0;
    }
   }
   dim3 block(8,8,8);
   dim3 grid(2,2,2);
   cudaMemcpy(da,a,sizeof(int)*n*n,cudaMemcpyHostToDevice);
   cudaMemcpy(db,b,sizeof(int)*n*n,cudaMemcpyHostToDevice);
   cudaMemcpy(dc,c,bytes,cudaMemcpyHostToDevice);
   transpose<<<grid,block>>>(db,n,n,dbt);  
   cudaDeviceSynchronize();
   multiplication<<<grid,block>>>(da,dbt,dc,n);
   cudaDeviceSynchronize();
   cudaMemcpy(a,dc,sizeof(int)*n*n,cudaMemcpyDeviceToHost);
   for(int i=0;i<n;i++){
    for(int j=0;j<n;j++){
        printf("%d ",a[i*n+j]);
    }
    printf("\n");
   }
   
     

}