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
     if(row<n && col<n){
        int sum = 0;
        for(int i=0;i<n;i++){
            sum+=(a[row*n+i]*b[col*n+i]);
        }
        c[row*n+col]=sum;
        
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
   dim3 block(16,16);
   cudaMemcpy(da,a,sizeof(int)*n*n,cudaMemcpyHostToDevice);
   cudaMemcpy(db,b,sizeof(int)*n*n,cudaMemcpyHostToDevice);
   cudaMemcpy(dc,c,bytes,cudaMemcpyHostToDevice);
   transpose<<<1,block>>>(db,n,n,dbt);  
   cudaDeviceSynchronize();
   multiplication<<<1,block>>>(da,dbt,dc,n);
   cudaDeviceSynchronize();
   cudaMemcpy(a,dc,sizeof(int)*n*n,cudaMemcpyDeviceToHost);
   for(int i=0;i<n;i++){
    for(int j=0;j<n;j++){
        printf("%d ",a[i*n+j]);
    }
    printf("\n");
   }
   
     

}