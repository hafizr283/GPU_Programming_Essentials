__global__ void kernel(int *dmat, int n, int a, int b, int c, int d, int k, int *dres){
    int x = threadIdx.x;
    int y = threadIdx.y;
    if(x>=a && y>=b && x<=c && y<=d){
        *dres = *dres + k*dmat[x*n+y];
    }
    else if(x<n && y<n)
    *dres = *dres+dmat[n*x][y];
}
int main(){
    
    int n;scanf("%d",&n);
    int size = sizeof(int)*n*n;
        int *mat = (int*) malloc(size);
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            scanf("%d",&mat[i*n+j]);
        }
    }
    int a,b,c,d,k;
    scanf("%d %d %d %d %d",&a,&b,&c,&d,&k);
    int *dmat, *dres,*res=(int*)malloc(sizeof(int));

    cudaMalloc((void**)&dmat,size);
    cudaMemcpy(dmat,mat,size, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&dres,sizeof(int));
    cudaMemset(dres,0,sizeof(int));
    dim3 block(16,16);
    kernel<<<1,block>>>(dmat,n,a,b,c,d,k,dres);
    cudaMemcpy(res,dres,sizeof(int),cudaMemcpyDeviceToHost);
    printf("%d\n",*res);
    
}