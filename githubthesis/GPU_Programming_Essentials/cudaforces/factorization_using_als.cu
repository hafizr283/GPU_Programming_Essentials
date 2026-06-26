
__global__ void kernel(int *r, int *u, int *v, int *resu, int *resv, int n,int k){
    // complete r = u*v matrix factorization in als method

}
int main(){
    int *r;
    int n;scanf("%d",&n);
    int k;scanf("%d",&k);
    int size = sizeof(int)*n*n;
    r = (int*) malloc(size);
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            scanf("%d",*a[i*n+j]);
        }
    }
   float *u = (float*)malloc(sizeof(int)*k);
   float *v = (float*)malloc(sizeof(int)*k);
   for(int i=0;i<n;i++){
    for(int j=0;j<k;j++){
        u[i*n+j]=1.0;
        v[i*n+j]=2.0;
    }
   }
   int *dr, *du, *dv,*dresu, *dresv;
   cudaMalloc((void**)&du,sizeof(int)*n*k);
   cudaMalloc((void**)&dv,sizeof(int)*n*k);
   cudaMalloc((void**)&dr,sizeof(int)*n*k);
   cudaMalloc((void**)&dresu,sizeof(int)*n*k);
   cudaMalloc((void**)&dresv,sizeof(int)*n*k);
   cudaMemcpy(du,u,sizeof(int)*n*k,cudaMemcpyHostToDevice);
   cudaMemcpy(dv,v,sizeof(int)*n*k,cudaMemcpyHostToDevice);
   cudaMemcpy(dr,r,sizeof(int)*n*k,cudaMemcpyHostToDevice);
   dim3 block(16,16);
   kernel<<<1,block>>>(dr,du,dv,dresu,dresv,n,k);
   

} 