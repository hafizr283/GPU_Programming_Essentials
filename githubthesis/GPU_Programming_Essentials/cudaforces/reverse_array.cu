
__global__ void kernel(int *a, int n, int *b){
    int tid = threadIdx.x;
    if(tid<n) b[tid]=a[n-1-tid];
}

int main(){
        int n;scanf("%d",&n);
        int size = sizeof(int)*n;
        int *a = (int *) malloc(sizeof(int)*n);
        for(int i=0;i<n;i++){
            scanf("%d",&a[i]);
        }
        int *da, *dres,*res;
         cudaMalloc((void**)&da, size);
         cudaMemcpy(da,a,size,cudaMemcpyHostToDevice);
         res= (int*) malloc(size);
         cudaMalloc((void**)&dres,size);
         kernel<<<1,256>>>(da,n,dres);
         cudaMemcpy(res,dres,size,cudaMemcpyDeviceToHost);
         for(int i=0;i<n;i++){
            printf("%d ",res[i]);
         }
         printf("\n");





}