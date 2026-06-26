__global__ void kernel(int *a, int *res) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx==0) res[idx]=min(a[0],a[1]);
    else res[idx]=max(a[0],a[1]);
}

int main() {
    int *h_a, *d_a,*dres;
    int a,b;
    scanf("%d %d",&a,&b);
    h_a = (int *)malloc(sizeof(int)*2);
    h_a[0]=a;
    h_a[1]=b;
    cudaMalloc((void **)&d_a, sizeof(int)*2);
    cudaMemcpy(d_a,h_a,sizeof(int)*2,cudaMemcpyHostToDevice);
    cudaMalloc((void **)&dres, sizeof(int)*2);

    kernel<<<1, 2>>>(d_a,dres);
    cudaMemcpy(h_a,dres,sizeof(int)*2,cudaMemcpyDeviceToHost);
    printf("%d %d",h_a[0],h_a[1]);
    cudaDeviceSynchronize();
}