__global__ void kernel() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Your CUDA kernel here
    printf("Hello World Cuda\n");
}

int main() {
    int *h_a, *d_a;
    h_a = (int *)malloc(sizeof(int));
    cudaMalloc((void **)&d_a, sizeof(int));
    int n;cin>>n;
    kernel<<<1, n>>>();
    cudaDeviceSynchronize();
}