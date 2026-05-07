#define TILE_SIZE 16

__global__ void multiplication_tiled(int *a, int* b, int* c, int n){
    // Shared memory তেরি করা হলো a এবং b এর ছোট ব্লকের (Tile) জন্য
    __shared__ int sA[TILE_SIZE][TILE_SIZE];
    __shared__ int sB[TILE_SIZE][TILE_SIZE];

    int row = threadIdx.y + blockIdx.y * TILE_SIZE;
    int col = threadIdx.x + blockIdx.x * TILE_SIZE;

    int sum = 0;

    // পুরো ম্যাট্রিক্সের উপর Tile গুলো স্লাইড করার লুপ
    for (int t = 0; t < (n + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        
        // a থেকে ডেটা এনে sA তে রাখা
        if (row < n && t * TILE_SIZE + threadIdx.x < n)
            sA[threadIdx.y][threadIdx.x] = a[row * n + t * TILE_SIZE + threadIdx.x];
        else
            sA[threadIdx.y][threadIdx.x] = 0;

        // b থেকে ডেটা এনে sB তে রাখা (transpose এর দরকার নেই)
        if (col < n && t * TILE_SIZE + threadIdx.y < n)
            sB[threadIdx.y][threadIdx.x] = b[(t * TILE_SIZE + threadIdx.y) * n + col];
        else
            sB[threadIdx.y][threadIdx.x] = 0;

        // সব থ্রেডের ডেটা লোড হওয়া পর্যন্ত অপেক্ষা করা
        __syncthreads();

        // Tile এর ভেতরের ডেটাগুলো গুণ করে যোগ করা
        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += sA[threadIdx.y][i] * sB[i][threadIdx.x];
        }

        // পরবর্তী Tile লোড করার আগে আবার অপেক্ষা করা
        __syncthreads();
    }

    if (row < n && col < n) {
        c[row * n + col] = sum;
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
   dim3 block(TILE_SIZE, TILE_SIZE);
   dim3 grid((n + TILE_SIZE - 1) / TILE_SIZE, (n + TILE_SIZE - 1) / TILE_SIZE);

   cudaMemcpy(da, a, sizeof(int)*n*n, cudaMemcpyHostToDevice);
   cudaMemcpy(db, b, sizeof(int)*n*n, cudaMemcpyHostToDevice);
   cudaMemcpy(dc, c, bytes, cudaMemcpyHostToDevice);
   
   // Tiled multiplication কল করা হচ্ছে (dbt বা transpose এর দরকার নেই)
   multiplication_tiled<<<grid, block>>>(da, db, dc, n);
   cudaDeviceSynchronize();
   cudaMemcpy(a,dc,sizeof(int)*n*n,cudaMemcpyDeviceToHost);
   for(int i=0;i<n;i++){
    for(int j=0;j<n;j++){
        printf("%d ",a[i*n+j]);
    }
    printf("\n");
   }
   
     

}