#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cstring>
#include <cmath>

#define TILE_WIDTH 32 

int printArray(unsigned char* arrayToPrint);
void printDeviceProperties(cudaDeviceProp deviceProp);

void generate_random_matrix(float *mat, unsigned long matrix_height, unsigned long matrix_width) {
    for (int i = 0; i < matrix_height * matrix_width; i++) {
        mat[i] = static_cast<float>(rand() % 100);
    }
}

void printResults(float *h_matA, float *h_matB, float *h_matC, int n, int k, int m){
	printf("Matrix A:\n");
	for(int i=0; i< (n*k); i++){
		printf("%f	", h_matA[i]);
		if( (i+1) % k  == 0 ){
			printf("\n");
		}

	}
	printf("Matrix B:\n");
	for(int i=0; i< (k * m); i++){
		printf("%f	", h_matB[i]);
		if( (i+1) % m  == 0 ){
			printf("\n");
		}
	}

	printf("Matrix C:\n");
	for(int i=0; i< (n * m); i++){
		printf("%f	", h_matC[i]);
		if( (i+1) % m  == 0 ){
			printf("\n");
		}
	}
}

__global__ void matmul_rec_glob(float *a, float *b, float *c, int n, int k, int m) { 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if( col < m && row < n) {
    	float sum = 0.0f;
        for(int i = 0; i < k; i++) {
            sum += a[row*k + i] * b[i*m + col];
        }
        c[row * m + col] = sum;
    }
}


__global__ void matmul_rec_shared(float *a, float *b, float *c, int n, int k, int m) {

	__shared__ float sA[TILE_WIDTH][TILE_WIDTH];
	__shared__ float sB[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x, by = blockIdx.y,
       tx = threadIdx.x, ty = threadIdx.y,
       Row = by * TILE_WIDTH + ty,
       Col = bx * TILE_WIDTH + tx;
    float Pvalue = 0;

    for (int i = 0; i < (k + TILE_WIDTH - 1) / TILE_WIDTH; ++i) {
       if (Row < n && i*TILE_WIDTH+tx < k)
          sA[ty][tx] = a[Row*k + i*TILE_WIDTH+tx];
       else
          sA[ty][tx] = 0;
        
       if (Col < m && i*TILE_WIDTH+ty < k)
          sB[ty][tx] = b[(i*TILE_WIDTH+ty)*m + Col];
       else
          sB[ty][tx] = 0;

       __syncthreads();
       for (int j = 0; j < TILE_WIDTH; ++j)
          Pvalue += sA[ty][j] * sB[j][tx];
       __syncthreads();
    }
    if (Row < n && Col < m)
       c[Row*m+Col] = Pvalue;
}

int main(int argc, char* argv[]) {
    srand(time(0));
    int matrix_1_height;
    int matrix_1_width;
    int matrix_2_height;
    int matrix_2_width;

    for (int i = 0; i < argc; i++) {
        if(strcmp(argv[i], "--matrix_1_height") == 0 && i + 1 < argc) {
            matrix_1_height = atoi(argv[i+1]);
            printf("Matrix 1 height: %d\n", matrix_1_height);
        }

        if(strcmp(argv[i], "--matrix_1_width") == 0 && i + 1 < argc) {
            matrix_1_width = atoi(argv[i+1]);
            matrix_2_height = atoi(argv[i+1]);
            printf("Matrix 1 width: %d\n", matrix_1_width);
            printf("Matrix 2 height: %d\n", matrix_2_height);
        }

        if(strcmp(argv[i], "--matrix_2_width") == 0 && i + 1 < argc) {
            matrix_2_width = atoi(argv[i+1]);
            printf("Matrix 1 height: %d\n", matrix_2_width);
        }
    }

    FILE* csv = fopen("/content/drive/MyDrive/CS 239/Backups/exercise_2/code/matrix_mul.csv", "a");
    if (!csv) {
        printf("Failed to open CSV file\n");
        return 1;
    }
    fprintf(csv, "kernel,mat_1_size,mat_2_size,result_size,run,threads_per_block,blocks_per_grid,time_ms\n");

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("Found %d CUDA capable device(s).\n", deviceCount);
    int device;
    for (device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        printDeviceProperties(deviceProp);
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 blocksPerGrid(
        (matrix_2_width + TILE_WIDTH - 1) / TILE_WIDTH,
        (matrix_1_height + TILE_WIDTH - 1) / TILE_WIDTH
    );
    int threadsPerBlockCount = TILE_WIDTH * TILE_WIDTH;
    int blocksPerGridCount = blocksPerGrid.x * blocksPerGrid.y;

    float *h_A, *h_B, *h_C;
    h_A = (float*) malloc(sizeof(float) * matrix_1_height * matrix_1_width);
    h_B = (float*) malloc(sizeof(float) * matrix_2_height * matrix_2_width);
    h_C = (float*) malloc(sizeof(float) * matrix_1_height * matrix_2_width);

    float *d_A, *d_B, *d_C;
    cudaMalloc((void **) &d_A, sizeof(float) * matrix_1_height * matrix_1_width);
    cudaMalloc((void **) &d_B, sizeof(float) * matrix_2_height * matrix_2_width);
    cudaMalloc((void **) &d_C, sizeof(float) * matrix_1_height * matrix_2_width);

    cudaEvent_t start,end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    float ms = 0.0;

    for(int i = 0; i<10; i++) {
        generate_random_matrix(h_A, matrix_1_height, matrix_1_width);
        generate_random_matrix(h_B, matrix_2_height, matrix_2_width);
        cudaMemcpy(d_A, h_A, sizeof(float) * matrix_1_height * matrix_1_width, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, sizeof(float) * matrix_2_height * matrix_2_width, cudaMemcpyHostToDevice);
        cudaEventRecord(start);
        matmul_rec_glob<<< blocksPerGrid, threadsPerBlock >>>(d_A, d_B, d_C, matrix_1_height, matrix_1_width, matrix_2_width);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(err));
        }
        cudaEventRecord(end);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&ms, start, end);
		cudaMemcpy(h_C, d_C, sizeof(float) * matrix_1_height * matrix_2_width, cudaMemcpyDeviceToHost);
        fprintf(csv, "global,%dx%d,%dx%d,%dx%d,%d,%d,%d,%.4f\n", matrix_1_height, matrix_1_width, matrix_2_height, matrix_2_width, matrix_1_height, matrix_2_width, i + 1, threadsPerBlockCount, blocksPerGridCount, ms);
        fflush(csv);
	}

    for(int i = 0; i<10; i++) {
        generate_random_matrix(h_A, matrix_1_height, matrix_1_width);
        generate_random_matrix(h_B, matrix_2_height, matrix_2_width);
        cudaMemcpy(d_A, h_A, sizeof(float) * matrix_1_height * matrix_1_width, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, sizeof(float) * matrix_2_height * matrix_2_width, cudaMemcpyHostToDevice);
        cudaEventRecord(start);
        matmul_rec_shared<<< blocksPerGrid, threadsPerBlock >>>(d_A, d_B, d_C, matrix_1_height, matrix_1_width, matrix_2_width);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(err));
        }
        cudaEventRecord(end);
        cudaEventSynchronize(end);
		cudaEventElapsedTime(&ms, start, end);

		cudaMemcpy(h_C, d_C, sizeof(float) * matrix_1_height * matrix_2_width, cudaMemcpyDeviceToHost); 
        fprintf(csv, "shared,%dx%d,%dx%d,%dx%d,%d,%d,%d,%.4f\n", matrix_1_height, matrix_1_width, matrix_2_height, matrix_2_width, matrix_1_height, matrix_2_width, i + 1, threadsPerBlockCount, blocksPerGridCount, ms);
        fflush(csv);
	}
    fclose(csv);
    // printResults(h_A, h_B, h_C, matrix_1_height, matrix_1_width, matrix_2_width);

    cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	free(h_A);
	free(h_B);
	free(h_C);

	return 0;
	
}

int printArray(unsigned char* arrayToPrint) {
    for (int i = 0; i < 16; i++) {
        printf("%02x", arrayToPrint[i]);
    }
    return 0;
}

void printDeviceProperties(cudaDeviceProp deviceProp) {
    printf("Device Name: %s\n", deviceProp.name);
    printf("Device UUID: ");
    printArray((unsigned char*)deviceProp.uuid.bytes);
    printf("\n");
    printf("LUID: ");
    printArray((unsigned char*) deviceProp.luid);
    printf("\n");
    printf("LUID Device Node Mask: %u\n", deviceProp.luidDeviceNodeMask);
    printf("Total Global Memory: %zu bytes\n", deviceProp.totalGlobalMem);
    printf("Shared Memory per Block: %zu bytes\n", deviceProp.sharedMemPerBlock);
    printf("Registers per Block: %d\n", deviceProp.regsPerBlock);
    printf("Warp Size: %d\n", deviceProp.warpSize);
    printf("Memory Pitch: %zu bytes\n", deviceProp.memPitch);
    printf("Max Threads per Block: %d\n", deviceProp.maxThreadsPerBlock);
    printf("Max Threads Dim:  %d x %d x %d\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
    printf("Max Grid Size: %d x %d x %d\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
    printf("Total Constant Memory: %zu bytes\n", deviceProp.totalConstMem);
    printf("Major: %d\n", deviceProp.major);
    printf("Minor: %d\n", deviceProp.minor);
    printf("Compute Capability: %d.%d\n", deviceProp.major, deviceProp.minor);
    printf("Texture Alignment: %zu\n", deviceProp.textureAlignment);
    printf("Texture Pitch: %zu\n", deviceProp.texturePitchAlignment);
    printf("MultiProcessor Count: %d\n", deviceProp.multiProcessorCount);
    printf("Integrated: %d\n", deviceProp.integrated);
    printf("Can Map Host Memory: %d\n", deviceProp.canMapHostMemory);
    printf("Max Texture 1D: %d\n", deviceProp.maxTexture1D);
    printf("Max Texture 1D Mipmap: %d\n", deviceProp.maxTexture1DMipmap);
    printf("Max Texture 2D:  %d x %d\n", deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1]);
    printf("Max Texture 2D Mipmap:  %d x %d\n", deviceProp.maxTexture2DMipmap[0], deviceProp.maxTexture2DMipmap[1]);
    printf("Max Texture 2D Linear:  %d x %d x %d\n", deviceProp.maxTexture2DLinear[0], deviceProp.maxTexture2DLinear[1], deviceProp.maxTexture2DLinear[2]);
    printf("Max Texture 2D Gather:  %d x %d\n", deviceProp.maxTexture2DGather[0], deviceProp.maxTexture2DGather[1]);
    printf("Max Texture 3D:  %d x %d x %d\n", deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);
    printf("Max Texture 3D Alt:  %d x %d x %d\n", deviceProp.maxTexture3DAlt[0], deviceProp.maxTexture3DAlt[1], deviceProp.maxTexture3DAlt[2]);
    printf("Max Texture Cubemap: %d\n", deviceProp.maxTextureCubemap);
    printf("Max Texture 1D Layered:  %d x %d\n", deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1]);
    printf("Max Texture 2D Layered:  %d x %d x %d\n", deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1], deviceProp.maxTexture2DLayered[2]);
    printf("Max Texture Cubemap Layered:  %d x %d\n", deviceProp.maxTextureCubemapLayered[0], deviceProp.maxTextureCubemapLayered[1]);
    printf("Max Surface 1D: %d\n", deviceProp.maxSurface1D);
    printf("Max Surface 2D:  %d x %d\n", deviceProp.maxSurface2D[0], deviceProp.maxSurface2D[1]);
    printf("Max Surface 3D:  %d x %d x %d\n", deviceProp.maxSurface3D[0], deviceProp.maxSurface3D[1], deviceProp.maxSurface3D[2]);
    printf("Max Surface 1D Layered: %d x %d\n", deviceProp.maxSurface1DLayered[0], deviceProp.maxSurface1DLayered[1]);
    printf("Max Surface 2D Layered: %d x %d x %d\n", deviceProp.maxSurface2DLayered[0], deviceProp.maxSurface2DLayered[1], deviceProp.maxSurface2DLayered[2]);
    printf("Max Surface Cubemap: %d\n", deviceProp.maxSurfaceCubemap);
    printf("Max Surface Cubemap Layered: %d x %d\n", deviceProp.maxSurfaceCubemapLayered[0], deviceProp.maxSurfaceCubemapLayered[1]);
    printf("Surface Alignment: %zu\n", deviceProp.surfaceAlignment);
    printf("Concurrent Kernels: %d\n", deviceProp.concurrentKernels);
    printf("ECC Enabled: %d\n", deviceProp.ECCEnabled);
    printf("PCI Bus ID: %d\n", deviceProp.pciBusID);
    printf("PCI Device ID: %d\n", deviceProp.pciDeviceID);
    printf("PCI Domain ID: %d\n", deviceProp.pciDomainID);
    printf("TCC Driver: %d\n", deviceProp.tccDriver);
    printf("Async Engine Count: %d\n", deviceProp.asyncEngineCount);
    printf("Unified Addressing: %d\n", deviceProp.unifiedAddressing);
    printf("Memory Bus Width: %d bits\n", deviceProp.memoryBusWidth);
    printf("L2 Cache Size: %d bytes\n", deviceProp.l2CacheSize);
    printf("Persisting L2 Cache Max Size: %d bytes\n", deviceProp.persistingL2CacheMaxSize);
    printf("Max Threads per Multiprocessor: %d\n", deviceProp.maxThreadsPerMultiProcessor);
    printf("Stream Priorities Supported: %d\n", deviceProp.streamPrioritiesSupported);
    printf("Global L1 Cache Supported: %d\n", deviceProp.globalL1CacheSupported);
    printf("Local L1 Cache Supported: %d\n", deviceProp.localL1CacheSupported);
    printf("Shared Memory per Multiprocessor: %zu bytes\n", deviceProp.sharedMemPerMultiprocessor);
    printf("Registers per Multiprocessor: %d\n", deviceProp.regsPerMultiprocessor);
    printf("Managed Memory Supported: %d\n", deviceProp.managedMemory);
    printf("Is Multi-GPU Board: %d\n", deviceProp.isMultiGpuBoard);
    printf("Multi-GPU Board Group ID: %d\n", deviceProp.multiGpuBoardGroupID);
    printf("Host Native Atomics Supported: %d\n", deviceProp.hostNativeAtomicSupported);
    printf("Pageable Memory Access: %d\n", deviceProp.pageableMemoryAccess);
    printf("Concurrent Managed Access: %d\n", deviceProp.concurrentManagedAccess);
    printf("Compute Preemption Supported: %d\n", deviceProp.computePreemptionSupported);
    printf("Can Use Host Pointer for Registered Mem: %d\n", deviceProp.canUseHostPointerForRegisteredMem);
    printf("Cooperative Launch: %d\n", deviceProp.cooperativeLaunch);
    printf("Shared Memory per Block Optin: %zu bytes\n", deviceProp.sharedMemPerBlockOptin);
    printf("Pageable Memory Access Uses Host Page Tables: %d\n", deviceProp.pageableMemoryAccessUsesHostPageTables);
    printf("Direct Managed Mem Access from Host: %d\n", deviceProp.directManagedMemAccessFromHost);
    printf("Max Blocks per Multiprocessor: %d\n", deviceProp.maxBlocksPerMultiProcessor);
    printf("Access Policy Max Window Size: %d\n", deviceProp.accessPolicyMaxWindowSize);
    printf("Reserved Shared Mem per Block: %zu bytes\n", deviceProp.reservedSharedMemPerBlock);
    printf("Host Register Supported: %d\n", deviceProp.hostRegisterSupported);
    printf("Sparse CUDA Array Supported: %d\n", deviceProp.sparseCudaArraySupported);
    printf("Host Register Read-Only Supported: %d\n", deviceProp.hostRegisterReadOnlySupported);
    printf("Timeline Semaphore Interop Supported: %d\n", deviceProp.timelineSemaphoreInteropSupported);
    printf("Memory Pools Supported: %d\n", deviceProp.memoryPoolsSupported);
    printf("GPU Direct RDMA Supported: %d\n", deviceProp.gpuDirectRDMASupported);
    printf("GPU Direct RDMA Flush Writes Options: %u\n", deviceProp.gpuDirectRDMAFlushWritesOptions);
    printf("GPU Direct RDMA Writes Ordering: %d\n", deviceProp.gpuDirectRDMAWritesOrdering);
    printf("Memory Pool Supported Handle Types: %u\n", deviceProp.memoryPoolSupportedHandleTypes);
    printf("Deferred Mapping CUDA Array Supported: %d\n", deviceProp.deferredMappingCudaArraySupported);
    printf("IPC Event Supported: %d\n", deviceProp.ipcEventSupported);
    printf("Cluster Launch: %d\n", deviceProp.clusterLaunch);
    printf("Unified Function Pointers: %d\n", deviceProp.unifiedFunctionPointers);
}