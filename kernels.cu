#ifndef VJ_KERNELS_CU
#define VJ_KERNELS_CU

// function prototypes
__device__ void row_scan(int *in_data, int *out_data, int width);
__device__ void transpose();

__global__ void nearest_neighbor_row_scan_kernel(
	int in_width, int in_height, int out_width, int out_height,
	unsigned char *in, unsigned char *scaled, int *sum, int *squ
);
__global__ void col_scan_kernel(
	MyImage *in_scaled, MyIntImage *sum, MyIntImage *squ
);

__device__ void row_scan(int *in_data, int *out_data, int width) {

}

__device__ void transpose() {

}

__global__ void nearest_neighbor_row_scan_kernel(
	int in_width, int in_height, int out_width, int out_height,
	unsigned char *in, unsigned char *scaled, int *sum, int *squ
) {
	// nearest neighbor
	float x_ratio = in_width / (float) out_width;
	float y_ratio = in_height / (float) out_height;

	int r_out = blockIdx.y;
	int c_out = blockIdx.x * blockDim.x + threadIdx.x;
	int r_in = (int) r_out * x_ratio;
	int c_in = (int) c_out * y_ratio;

	int idx_in = r_in * in_width + c_in;
	int idx_out = r_out * out_width + c_out;
	scaled[idx_out] = in[idx_in];

	// row_scan();	
}

__global__ void col_scan_kernel(
	MyImage *in_scaled, MyIntImage *sum, MyIntImage *squ
) {
	// transpose()
	// row_scan()
	// transpose()
}

#endif
