#ifndef VJ_KERNELS_CU
#define VJ_KERNELS_CU

#define STAGE_THRESH_MULTIPLIER 0.4

#define EVAL_STAGE(N) \
	do { \
		stage_sum = 0.0f; \
		for (i = 0; i < stage_lengths[N - 1]; i++) { \
			stage_sum += eval_weak_classifier( \
				std_dev, sum + img_start_index, filter_index, shared_stage_data \
			); \
			filter_index += 18; \
		} \
		if (stage_sum < STAGE_THRESH_MULTIPLIER * stage_thresholds[N - 1]) { \
			isFaceCandidate = 0; \
		} \
	} while (0)

#include <stdint.h>

// function prototypes
__device__ void row_scan(int *in_data, int *out_data, int width);
__device__ void transpose();

// read-only memory (from device perspective) for filter stage meta-data
__constant__ float stage_thresholds[25];
__constant__ int stage_lengths[25];

/* Cascade segment breakdown:
 * Each segment has a corresponding kernel, of which each thread evaluates
 * one detection window. The set of detection windows passed to successive
 * kernels will decrease as windows are ruled out.
 * 
 * Memory usage based on 18 ints (or floats) per node == 72 B.
 * # / first stage / last stage / SHMEM usage / total nodes (filters)
 * 1 | 1           | 11         | 42 K        | 596
 * 2 | 12          | 15         | 36 K        | 513
 * 3 | 16          | 19         | 44 K        | 620
 * 4 | 20          | 22         | 40 K        | 574
 * 5 | 23          | 25         | 43 K        | 610
 */

// kernel prototypes
/* __global__ void nearest_neighbor_row_scan_kernel(
	int in_width, int in_height, int out_width, int out_height,
	unsigned char *in, unsigned char *scaled, int *sum, int *squ
);
__global__ void col_scan_kernel(
	MyImage *in_scaled, MyIntImage *sum, MyIntImage *squ
);
__global__ void cascade_segment1_kernel(
); */

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

// The pointer *integral_data holds the address of the top-left coordinate
// of the rectangle to be evaluated.
// TODO: address memory coalescing issues when
// shifting between rows of the detection window.
// TODO: evaluate register limit issues vs performance increase with __forceinline__
__device__ float eval_weak_classifier(
	float std_dev, int *integral_data, int filter_index, float *stage_data
) {
	int rect_index = filter_index;
	int weight_index = filter_index + 4; // skip the 4 coords of first rectangle
	float sum = (*(integral_data + (int) stage_data[rect_index])
			  - *(integral_data + (int) stage_data[rect_index + 1])
			  - *(integral_data + (int) stage_data[rect_index + 2])
			  + *(integral_data + (int) stage_data[rect_index + 3]))
			  * stage_data[weight_index];
	weight_index += 4;
	rect_index += 4;
	sum += (*(integral_data + (int) stage_data[rect_index])
	 	   - *(integral_data + (int) stage_data[rect_index + 1])
	  	   - *(integral_data + (int) stage_data[rect_index + 2])
	  	   + *(integral_data + (int) stage_data[rect_index + 3]))
	  	   * stage_data[weight_index];
	weight_index += 4;
	rect_index += 4;
	sum += (*(integral_data + (int) stage_data[rect_index])
	 	   - *(integral_data + (int) stage_data[rect_index + 1])
	  	   - *(integral_data + (int) stage_data[rect_index + 2])
	  	   + *(integral_data + (int) stage_data[rect_index + 3]))
	  	   * stage_data[weight_index];
	// see class.txt format: 16th line of each filter is its threshold
	float threshold = stage_data[filter_index + 15] * std_dev;
	/* if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
		printf("thread 0");
	} */

	if (sum < threshold) {
		// see class.txt format: 16th line of each filter is its left child
		return stage_data[filter_index + 16];
	} else {
		// see class.txt format: 17th line of each filter is its right child
		return stage_data[filter_index + 17];
	}
}

__global__ void cascade_segment1_kernel(
	int *sum, int *squ, uint8_t *results,
	float *stage_data, int width, int height
) {
	int i;
	// this flattened float array is in the same format as class.txt;
	// coordinates of rectangles do not need to be stored as floats,
	// unless all the data is kept in this single array.
	__shared__ float shared_stage_data[18 * 596];
	for (i = 0; i < 10; i++) {
		shared_stage_data[threadIdx.x + i * 1024] =
			stage_data[threadIdx.x + i * 1024];	
	}
	// some divergence is inevitable here
	if (i * 1024 + threadIdx.x < 18 * 596) {
		shared_stage_data[threadIdx.x + i * 1024] =
			stage_data[threadIdx.x + i * 1024];
	}
	__syncthreads();

	int window_start_x = blockDim.x * blockIdx.x + threadIdx.x;
	int window_start_y = blockDim.y * blockIdx.y + threadIdx.y;
	if (window_start_x > width - 24 || window_start_y > height - 24) {
		// edge case: this window lies outside the image boundaries
		return;
	}
	// calculate the standard deviation of the pixel values in the window
	int img_start_index = width * window_start_y + window_start_x;
	/* if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
		printf("%d, %d\n", window_start_x, window_start_y);
	} */
	unsigned int integral = sum[img_start_index + width * 23 + 23]
				  - sum[img_start_index + width * 23]
				  - sum[img_start_index + 23] + sum[img_start_index];
	unsigned int square_integral = squ[img_start_index + width * 23 + 23]
				  - squ[img_start_index + width * 23]
				  - squ[img_start_index + 23] + squ[img_start_index];
	float std_dev = square_integral * 24 * 24 - integral * integral;
	if (std_dev > 0) {
		std_dev = sqrt(std_dev);
	} else {
		std_dev = 1;
	}

	uint8_t isFaceCandidate = 1;
	int filter_index = 0;
	// filter_index = (N - 1) * (18 * stage_lengths[N - 1] + 1);
	float stage_sum;
	// The EVAL_STAGE(N) macro eliminates the need to write 25 identical
	// for loops, threshold checks, &c.
	EVAL_STAGE(1);
	EVAL_STAGE(2);
	EVAL_STAGE(3);
	EVAL_STAGE(4);
	EVAL_STAGE(5);
	EVAL_STAGE(6);
	EVAL_STAGE(7);
	EVAL_STAGE(8);
	EVAL_STAGE(9);
	EVAL_STAGE(10);
	EVAL_STAGE(11);
	// the result array is a large, flattened 2D array, and the unique index
	// is retrieved from the thread's coordinates within the grid.
	int result_index = width * window_start_y + window_start_x;
	results[result_index] = isFaceCandidate;
}

__global__ void cascade_segment2_kernel(
	int *sum, int *squ, uint8_t *results,
	float *stage_data, int width, int height
) {
	int i;
	__shared__ float shared_stage_data[18 * 513];
	// FIXME: add offset for skipping first 11 stages
	for (i = 0; i < 10; i++) {
		shared_stage_data[threadIdx.x + i * 1024] =
			stage_data[threadIdx.x + i * 1024];	
	}
	// some divergence is inevitable here
	if (i * 1024 + threadIdx.x < 18 * 513) {
		shared_stage_data[threadIdx.x + i * 1024] =
			stage_data[threadIdx.x + i * 1024];
	}
	__syncthreads();

	int window_start_x = blockDim.x * blockIdx.x + threadIdx.x;
	int window_start_y = blockDim.y * blockIdx.y + threadIdx.y;
	if (window_start_x > width - 24 || window_start_y > height - 24) {
		// edge case: this window lies outside the image boundaries
		return;
	}
	
	// FIXME: huge divergence until re-grid
	// see comments in haar.cu:scale_image_invoker()
	int result_index = width * window_start_y + window_start_x;
	if (!results[result_index]) {
		return;
	}
	// calculate the standard deviation of the pixel values in the window
	int img_start_index = width * window_start_y + window_start_x;
	unsigned int integral = sum[img_start_index + width * 23 + 23]
				  - sum[img_start_index + width * 23]
				  - sum[img_start_index + 23] + sum[img_start_index];
	unsigned int square_integral = squ[img_start_index + width * 23 + 23]
				  - squ[img_start_index + width * 23]
				  - squ[img_start_index + 23] + squ[img_start_index];
	float std_dev = square_integral * 24 * 24 - integral * integral;
	if (std_dev > 0) {
		std_dev = sqrtf(std_dev / 256.0f);
	} else {
		std_dev = 1;
	}

	uint8_t isFaceCandidate = 1;
	int filter_index = 0;
	// The EVAL_STAGE(N) macro eliminates the need to write 25 identical
	// for loops, threshold checks, &c.
	float stage_sum;
	EVAL_STAGE(12);
	EVAL_STAGE(13);
	EVAL_STAGE(14);
	EVAL_STAGE(15);
	results[result_index] = isFaceCandidate;
}

__global__ void cascade_segment3_kernel(
	int *sum, int *squ, uint8_t *results,
	float *stage_data, int width, int height
) {
	int i;
	__shared__ float shared_stage_data[18 * 620];
	for (i = 0; i < 10; i++) {
		shared_stage_data[threadIdx.x + i * 1024] =
			stage_data[threadIdx.x + i * 1024];	
	}
	// some divergence is inevitable here
	if (i * 1024 + threadIdx.x < 18 * 620) {
		shared_stage_data[threadIdx.x + i * 1024] =
			stage_data[threadIdx.x + i * 1024];
	}
	__syncthreads();

	int window_start_x = blockDim.x * blockIdx.x + threadIdx.x;
	int window_start_y = blockDim.y * blockIdx.y + threadIdx.y;
	if (window_start_x > width - 24 || window_start_y > height - 24) {
		// edge case: this window lies outside the image boundaries
		return;
	}
	
	// FIXME: huge divergence until re-grid
	// see comments in haar.cu:scale_image_invoker()
	int result_index = width * window_start_y + window_start_x;
	if (!results[result_index]) {
		return;
	}
	// calculate the standard deviation of the pixel values in the window
	int img_start_index = width * window_start_y + window_start_x;
	unsigned int integral = sum[img_start_index + width * 23 + 23]
				  - sum[img_start_index + width * 23]
				  - sum[img_start_index + 23] + sum[img_start_index];
	unsigned int square_integral = squ[img_start_index + width * 23 + 23]
				  - squ[img_start_index + width * 23]
				  - squ[img_start_index + 23] + squ[img_start_index];
	float std_dev = square_integral * 24 * 24 - integral * integral;
	if (std_dev > 0) {
		std_dev = sqrt(std_dev / 256.0f);
	} else {
		std_dev = 1;
	}

	uint8_t isFaceCandidate = 1;
	int filter_index = 0;
	float stage_sum;
	EVAL_STAGE(16);
	EVAL_STAGE(17);
	EVAL_STAGE(18);
	EVAL_STAGE(19);
	results[result_index] = isFaceCandidate;
}

__global__ void cascade_segment4_kernel(
	int *sum, int *squ, uint8_t *results,
	float *stage_data, int width, int height
) {
	int i;
	__shared__ float shared_stage_data[18 * 574];
	for (i = 0; i < 10; i++) {
		shared_stage_data[threadIdx.x + i * 1024] =
			stage_data[threadIdx.x + i * 1024];	
	}
	// some divergence is inevitable here
	if (i * 1024 + threadIdx.x < 18 * 574) {
		shared_stage_data[threadIdx.x + i * 1024] =
			stage_data[threadIdx.x + i * 1024];
	}
	__syncthreads();

	int window_start_x = blockDim.x * blockIdx.x + threadIdx.x;
	int window_start_y = blockDim.y * blockIdx.y + threadIdx.y;
	if (window_start_x > width - 24 || window_start_y > height - 24) {
		// edge case: this window lies outside the image boundaries
		return;
	}
	
	// FIXME: huge divergence until re-grid
	// see comments in haar.cu:scale_image_invoker()
	int result_index = width * window_start_y + window_start_x;
	if (!results[result_index]) {
		return;
	}
	// calculate the standard deviation of the pixel values in the window
	unsigned int img_start_index = width * window_start_y + window_start_x;
	unsigned int integral = sum[img_start_index + width * 23 + 23]
				  - sum[img_start_index + width * 23]
				  - sum[img_start_index + 23] + sum[img_start_index];
	int square_integral = squ[img_start_index + width * 23 + 23]
				  - squ[img_start_index + width * 23]
				  - squ[img_start_index + 23] + squ[img_start_index];
	float std_dev = square_integral * 24 * 24 - integral * integral;
	if (std_dev > 0) {
		std_dev = sqrt(std_dev / 256.0f);
	} else {
		std_dev = 1;
	}

	uint8_t isFaceCandidate = 1;
	int filter_index = 0;
	float stage_sum;
	EVAL_STAGE(20);
	EVAL_STAGE(21);
	EVAL_STAGE(22);
	results[result_index] = isFaceCandidate;
}

__global__ void cascade_segment5_kernel(
	int *sum, int *squ, uint8_t *results,
	float *stage_data, int width, int height
) {
	int i;
	__shared__ float shared_stage_data[18 * 610];
	for (i = 0; i < 10; i++) {
		shared_stage_data[threadIdx.x + i * 1024] =
			stage_data[threadIdx.x + i * 1024];	
	}
	// some divergence is inevitable here
	if (i * 1024 + threadIdx.x < 18 * 610) {
		shared_stage_data[threadIdx.x + i * 1024] =
			stage_data[threadIdx.x + i * 1024];
	}
	__syncthreads();

	int window_start_x = blockDim.x * blockIdx.x + threadIdx.x;
	int window_start_y = blockDim.y * blockIdx.y + threadIdx.y;
	if (window_start_x > width - 24 || window_start_y > height - 24) {
		// edge case: this window lies outside the image boundaries
		return;
	}
	
	// FIXME: huge divergence until re-grid
	// see comments in haar.cu:scale_image_invoker()
	int result_index = width * window_start_y + window_start_x;
	if (!results[result_index]) {
		return;
	}
	// calculate the standard deviation of the pixel values in the window
	int img_start_index = width * window_start_y + window_start_x;
	unsigned int integral = sum[img_start_index + width * 23 + 23]
				  - sum[img_start_index + width * 23]
				  - sum[img_start_index + 23] + sum[img_start_index];
	unsigned int square_integral = squ[img_start_index + width * 23 + 23]
				  - squ[img_start_index + width * 23]
				  - squ[img_start_index + 23] + squ[img_start_index];
	float std_dev = square_integral * 24 * 24 - integral * integral;
	if (std_dev > 0) {
		std_dev = sqrt(std_dev / 256.0f);
	} else {
		std_dev = 1;
	}

	uint8_t isFaceCandidate = 1;
	int filter_index = 0;
	float stage_sum;
	EVAL_STAGE(23);
	EVAL_STAGE(24);
	EVAL_STAGE(25);
	results[result_index] = isFaceCandidate;
}
#endif
