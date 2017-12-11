#ifndef VJ_KERNELS_CU
#define VJ_KERNELS_CU

#define STAGE_THRESH_MULTIPLIER 0.85

#include <stdint.h>

// function prototypes
__device__ void row_scan(int *in_data, int *out_data, int width);
__device__ void transpose();

// read-only memory (from device perspective) for filter data
__constant__ float stage_thresholds[25];
__constant__ int stage_lengths[25];
// __constant__ float stages_1_to_11[596];
// __constant__ float stages_12_to_15[513];
// __constant__ float stages_16_to_19[620];

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
__global__ void nearest_neighbor_row_scan_kernel(
	int in_width, int in_height, int out_width, int out_height,
	unsigned char *in, unsigned char *scaled, int *sum, int *squ
);
__global__ void col_scan_kernel(
	MyImage *in_scaled, MyIntImage *sum, MyIntImage *squ
);
__global__ void cascade_segment1_kernel(
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

__forceinline__ __device__ float eval_weak_classifier(
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
	float threshold = stage_data[filter_index + 15];

	if (sum >= threshold) {
		// see class.txt format: 17th line of each filter is its right child
		return stage_data[filter_index + 17];
	} else {
		// see class.txt format: 17th line of each filter is its left child
		return stage_data[filter_index + 16];
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
		shared_stage_data[i * 1024] = stage_data[i * 1024];	
	}
	// some divergence is inevitable here
	if (i * 1024 < 18 * 596) {
		shared_stage_data[i * 1024] = stage_data[i * 1024];
	}

	int window_start_x = blockDim.x * blockIdx.x + threadIdx.x;
	int window_start_y = blockDim.y * blockIdx.y + threadIdx.y;
	if (window_start_x > width - 24 || window_start_y > height - 24) {
		// edge case: this window lies outside the image boundaries
		return;
	}
	// calculate the standard deviation of the pixel values in the window
	int img_start_index = width * window_start_y + window_start_x;
	int integral = sum[img_start_index + width * 24 + 24]
				  - sum[img_start_index + width * 24]
				  - sum[img_start_index + 24] + sum[img_start_index];
	int square_integral = squ[img_start_index + width * 24 + 24]
				  - squ[img_start_index + width * 24]
				  - squ[img_start_index + 24] + squ[img_start_index];
	float std_dev = square_integral * 24 * 24 - integral * integral;

	uint8_t isFaceCandidate = 1;
	// process stage 1
	float stage1sum = 0.0f;
	int filter_index = 0;
	for (i = 0; i < stage_lengths[0]; i++) {
		stage1sum += eval_weak_classifier(
			std_dev, sum + img_start_index, filter_index, shared_stage_data
		);
		filter_index += 18;
	}
	// Rejection does not disqualify any window from stages within this
	// segment, for the sake of preserving the SIMD layout.
	if (stage1sum < STAGE_THRESH_MULTIPLIER * stage_thresholds[0]) {
		isFaceCandidate = 0;
	}
	// process stage 2
	float stage2sum = 0.0f;
	filter_index = 1 * (18 * stage_lengths[1] + 1);
	for (i = 0; i < stage_lengths[1]; i++) {
		stage2sum += eval_weak_classifier(
			std_dev, sum + img_start_index, filter_index, shared_stage_data
		);
		filter_index += 18;
	}
	if (stage2sum < STAGE_THRESH_MULTIPLIER * stage_thresholds[1]) {
		isFaceCandidate = 0;
	}
	// process stage 3
	float stage3sum = 0.0f;
	filter_index = 2 * (18 * stage_lengths[2] + 1);
	for (i = 0; i < stage_lengths[2]; i++) {
		stage3sum += eval_weak_classifier(
			std_dev, sum + img_start_index, filter_index, shared_stage_data
		);
		filter_index += 18;
	}
	if (stage3sum < STAGE_THRESH_MULTIPLIER * stage_thresholds[2]) {
		isFaceCandidate = 0;
	}
	// process stage 4
	float stage4sum = 0.0f;
	filter_index = 3 * (18 * stage_lengths[3] + 1);
	for (i = 0; i < stage_lengths[3]; i++) {
		stage4sum += eval_weak_classifier(
			std_dev, sum + img_start_index, filter_index, shared_stage_data
		);
		filter_index += 18;
	}
	if (stage4sum < STAGE_THRESH_MULTIPLIER * stage_thresholds[3]) {
		isFaceCandidate = 0;
	}
	// process stage 5
	float stage5sum = 0.0f;
	filter_index = 4 * (18 * stage_lengths[4] + 1);
	for (i = 0; i < stage_lengths[4]; i++) {
		stage5sum += eval_weak_classifier(
			std_dev, sum + img_start_index, filter_index, shared_stage_data
		);
		filter_index += 18;
	}
	if (stage5sum < STAGE_THRESH_MULTIPLIER * stage_thresholds[4]) {
		isFaceCandidate = 0;
	}
	// the result array is a large, flattened 2D array, and the unique index
	// is retrieved from the thread's coordinates within the grid.
	int result_index = width * window_start_y + window_start_x;
	results[result_index] = isFaceCandidate;
}

#endif
