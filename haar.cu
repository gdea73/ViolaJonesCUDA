/*
 *  TU Eindhoven
 *  Eindhoven, The Netherlands
 *
 *  Name            :   haar.cpp
 *
 *  Author          :   Francesco Comaschi (f.comaschi@tue.nl)
 *
 *  Date            :   November 12, 2012
 *
 *  Function        :   Haar features evaluation for face detection
 *
 *  History         :
 *      12-11-12    :   Initial version.
 *
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program;  If not, see <http://www.gnu.org/licenses/>
 *
 * In other words, you are welcome to use, share and improve this program.
 * You are forbidden to forbid anyone else to use, share and improve
 * what you give them.   Happy coding!
 */

#include "haar.h"
#include "image.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "stdio-wrapper.h"

/* include segment kernel metadata */
#include "segments.h"
#include "kernels.cu"

//#define PARALLEL_SCAN_TRANSPOSE

// this is duplicated in kernels.cu for the GPU Haar cascade
#define STAGE_THRESH_MULTIPLIER_CPU 0.85
// depending on our choice of precision, we may want > 12 chars when reading
// floating-point values in class.txt
#define FGETS_BUF_SIZE 12
#define MAX_FACES 500


// one element per stage; holds stage lengths in number of filters
static int *stages_array;
static float *stages_thresh_array;
static int **scaled_rectangles_array;

// the host (CPU) copy of the filters
float *stage_data;
// GPU DRAM pointers; stage data allocated in read_text_classifiers
float *stage_data_GPU;
// allocated in scale_image_invoker only for the first pyramid iteration
static int *remaining_candidates;
static uint8_t *results;
#ifndef PARALLEL_SCAN_TRANSPOSE
static int *sum_GPU, *squ_GPU;
#endif
#ifdef PARALLEL_SCAN_TRANSPOSE
float *sum_GPU, *squ_GPU;
#endif

int n_features = 0;
int iter_counter = 0;

/* compute integral images */
void integralImages(MyImage *src, MyIntImage *sum, MyIntImage *sqsum);

/* scale the image down and detect faces at the new scale */
void scale_image_invoker(
	myCascade *cascade, float factor, int sum_height,
	int sum_width, std::vector<MyRect> &face_vector_output
);

/* compute scaled image */
void nearestNeighbor(MyImage *src, MyImage *dst);

/* rounding function */
inline int myRound (float value) {
	return (int)(value + (value >= 0 ? 0.5 : -0.5));
}

// // "warms up" GPU; helpful for discounting the 0.5-1s required to "spin up" CUDA
void init_GPU() {
	cudaFree(0);
}

std::vector<MyRect> detectObjects( MyImage* _img, MySize minSize, MySize maxSize, myCascade* cascade, float scaleFactor, int minNeighbors) {

	/* group overlaping windows */
	const float GROUP_EPS = 0.4f;
	/* pointer to input image */
	MyImage *img = _img;
	/***********************************
	 * create structs for images
	 * see haar.h for details
	 * img1: normal image (unsigned char)
	 * sum1: integral image (int)
	 * sqsum1: square integral image (int)
	 **********************************/
	MyImage image1Obj;
	MyIntImage sum1Obj;
	MyIntImage sqsum1Obj;
	/* pointers for the created structs */
	MyImage *img1 = &image1Obj;
	MyIntImage *sum1 = &sum1Obj;
	MyIntImage *sqsum1 = &sqsum1Obj;

	/********************************************************
	 * allCandidates is the preliminaray face candidate,
	 * which will be refined later.
	 *
	 * std::vector is a sequential container
	 * http://en.wikipedia.org/wiki/Sequence_container_(C++)
	 *
	 * Each element of the std::vector is a "MyRect" struct
	 * MyRect struct keeps the info of a rectangle (see haar.h)
	 * The rectangle contains one face candidate
	 *****************************************************/
	std::vector<MyRect> allCandidates;

	/* scaling factor */
	float factor;

	/* maxSize */
	if( maxSize.height == 0 || maxSize.width == 0 )
	{
		maxSize.height = img->height;
		maxSize.width = img->width;
	}

	/* window size of the training set */
	MySize winSize0 = cascade->orig_window_size;

	/* malloc for img1: unsigned char */
	createImage(img->width, img->height, img1);
	/* malloc for sum1: unsigned char */
	createSumImage(img->width, img->height, sum1);
	/* malloc for sqsum1: unsigned char */
	createSumImage(img->width, img->height, sqsum1);
	#ifdef PARALLEL_SCAN_TRANSPOSE
	// malloc GPU arrays
	cudaMalloc((void **) &sum_GPU, img->height*img->width* sizeof(float));
	cudaMalloc((void **) &squ_GPU, img->height*img->width * sizeof(float));
	unsigned char* imgdata = NULL;
	cudaMalloc( (void**) &imgdata,img->height*img->width*sizeof(unsigned char));
	#endif
	/* initial scaling factor */
	factor = 1;

	/* iterate over the image pyramid */
	for( factor = 1; ; factor *= scaleFactor )
	{
		/* iteration counter */
		iter_counter++;

		/* size of the image scaled up */
		MySize winSize = { myRound(winSize0.width*factor), myRound(winSize0.height*factor) };

		/* size of the image scaled down (from bigger to smaller) */
		MySize sz = { ( img->width/factor ), ( img->height/factor ) };

		/* difference between sizes of the scaled image and the original detection window */
		MySize sz1 = { sz.width - winSize0.width, sz.height - winSize0.height };

		/* if the actual scaled image is smaller than the original detection window, break */
		if( sz1.width < 0 || sz1.height < 0 )
			break;

		/* if a minSize different from the original detection window is specified, continue to the next scaling */
		if( winSize.width < minSize.width || winSize.height < minSize.height )
			continue;

		/*************************************
		 * Set the width and height of
		 * img1: normal image (unsigned char)
		 * sum1: integral image (int)
		 * sqsum1: squared integral image (int)
		 * see image.c for details
		 ************************************/
		setImage(sz.width, sz.height, img1);
		setSumImage(sz.width, sz.height, sum1);
		setSumImage(sz.width, sz.height, sqsum1);

		/***************************************
		 * Compute-intensive step:
		 * building image pyramid by downsampling
		 * downsampling using nearest neighbor
		 **************************************/
		nearestNeighbor(img, img1);

		/***************************************************
		 * Compute-intensive step:
		 * At each scale of the image pyramid,
		 * compute a new integral and squared integral image
		 ***************************************************/
		#ifndef PARALLEL_SCAN_TRANSPOSE
		integralImages(img1, sum1, sqsum1);
		#endif
		#ifdef PARALLEL_SCAN_TRANSPOSE
		cudaMemcpy( imgdata, img1->data,img1->height*img1->width*sizeof(unsigned char),cudaMemcpyHostToDevice);
		prescanArray(sum_GPU,squ_GPU,imgdata,img1->height,img1->width);
		#endif

		/* sets images for haar classifier cascade */
		/**************************************************
		 * Note:
		 * Summing pixels within a haar window is done by
		 * using four corners of the integral image:
		 * http://en.wikipedia.org/wiki/Summed_area_table
		 *
		 * This function loads the four corners,
		 * but does not do compuation based on four coners.
		 * The computation is done next in ScaleImage_Invoker
		 *************************************************/
		// setImageForCascadeClassifier( cascade, sum1, sqsum1);
		cascade->sum = *sum1;
		cascade->sqsum = *sqsum1;

		/* print out for each scale of the image pyramid */
		printf("detecting faces, iter := %d\n", iter_counter);

		/****************************************************
		 * Process the current scale with the cascaded fitler.
		 * The main computations are invoked by this function.
		 * Optimization oppurtunity:
		 * the same cascade filter is invoked each time
		 ***************************************************/
		scale_image_invoker(cascade, factor, sum1->height, sum1->width,
							allCandidates);

	} /* end of the factor loop, finish all scales in pyramid*/

	if (minNeighbors != 0) {
		groupRectangles(allCandidates, minNeighbors, GROUP_EPS);
	}

	freeImage(img1);
	freeSumImage(sum1);
	freeSumImage(sqsum1);
	return allCandidates;
}

void scale_image_invoker(
	myCascade *cascade, float factor, int sum_height,
	int sum_width, std::vector<MyRect> &face_vector_output
) {
	uint8_t *results_host;
	int *face_thread_IDs;
	// int tile_size = _cascade->orig_window_size->height;
	int n_windows_x = sum_width - 24 + 1;
	int n_windows_y = sum_height - 24 + 1;
	int n_blocks_x = n_windows_x / 32;
	int n_blocks_y = n_windows_y / 32;
	n_blocks_x = (n_windows_x % 32) ? n_blocks_x + 1 : n_blocks_x;
	n_blocks_y = (n_windows_y % 32) ? n_blocks_y + 1 : n_blocks_y;
	dim3 gridDims(n_blocks_x, n_blocks_y, 1);
	dim3 blockDims(32, 32, 1);
	int n_windows = n_windows_y * n_windows_x;
	int sum_area = sum_width * sum_height;
	// We only perform GPU allocation for the first scale; despite the fact that
	// subsequent scales in the pyramid will require less memory, it incurs
	// additional overhead to cudaMalloc and cudaFree for each scale. The excess
	// memory will merely go unaccessed for smaller pyramid iterations.
	if (factor == 1) {
		#ifndef PARALLEL_SCAN_TRANSPOSE
		cudaMalloc((void **) &sum_GPU, sum_area * sizeof(int));
		cudaMalloc((void **) &squ_GPU, sum_area * sizeof(int));
		#endif
		cudaMalloc((void **) &results, n_windows * sizeof(uint8_t));
	}
	// the results flags need to be reset between scales
	cudaMemset(results, 0, n_windows * sizeof(uint8_t));
	face_thread_IDs = (int *) calloc(n_windows, sizeof(int));
	results_host = (uint8_t *) calloc(n_windows, sizeof(uint8_t));
	#ifndef PARALLEL_SCAN_TRANSPOSE
	cudaMemcpy(sum_GPU, cascade->sum.data, sum_area * sizeof(int),
			   cudaMemcpyHostToDevice);
	cudaMemcpy(squ_GPU, cascade->sqsum.data, sum_area * sizeof(int),
			   cudaMemcpyHostToDevice);
	#endif
	printf("Scaling factor: %f; number of detection windows: %d.\n",
		   factor, n_windows);
	cascade_segment1_kernel<<<gridDims, blockDims>>>(
		sum_GPU, squ_GPU, results, stage_data_GPU, sum_width, sum_height
	);
	cudaDeviceSynchronize();
	cascade_segment2_kernel<<<gridDims, blockDims>>>(
		sum_GPU, squ_GPU, results, stage_data_GPU + 18 * SEG1_NODES,
		sum_width, sum_height
	);
	cudaDeviceSynchronize();
	cascade_segment3_kernel<<<gridDims, blockDims>>>(
		sum_GPU, squ_GPU, results,
		stage_data_GPU + 18 * (SEG1_NODES + SEG2_NODES), sum_width, sum_height
	);
	cudaDeviceSynchronize();
	cascade_segment4_kernel<<<gridDims, blockDims>>>(
		sum_GPU, squ_GPU, results,
		stage_data_GPU + 18 * (SEG1_NODES + SEG2_NODES + SEG3_NODES),
		sum_width, sum_height
	);
	cudaDeviceSynchronize();
	cascade_segment5_kernel<<<gridDims, blockDims>>>(
		sum_GPU, squ_GPU, results,
		stage_data_GPU + 18 * (SEG1_NODES + SEG2_NODES
							   + SEG3_NODES + SEG4_NODES),
		sum_width, sum_height
	);
	cudaDeviceSynchronize();
	cascade_segment6_kernel<<<gridDims, blockDims>>>(
		sum_GPU, squ_GPU, results,
		stage_data_GPU + 18 * (SEG1_NODES + SEG2_NODES + SEG3_NODES
							   + SEG4_NODES + SEG5_NODES),
		sum_width, sum_height
	);
	cudaDeviceSynchronize();
	int n_faces = 0;
	cudaMemcpy(results_host, results, n_windows * sizeof(uint8_t),
			   cudaMemcpyDeviceToHost);
	for (int i = 0; i < n_windows; i++) {
		if (!results_host[i]) { continue; }
		face_thread_IDs[n_faces++] = i;
	}
	printf("Number of GPU threads that claim to contain faces (post-seg 6): %d.\n", n_faces);
	// retrieve the remaining rectangles (which passed every stage)
	MySize winSizeOrig = cascade->orig_window_size;
	MySize winSizeScaled;

	winSizeScaled.width =  myRound(winSizeOrig.width*factor);
	winSizeScaled.height =  myRound(winSizeOrig.height*factor);

	n_faces = 0;
	cudaMemcpy(results_host, results, n_windows * sizeof(uint8_t),
			   cudaMemcpyDeviceToHost);
	for (int i = 0; i < n_windows; i++) {
		if (!results_host[i]) { continue; }
		face_thread_IDs[n_faces++] = i;
	}
	printf("Number of GPU threads that claim to contain faces: %d.\n", n_faces);
	std::vector<MyRect> *face_vector = &face_vector_output;
	for (int i = 0; i < n_faces; i++) {
		int window_start_y = face_thread_IDs[i] / cascade->sum.width;
		int window_start_x = face_thread_IDs[i] - cascade->sum.width
						   * window_start_y;
		// int window_start_x = window_start_y % cascade->sum.width;
		if (window_start_x < 0 || window_start_x > cascade->sum.width - 24
			|| window_start_y > cascade->sum.height - 24) {
			printf("edge rectangle detected (%d, %d)\n", window_start_x, window_start_y);
		} else {
			// printf("found a face beginning at (%d, %d).\n",
				   //myRound(window_start_x * factor), myRound(window_start_y * factor));
			MyRect r = {
				myRound(window_start_x * factor), myRound(window_start_y * factor),
				winSizeScaled.width, winSizeScaled.height
			};
			face_vector->push_back(r);
		}
	}
	free(face_thread_IDs);
	free(results_host);
}

/*****************************************************
 * Compute the integral image (and squared integral)
 * Integral image helps quickly sum up an area.
 * TODO: parallelize this
 ****************************************************/
void integralImages(MyImage *src, MyIntImage *sum, MyIntImage *sqsum) {
	int x, y, s, sq, t, tq;
	unsigned char it;
	int height = src->height;
	int width = src->width;
	unsigned char *data = src->data;
	int * sumData = sum->data;
	int * sqsumData = sqsum->data;
	for( y = 0; y < height; y++)
	{
		s = 0;
		sq = 0;
		/* loop over the number of columns */
		for( x = 0; x < width; x ++)
		{
			it = data[y*width+x];
			/* sum of the current row (integer)*/
			s += it;
			sq += it*it;

			t = s;
			tq = sq;
			if (y != 0)
			{
				t += sumData[(y-1)*width+x];
				tq += sqsumData[(y-1)*width+x];
			}
			sumData[y*width+x]=t;
			sqsumData[y*width+x]=tq;
		}
	}
}

/***********************************************************
 * This function downsample an image using nearest neighbor
 * It is used to build the image pyramid
 * TODO: parallelize this process
 **********************************************************/
void nearestNeighbor(MyImage *src, MyImage *dst) {
int y; int j;
	int x;
	int i;
	unsigned char* t;
	unsigned char* p;
	int w1 = src->width;
	int h1 = src->height;
	int w2 = dst->width;
	int h2 = dst->height;

	int rat = 0;

	unsigned char* src_data = src->data;
	unsigned char* dst_data = dst->data;


	int x_ratio = (int)((w1<<16)/w2) +1;
	int y_ratio = (int)((h1<<16)/h2) +1;

	for (i=0;i<h2;i++)
	{
		t = dst_data + i*w2;
		y = ((i*y_ratio)>>16);
		p = src_data + y*w1;
		rat = 0;
		for (j=0;j<w2;j++)
		{
			x = (rat>>16);
			*t++ = p[x];
			rat += x_ratio;
		}
	}
}

/* parameters:
* source
* scaled (NN output pre-integral)
* integral output
* squared integral output
 */
void nn_integral(MyImage *src, MyImage *dst, MyIntImage *sum, MyIntImage *squ) {
	// tile the dimensions of the output image
	if (dst->width > 2048) {
		fprintf(stderr, "Pyramid result was > 2,048 pixels in width.");
	}
	int blocksPerRow = dst->width / 1024;
	blocksPerRow = (dst->width % 1024) ? blocksPerRow + 1 : blocksPerRow;
	// blocks are restricted to <= 1,024 threads
	// for smaller output images (i.e., width < 1,024), we run one block per row
	dim3 blockDims((dst->width < 1024) ? dst->width : 1024, 1, 1);
	// grid dimensions are far less restrictive; up to 65,536 blocks
	dim3 gridDims(blocksPerRow, dst->height, 1);
	nearest_neighbor_row_scan_kernel<<<gridDims, blockDims>>>(
		src->width, src->height, dst->width, dst->height,
		src->data, dst->data, sum->data, squ->data
	);
	cudaDeviceSynchronize();
	// col_scan_kernel(dst, sum, squ);
}

/* This function is intended to replace readTextClassifier(); its data
 * structures are optimized for GPU processing. Instead of keeping separate
 * arrays for rectangle coordinates, weights, and filter child nodes, we can
 * keep the flattened structure from the class.txt layout, and read each value
 * as a float. While it makes more intuitive sense to store the coordinates of
 * the rectangles as integers, it ultimately vastly simplifies the process of
 * loading stages' data to the GPU if all attributes are contiguous in memory.
 */
void read_text_classifiers() {
	// number of stages of the cascade classifier
	int stages;
	// total number of weak classifiers (one node each)
	int total_nodes = 0;
	int i = 0, j, k;
	char fgets_buf[FGETS_BUF_SIZE];
	FILE *finfo = fopen("info.txt", "r");

	// There had better be 25 stages, or else kernels.cu will need an overhaul.
	if (fgets(fgets_buf, FGETS_BUF_SIZE, finfo) != NULL) {
		stages = atoi(fgets_buf);
	}

	// FIXME: pinned memory probably ideal for bigger arrays on host
	stages_array = (int *) malloc(sizeof(int) * stages);
	stages_thresh_array = (float*) malloc(sizeof(float) * stages);

	// The number of filters per stage is also taken into account in kernels.cu
	// when the cascade is segmented; optimal usage of shared memory would yield
	// around 680 filters per segment (very near the maximum of 48 K). However,
	// stages cannot be split when segmenting the cascade, so we err on the side
	// of underutilization; about 40 K of shared memory per block (570 filters).
	while (fgets(fgets_buf, FGETS_BUF_SIZE ,finfo) != NULL) {
		stages_array[i] = atoi(fgets_buf);
		total_nodes += stages_array[i];
		i++;
	}
	fclose(finfo);

	// huge, monolithic chunk of memory to hold essentially an exact copy
	// of class.txt; each thread in a given cascade segment kernel will be
	// simultaneously reading the same area within this array.
	stage_data = (float *) malloc(sizeof(float) * total_nodes * 18);
	int stage_data_index = 0;
	FILE *fp = fopen("class.txt", "r");

	/******************************************
	 * Read the filter parameters in class.txt
	 *
	 * Each stage of the cascaded filter has:
	 * 18 parameter per filter x tilter per stage
	 * + 1 threshold per stage
	 *
	 * For example, in 5kk73,
	 * the first stage has 9 filters,
	 * the first stage is specified using
	 * 18 * 9 + 1 = 163 parameters
	 * They are line 1 to 163 of class.txt
	 *
	 * The 18 parameters for each filter are:
	 * 1 to 4: coordinates of rectangle 1
	 * 5: weight of rectangle 1
	 * 6 to 9: coordinates of rectangle 2
	 * 10: weight of rectangle 2
	 * 11 to 14: coordinates of rectangle 3
	 * 15: weight of rectangle 3
	 * 16: threshold of the filter
	 * 17: alpha 1 of the filter
	 * 18: alpha 2 of the filter
	 ******************************************/

	for (i = 0; i < stages; i++) { // iterate over each stage in the cascade
		for (j = 0; j < stages_array[i]; j++) {	// iterate over each filter/tree
			for (k = 0; k < 3; k++) { // loop over each rectangle
				// add the 4 coordinates for each of the 3 rectangles
				if (fgets(fgets_buf, FGETS_BUF_SIZE , fp) != NULL) {
					stage_data[stage_data_index++] = strtof(fgets_buf, NULL);
				} else {
					break;
				}
				if (fgets(fgets_buf, FGETS_BUF_SIZE , fp) != NULL) {
					stage_data[stage_data_index++] = strtof(fgets_buf, NULL);
				} else {
					break;
				}
				// the third coordinate is the width; we store it as width + x0
				if (fgets(fgets_buf, FGETS_BUF_SIZE , fp) != NULL) {
					stage_data[stage_data_index] = strtof(fgets_buf, NULL)
						+ stage_data[stage_data_index - 2];
					stage_data_index++;
				} else {
					break;
				}
				// the third coordinate is the height, but stored as height + y0
				if (fgets(fgets_buf, FGETS_BUF_SIZE , fp) != NULL) {
					stage_data[stage_data_index] = strtof(fgets_buf, NULL)
						+ stage_data[stage_data_index - 2];
					stage_data_index++;
				} else {
					break;
				}
				if (fgets(fgets_buf, FGETS_BUF_SIZE , fp) != NULL) {
					// add the weights for the 3 rectangles
					// TODO: re-generate class.txt from OpenCV XML format,
					// preserving FP accuracy (existing class.txt divides by
					// 4096 and rounds to some fixed point format).
					stage_data[stage_data_index++] = strtof(fgets_buf, NULL) / 4096.0f;
				} else {
					break;
				}
			}
			if (fgets(fgets_buf, FGETS_BUF_SIZE, fp) != NULL) {
				// The same is true here (see above TODO), except the scaling
				// factor is 256, and there actually is loss of precision here
				// versus OpenCV, whereas for the weights are usually integers.
				stage_data[stage_data_index++] = strtof(fgets_buf, NULL);
			} else { break; }
			if (fgets(fgets_buf, FGETS_BUF_SIZE, fp) != NULL) {
				// add "alpha1" for this filter
				stage_data[stage_data_index++] = strtof(fgets_buf, NULL);
			} else { break; }
			if (fgets(fgets_buf, FGETS_BUF_SIZE, fp) != NULL) {
				// add "alpha2" for this filter
				stage_data[stage_data_index++] = strtof(fgets_buf, NULL);
			} else { break; }
		}
		// at the end of the data for each stage, parse its threshold
		if (fgets(fgets_buf, FGETS_BUF_SIZE, fp) != NULL) {
			stages_thresh_array[i] = strtof(fgets_buf, NULL);
		} else { break; }
	}
	fclose(fp);
	// The lengths of each stage are both universally required by any cascade
	// segment kernel, and comprise only 100 bytes of data. Therefore, they are
	// a good candidate for read-only constant memory on the GPU.
	cudaMemcpyToSymbol(stage_lengths, stages_array, 25 * sizeof(int));
	// The same is true for the thresholds (sizeof(float) == sizeof(int) == 4).
	cudaMemcpyToSymbol(stage_thresholds, stages_thresh_array, 25 * sizeof(float));

	cudaMalloc((void **) &stage_data_GPU, sizeof(float) * total_nodes * 18);
	cudaMemcpy(stage_data_GPU, stage_data, sizeof(float) * total_nodes * 18,
			   cudaMemcpyHostToDevice);
}

void free_text_classifiers() {
	free(stages_array);
	free(stage_data);
}

void free_GPU_pointers() {
	cudaFree(stage_data_GPU);
	cudaFree(results);
	cudaFree(sum_GPU);
	cudaFree(squ_GPU);
	#ifdef PARALLEL_SCAN_TRANSPOSE
	cudaFree(imgdata);
	#endif
}
