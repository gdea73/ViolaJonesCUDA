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

/* include kernels */
#include "kernels.cu"

// this is duplicated in kernels.cu for the GPU Haar cascade
#define STAGE_THRESH_MULTIPLIER_CPU 0.85
// depending on our choice of precision, we may want > 12 chars when reading
// floating-point values in class.txt
#define FGETS_BUF_SIZE 12
#define MAX_FACES 500 

/* TODO: use matrices */
/* classifier parameters */
/************************************
 * Notes:
 * To paralleism the filter,
 * these monolithic arrays may
 * need to be splitted or duplicated
 ***********************************/
static int *stages_array;
static int *rectangles_array;
static float *weights_array;
static float *alpha1_array;
static float *alpha2_array;
static float *tree_thresh_array;
static float *stages_thresh_array;
static int **scaled_rectangles_array;

// the host (CPU) copy of the filters
float *stage_data;
// GPU DRAM pointers; stage data allocated in read_text_classifiers
float *stage_data_GPU;
// allocated in scale_image_invoker only for the first pyramid iteration
static int *remaining_candidates;
static uint8_t *results;
static int *sum_GPU, *squ_GPU;


int clock_counter = 0;
int n_features = 0;


int iter_counter = 0;
/* To warm up the gpu */
void init_gpu(){

	cudaFree(0);

}
/* compute integral images */
void integralImages( MyImage *src, MyIntImage *sum, MyIntImage *sqsum );

/* scale down the image */
void ScaleImage_Invoker( myCascade* _cascade, float _factor, int sum_row, int sum_col, std::vector<MyRect>& _vec);
void scale_image_invoker(
	myCascade *cascade, float factor, int sum_height,
	int sum_width, std::vector<MyRect> &face_vector_output
);

/* compute scaled image */
void nearestNeighbor (MyImage *src, MyImage *dst);

/* rounding function */
inline  int  myRound( float value )
{
	return (int)(value + (value >= 0 ? 0.5 : -0.5));
}

/*******************************************************
 * Function: detectObjects
 * Description: It calls all the major steps
 ******************************************************/

std::vector<MyRect> detectObjects( MyImage* _img, MySize minSize, MySize maxSize, myCascade* cascade,
		float scaleFactor, int minNeighbors)
{

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
		integralImages(img1, sum1, sqsum1);

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
		#ifdef USE_CUDA
		scale_image_invoker(cascade, factor, sum1->height, sum1->width,
							allCandidates);
		#else
		ScaleImage_Invoker(cascade, factor, sum1->height, sum1->width,
				allCandidates);
		#endif

	} /* end of the factor loop, finish all scales in pyramid*/

	if (minNeighbors != 0) {
		groupRectangles(allCandidates, minNeighbors, GROUP_EPS);
	}

	freeImage(img1);
	freeSumImage(sum1);
	freeSumImage(sqsum1);
	return allCandidates;
}

/***********************************************
 * Note:
 * The int_sqrt is softwar integer squre root.
 * GPU has hardware for floating squre root (sqrtf).
 * In GPU, it is wise to convert an int variable
 * into floating point, and use HW sqrtf function.
 * More info:
 * http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#standard-functions
 **********************************************/
/*****************************************************
 * The int_sqrt is only used in runCascadeClassifier
 * If you want to replace int_sqrt with HW sqrtf in GPU,
 * simple look into the runCascadeClassifier function.
 *****************************************************/
unsigned int int_sqrt (unsigned int value)
{
	int i;
	unsigned int a = 0, b = 0, c = 0;
	for (i=0; i < (32 >> 1); i++)
	{
		c<<= 2;
#define UPPERBITS(value) (value>>30)
		c += UPPERBITS(value);
#undef UPPERBITS
		value <<= 2;
		a <<= 1;
		b = (a<<1) | 1;
		if (c >= b)
		{
			c -= b;
			a++;
		}
	}
	return a;
}


void setImageForCascadeClassifier (myCascade* _cascade, MyIntImage* _sum, MyIntImage *_sqsum) {
	MyIntImage *sum = _sum;
	MyIntImage *sqsum = _sqsum;
	myCascade* cascade = _cascade;
	int i, j, k;
	MyRect equRect;
	int r_index = 0;
	int w_index = 0;
	MyRect tr;

	cascade->sum = *sum;
	cascade->sqsum = *sqsum;

	equRect.x = equRect.y = 0;
	equRect.width = cascade->orig_window_size.width;
	equRect.height = cascade->orig_window_size.height;

	cascade->inv_window_area = equRect.width*equRect.height;

	cascade->p0 = (sum->data) ;
	cascade->p1 = (sum->data +  equRect.width - 1) ;
	cascade->p2 = (sum->data + sum->width*(equRect.height - 1));
	cascade->p3 = (sum->data + sum->width*(equRect.height - 1) + equRect.width - 1);
	cascade->pq0 = (sqsum->data);
	cascade->pq1 = (sqsum->data +  equRect.width - 1) ;
	cascade->pq2 = (sqsum->data + sqsum->width*(equRect.height - 1));
	cascade->pq3 = (sqsum->data + sqsum->width*(equRect.height - 1) + equRect.width - 1);

	/****************************************
	 * Load the index of the four corners 
	 * of the filter rectangle
	 **************************************/

	/* loop over the number of stages */
	for( i = 0; i < cascade->n_stages; i++ )
	{
		/* loop over the number of haar features */
		for( j = 0; j < stages_array[i]; j++ )
		{
			int nr = 3;
			/* loop over the number of rectangles */
			for( k = 0; k < nr; k++ )
			{
				tr.x = rectangles_array[r_index + k*4];
				tr.width = rectangles_array[r_index + 2 + k*4];
				tr.y = rectangles_array[r_index + 1 + k*4];
				tr.height = rectangles_array[r_index + 3 + k*4];
				if (k < 2)
				{
					scaled_rectangles_array[r_index + k*4] = (sum->data + sum->width*(tr.y ) + (tr.x )) ;
					scaled_rectangles_array[r_index + k*4 + 1] = (sum->data + sum->width*(tr.y ) + (tr.x  + tr.width)) ;
					scaled_rectangles_array[r_index + k*4 + 2] = (sum->data + sum->width*(tr.y  + tr.height) + (tr.x ));
					scaled_rectangles_array[r_index + k*4 + 3] = (sum->data + sum->width*(tr.y  + tr.height) + (tr.x  + tr.width));
				}
				else
				{
					if ((tr.x == 0)&& (tr.y == 0) &&(tr.width == 0) &&(tr.height == 0))
					{
						scaled_rectangles_array[r_index + k*4] = NULL ;
						scaled_rectangles_array[r_index + k*4 + 1] = NULL ;
						scaled_rectangles_array[r_index + k*4 + 2] = NULL;
						scaled_rectangles_array[r_index + k*4 + 3] = NULL;
					}
					else
					{
						scaled_rectangles_array[r_index + k*4] = (sum->data + sum->width*(tr.y ) + (tr.x )) ;
						scaled_rectangles_array[r_index + k*4 + 1] = (sum->data + sum->width*(tr.y ) + (tr.x  + tr.width)) ;
						scaled_rectangles_array[r_index + k*4 + 2] = (sum->data + sum->width*(tr.y  + tr.height) + (tr.x ));
						scaled_rectangles_array[r_index + k*4 + 3] = (sum->data + sum->width*(tr.y  + tr.height) + (tr.x  + tr.width));
					}
				} /* end of branch if(k<2) */
			} /* end of k loop*/
			r_index+=12;
			w_index+=3;
		} /* end of j loop */
	} /* end i loop */
}


/****************************************************
 * evalWeakClassifier:
 * the actual computation of a haar filter.
 * More info:
 * http://en.wikipedia.org/wiki/Haar-like_features
 ***************************************************/
inline float evalWeakClassifier (float variance_norm_factor, int p_offset, int tree_index, int w_index, int r_index) {
	/* the node threshold is multiplied by the standard deviation of the image */
	float t = tree_thresh_array[tree_index] * variance_norm_factor;

	float sum = (*(scaled_rectangles_array[r_index] + p_offset)
			- *(scaled_rectangles_array[r_index + 1] + p_offset)
			- *(scaled_rectangles_array[r_index + 2] + p_offset)
			+ *(scaled_rectangles_array[r_index + 3] + p_offset))
		* weights_array[w_index];


	sum += (*(scaled_rectangles_array[r_index+4] + p_offset)
			- *(scaled_rectangles_array[r_index + 5] + p_offset)
			- *(scaled_rectangles_array[r_index + 6] + p_offset)
			+ *(scaled_rectangles_array[r_index + 7] + p_offset))
		* weights_array[w_index + 1];

	if ((scaled_rectangles_array[r_index+8] != NULL))
		sum += (*(scaled_rectangles_array[r_index+8] + p_offset)
				- *(scaled_rectangles_array[r_index + 9] + p_offset)
				- *(scaled_rectangles_array[r_index + 10] + p_offset)
				+ *(scaled_rectangles_array[r_index + 11] + p_offset))
			* weights_array[w_index + 2];

	if (sum >= t)
		return alpha2_array[tree_index];
	else
		return alpha1_array[tree_index];

}



int runCascadeClassifier (myCascade* _cascade, MyPoint pt, int start_stage) {
	int p_offset, pq_offset;
	int i, j;
	float mean;
	float variance_norm_factor;
	int haar_counter = 0;
	int w_index = 0;
	int r_index = 0;
	float stage_sum;
	myCascade* cascade;
	cascade = _cascade;

	p_offset = pt.y * (cascade->sum.width) + pt.x;
	pq_offset = pt.y * (cascade->sqsum.width) + pt.x;

	/**************************************************************************
	 * Image normalization
	 * mean is the mean of the pixels in the detection window
	 * cascade->pqi[pq_offset] are the squared pixel values (using the squared integral image)
	 * inv_window_area is 1 over the total number of pixels in the detection window
	 *************************************************************************/

	variance_norm_factor =  (cascade->pq0[pq_offset] - cascade->pq1[pq_offset] - cascade->pq2[pq_offset] + cascade->pq3[pq_offset]);
	mean = (cascade->p0[p_offset] - cascade->p1[p_offset] - cascade->p2[p_offset] + cascade->p3[p_offset]);

	variance_norm_factor = (variance_norm_factor*cascade->inv_window_area);
	variance_norm_factor =  variance_norm_factor - mean*mean;

	/***********************************************
	 * Note:
	 * The int_sqrt is softwar integer squre root.
	 * GPU has hardware for floating squre root (sqrtf).
	 * In GPU, it is wise to convert the variance norm
	 * into floating point, and use HW sqrtf function.
	 * More info:
	 * http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#standard-functions
	 **********************************************/
	if( variance_norm_factor > 0 )
		variance_norm_factor = sqrt(variance_norm_factor);
	else
		variance_norm_factor = 1;

	/**************************************************
	 * The major computation happens here.
	 * For each scale in the image pyramid,
	 * and for each shifted step of the filter,
	 * send the shifted window through cascade filter.
	 *
	 * Note:
	 *
	 * Stages in the cascade filter are independent.
	 * However, a face can be rejected by any stage.
	 * Running stages in parallel delays the rejection,
	 * which induces unnecessary computation.
	 *
	 * Filters in the same stage are also independent,
	 * except that filter results need to be merged,
	 * and compared with a per-stage threshold.
	 *************************************************/
	for (i = start_stage; i < cascade->n_stages; i++) {
		/****************************************************
		 * A shared variable that induces false dependency
		 * 
		 * To avoid it from limiting parallelism,
		 * we can duplicate it multiple times,
		 * e.g., using stage_sum_array[number_of_threads].
		 * Then threads only need to sync at the end
		 ***************************************************/
		stage_sum = 0;

		for (j = 0; j < stages_array[i]; j++) {
			/**************************************************
			 * Send the shifted window to a haar filter.
			 **************************************************/
			stage_sum += evalWeakClassifier(variance_norm_factor, p_offset, haar_counter, w_index, r_index);
			n_features++;
			haar_counter++;
			w_index+=3;
			r_index+=12;
		} /* end of j loop */

		/**************************************************************
		 * threshold of the stage. 
		 * If the sum is below the threshold, 
		 * no faces are detected, 
		 * and the search is abandoned at the i-th stage (-i).
		 * Otherwise, a face is detected (1)
		 **************************************************************/

		if (stage_sum < STAGE_THRESH_MULTIPLIER_CPU * stages_thresh_array[i]) {
			return -i;
		} /* end of the per-stage thresholding */
	} /* end of i loop */
	return 1;
}


void ScaleImage_Invoker (myCascade *_cascade, float _factor, int sum_row, int sum_col, std::vector<MyRect> &_vec) {
	myCascade* cascade = _cascade;

	float factor = _factor;
	MyPoint p;
	int result;
	int y1, y2, x2, x, y, step;
	std::vector<MyRect> *vec = &_vec;

	MySize winSize0 = cascade->orig_window_size;
	MySize winSize;

	winSize.width =  myRound(winSize0.width*factor);
	winSize.height =  myRound(winSize0.height*factor);
	y1 = 0;

	/********************************************
	 * When filter window shifts to image boarder,
	 * some margin need to be kept
	 *********************************************/
	y2 = sum_row - winSize0.height;
	x2 = sum_col - winSize0.width;

	/********************************************
	 * Step size of filter window shifting
	 * Reducing step makes program faster,
	 * but decreases quality of detection.
	 * example:
	 * step = factor > 2 ? 1 : 2;
	 * 
	 * For 5kk73, 
	 * the factor and step can be kept constant,
	 * unless you want to change input image.
	 *
	 * The step size is set to 1 for 5kk73,
	 * i.e., shift the filter window by 1 pixel.
	 *******************************************/	
	step = 1;

	/**********************************************
	 * Shift the filter window over the image.
	 * Each shift step is independent.
	 * Shared data structure may limit parallelism.
	 *
	 * Some random hints (may or may not work):
	 * Split or duplicate data structure.
	 * Merge functions/loops to increase locality
	 * Tiling to increase computation-to-memory ratio
	 *********************************************/
	for (x = 0; x <= x2; x += step) {
		for (y = y1; y <= y2; y += step) {
			p.x = x;
			p.y = y;

			/*********************************************
			 * Optimization Oppotunity:
			 * The same cascade filter is used each time
			 ********************************************/
			result = runCascadeClassifier( cascade, p, 0 );

			/*******************************************************
			 * If a face is detected,
			 * record the coordinates of the filter window
			 * the "push_back" function is from std:vec, more info:
			 * http://en.wikipedia.org/wiki/Sequence_container_(C++)
			 *
			 * Note that, if the filter runs on GPUs,
			 * the push_back operation is not possible on GPUs.
			 * The GPU may need to use a simpler data structure,
			 * e.g., an array, to store the coordinates of face,
			 * which can be later memcpy from GPU to CPU to do push_back
			 *******************************************************/
			if (result > 0) {
				MyRect r = {myRound(x*factor), myRound(y*factor), winSize.width, winSize.height};
				vec->push_back(r);
			}
		}
	}
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
		cudaMalloc((void **) &sum_GPU, sum_area * sizeof(int));
		cudaMalloc((void **) &squ_GPU, sum_area * sizeof(int));
		cudaMalloc((void **) &results, n_windows * sizeof(uint8_t));
	}
	face_thread_IDs = (int *) calloc(n_windows, sizeof(int));
	results_host = (uint8_t *) calloc(n_windows, sizeof(uint8_t));
	cudaMemcpy(sum_GPU, cascade->sum.data, sum_area * sizeof(int),
			   cudaMemcpyHostToDevice);
	cudaMemcpy(squ_GPU, cascade->sqsum.data, sum_area * sizeof(int),
			   cudaMemcpyHostToDevice);
	printf("Scaling factor: %f; number of detection windows: %d.\n",
		   factor, n_windows);
	cascade_segment1_kernel<<<gridDims, blockDims>>>(
		sum_GPU, squ_GPU, results, stage_data_GPU, sum_width, sum_height
	);
	cudaDeviceSynchronize();
	// For testing the first segment, it would be challenging to
	// return to processing the rest on the CPU (as the data structures
	// differ greatly). Instead, we can launch just as many threads for
	// each kernel, and accept the performance hit of many of those threads
	// terminating after failing the check set in results[] by the previous
	// stage.
	/* prototype of implementation for reducing number of threads for subsequent
	 * segments
	int n_remaining = 0;
	// cudamemcpy to host
	for (int i = 0; i < n_windows * results; i++) {
	// calc remaining on host
		remaining_candidates[n_remaining++]
	}
	// cudamemcpy back to device, pass that pointer to next kernel
	dim3 blockDims(1024, 1, 1);
	n_blocks_x = n_remaining / 1024;
	n_blocks_x = (n_remaining % 1024) ? n_blocks_x + 1 : n_blocks_x;
	*/
	int n_faces = 0;
	cudaMemcpy(results_host, results, n_windows * sizeof(uint8_t),
			   cudaMemcpyDeviceToHost);
	for (int i = 0; i < n_windows; i++) {
		if (!results_host[i]) { continue; }
		face_thread_IDs[n_faces++] = i;
	}
	printf("Number of GPU threads that claim to contain faces (post-seg 1): %d.\n", n_faces);
	cascade_segment2_kernel<<<gridDims, blockDims>>>(
		sum_GPU, squ_GPU, results, stage_data_GPU + 18 * 596, sum_width, sum_height
	);
	cudaDeviceSynchronize();
	n_faces = 0;
	cudaMemcpy(results_host, results, n_windows * sizeof(uint8_t),
			   cudaMemcpyDeviceToHost);
	for (int i = 0; i < n_windows; i++) {
		if (!results_host[i]) { continue; }
		face_thread_IDs[n_faces++] = i;
	}

	printf("Number of GPU threads that claim to contain faces (post-seg 2): %d.\n", n_faces);
	
	cascade_segment3_kernel<<<gridDims, blockDims>>>(
		sum_GPU, squ_GPU, results, stage_data_GPU + 18 * 1109, sum_width, sum_height
	);
	cudaDeviceSynchronize();
	n_faces = 0;
	cudaMemcpy(results_host, results, n_windows * sizeof(uint8_t),
			   cudaMemcpyDeviceToHost);
	for (int i = 0; i < n_windows; i++) {
		if (!results_host[i]) { continue; }
		face_thread_IDs[n_faces++] = i;
	}

	printf("Number of GPU threads that claim to contain faces (post-seg 3): %d.\n", n_faces);
	cascade_segment4_kernel<<<gridDims, blockDims>>>(
		sum_GPU, squ_GPU, results, stage_data_GPU + 18 * 1729, sum_width, sum_height
	);
	cudaDeviceSynchronize();
	n_faces = 0;
	cudaMemcpy(results_host, results, n_windows * sizeof(uint8_t),
			   cudaMemcpyDeviceToHost);
	for (int i = 0; i < n_windows; i++) {
		if (!results_host[i]) { continue; }
		face_thread_IDs[n_faces++] = i;
	}

	printf("Number of GPU threads that claim to contain faces (post-seg 4): %d.\n", n_faces);
	cascade_segment5_kernel<<<gridDims, blockDims>>>(
		sum_GPU, squ_GPU, results, stage_data_GPU + 18 * 2303, sum_width, sum_height
	);
	cudaDeviceSynchronize();
	n_faces = 0;
	cudaMemcpy(results_host, results, n_windows * sizeof(uint8_t),
			   cudaMemcpyDeviceToHost);
	for (int i = 0; i < n_windows; i++) {
		if (!results_host[i]) { continue; }
		face_thread_IDs[n_faces++] = i;
	}

	printf("Number of GPU threads that claim to contain faces (post-seg 5): %d.\n", n_faces);
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
		int window_start_y = (int) (face_thread_IDs[i] / cascade->sum.width) + 1;
		int window_start_x = face_thread_IDs[i] - cascade->sum.width * window_start_y;
		printf("found a \"face\" beginning at (%d, %d).\n",
			   window_start_x, window_start_y);
		MyRect r = {
			myRound(window_start_x * factor), myRound(window_start_y * factor),
			winSizeScaled.width, winSizeScaled.height
		};
		face_vector->push_back(r);
	}
	free(face_thread_IDs);
	free(results_host);
}

/*****************************************************
 * Compute the integral image (and squared integral)
 * Integral image helps quickly sum up an area.
 * More info:
 * http://en.wikipedia.org/wiki/Summed_area_table
 ****************************************************/
void integralImages (MyImage *src, MyIntImage *sum, MyIntImage *sqsum) {
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
 **********************************************************/
void nearestNeighbor (MyImage *src, MyImage *dst)
{

	int y;
	int j;
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
}

void readTextClassifier() {
	// number of stages of the cascade classifier
	int stages;
	/*total number of weak classifiers (one node each)*/
	int total_nodes = 0;
	int i, j, k, l;
	char mystring [12];
	int r_index = 0;
	int w_index = 0;
	int tree_index = 0;
	FILE *finfo = fopen("info.txt", "r");

	/**************************************************
	 * how many stages are in the cascaded filter? 
	 * the first line of info.txt is the number of stages 
	 * (in the 5kk73 example, there are 25 stages)
	 **************************************************/
	if ( fgets (mystring , 12 , finfo) != NULL )
	{
		stages = atoi(mystring);
	}
	i = 0;

	stages_array = (int *)malloc(sizeof(int)*stages);

	/**************************************************
	 * how many filters in each stage? 
	 * They are specified in info.txt,
	 * starting from second line.
	 * (in the 5kk73 example, from line 2 to line 26)
	 *************************************************/
	while ( fgets (mystring , 12 , finfo) != NULL )
	{
		stages_array[i] = atoi(mystring);
		total_nodes += stages_array[i];
		i++;
	}
	fclose(finfo);


	/* TODO: use matrices where appropriate */
	/***********************************************
	 * Allocate a lot of array structures
	 * Note that, to increase parallelism,
	 * some arrays need to be splitted or duplicated
	 **********************************************/
	rectangles_array = (int *)malloc(sizeof(int)*total_nodes*12);
	scaled_rectangles_array = (int **)malloc(sizeof(int*)*total_nodes*12);
	weights_array = (float *)malloc(sizeof(float)*total_nodes*3);
	alpha1_array = (float*)malloc(sizeof(float)*total_nodes);
	alpha2_array = (float*)malloc(sizeof(float)*total_nodes);
	tree_thresh_array = (float*)malloc(sizeof(float)*total_nodes);
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

	/* loop over n of stages */
	for (i = 0; i < stages; i++)
	{    /* loop over n of trees */
		for (j = 0; j < stages_array[i]; j++)
		{	/* loop over n of rectangular features */
			for(k = 0; k < 3; k++)
			{	/* loop over the n of vertices */
				for (l = 0; l <4; l++)
				{
					if (fgets (mystring , 12 , fp) != NULL)
						rectangles_array[r_index] = atoi(mystring);
					else
						break;
					r_index++;
				} /* end of l loop */
				if (fgets (mystring , 12 , fp) != NULL)
				{
					// weights_array[w_index] = atoi(mystring);
					weights_array[w_index] = atof(mystring) / 4096.0;
					/* Shift value to avoid overflow in the haar evaluation */
					/*TODO: make more general */
					/*weights_array[w_index]>>=8; */
				}
				else
					break;
				w_index++;
			} /* end of k loop */
			if (fgets (mystring , 12 , fp) != NULL)
				tree_thresh_array[tree_index]= atof(mystring) / 256.0;
			else
				break;
			if (fgets (mystring , 12 , fp) != NULL)
				alpha1_array[tree_index]= atof(mystring) / 256.0;
			else
				break;
			if (fgets (mystring , 12 , fp) != NULL)
				alpha2_array[tree_index]= atof(mystring) / 256.0;
			else
				break;
			tree_index++;
			if (j == stages_array[i]-1)
			{
				if (fgets (mystring , 12 , fp) != NULL)
					stages_thresh_array[i] = atof(mystring) / 256.0;
				else
					break;
			}
		} /* end of j loop */
	} /* end of i loop */
	fclose(fp);
}


void releaseTextClassifier()
{
	free(stages_array);
	free(rectangles_array);
	free(scaled_rectangles_array);
	free(weights_array);
	free(tree_thresh_array);
	free(alpha1_array);
	free(alpha2_array);
	free(stages_thresh_array);
}
/* End of file. */
