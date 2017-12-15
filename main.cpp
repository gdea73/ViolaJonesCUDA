/*
 *  TU Eindhoven
 *  Eindhoven, The Netherlands
 *
 *  Name            :   faceDetection.cpp
 *
 *  Author          :   Francesco Comaschi (f.comaschi@tue.nl)
 *
 *  Date            :   November 12, 2012
 *
 *  Function        :   Main function for face detection
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

#include <stdio.h>
#include <stdlib.h>
#include "image.h"
#include "stdio-wrapper.h"
#include "haar.h"

#include <time.h>
#include <sys/time.h>
struct timeval start_time, end_time;


using namespace std;


int main (int argc, char *argv[]) 
{

	double start_count , end_count;
        double elapsed_time;

	int flag;
	
	int mode = 1;
	int i;

	/* detection parameters */
	float scaleFactor = 1.2;
	int minNeighbours = 1;


	printf("-- entering main function --\r\n");

	printf("-- loading image --\r\n");

	MyImage imageObj;
	MyImage *image = &imageObj;

	if (argc != 3) {
		fprintf(stderr, "%s%s%s\n", "Usage: ", argv[0], " in.pgm out.pgm");
		return 1;
	}

	flag = readPgm(argv[1], image);
	if (flag == -1)
	{
		printf( "Unable to open input image\n");
		return 1;
	}

	printf("-- loading cascade classifier --\r\n");

	myCascade cascadeObj;
	myCascade *cascade = &cascadeObj;
	MySize minSize = {20, 20};
	MySize maxSize = {0, 0};

	/* classifier properties */
	cascade->n_stages=25;
	cascade->total_nodes=2913;
	cascade->orig_window_size.height = 24;
	cascade->orig_window_size.width = 24;


#ifdef USE_CUDA
	gettimeofday(&start_time,NULL);
	read_text_classifiers();
        gettimeofday(&end_time,NULL);
        start_count = (double) start_time.tv_sec + 1.e-6 * (double) start_time.tv_usec;
        end_count = (double) end_time.tv_sec + 1.e-6 * (double) end_time.tv_usec;
        elapsed_time = (end_count - start_count);
        printf("The total elapsed time for read_text_classifiers() is : %f seconds\n",elapsed_time);

#else
	readTextClassifier();
#endif
	

	std::vector<MyRect> result;

	printf("-- detecting faces --\r\n");

	gettimeofday(&start_time,NULL);
        result = detectObjects(image, minSize, maxSize, cascade, scaleFactor, minNeighbours);
        gettimeofday(&end_time,NULL);
        start_count = (double) start_time.tv_sec + 1.e-6 * (double) start_time.tv_usec;
        end_count = (double) end_time.tv_sec + 1.e-6 * (double) end_time.tv_usec;
        elapsed_time = (end_count - start_count);
        printf("The total elapsed time for detectObjects() is : %f seconds\n",elapsed_time);


	printf("-- detected %d faces --\r\n", result.size());

	for(i = 0; i < result.size(); i++ )
	{
		MyRect r = result[i];
		printf("Face number %d found at coordinates x: %d , y: %d - With width: %d , height: %d\n",i,r.x,r.y,r.width,r.height);
		drawRectangle(image, r);
	}

	printf("-- saving output --\r\n"); 
	flag = writePgm(argv[2], image); 

	printf("-- image saved --\r\n");

	/* delete image and free classifier */
	#ifdef USE_CUDA
	free_text_classifiers();
	free_GPU_pointers();
	#else
	releaseTextClassifier();
	#endif
	// FIXME: massive memory issues with the image itself
	// freeImage(image);

	return 0;
}
