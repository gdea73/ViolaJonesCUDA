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

#define INPUT_FILENAME "Face.pgm"
#define OUTPUT_FILENAME "Output.pgm"
#include <time.h>
#include <sys/time.h>
#include "timers.h"

using namespace std;


int main (int argc, char *argv[]) 
{

	startTimer(0);

	startTimer(4);
	// sequential so don't init_GPU();
	stopTimer(4);
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

	startTimer(2);
	flag = readPgm(argv[1], image);
	if (flag == -1)
	{
		printf( "Unable to open input image\n");
		return 1;
	}
	stopTimer(2);

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


	startTimer(3);
	readTextClassifier();
	stopTimer(3);

	std::vector<MyRect> result;

	printf("-- detecting faces --\r\n");

	startTimer(1);
	result = detectObjects(image, minSize, maxSize, cascade, scaleFactor, minNeighbours);
	stopTimer(1);

	for(i = 0; i < result.size(); i++ )
	{
		MyRect r = result[i];
		drawRectangle(image, r);
	}

	printf("-- detected %d face(s) --\r\n", result.size()); 

	printf("-- saving output --\r\n"); 
	flag = writePgm(argv[2], image); 

	printf("-- image saved --\r\n");

	/* delete image and free classifier */
	releaseTextClassifier();
	freeImage(image);

	stopTimer(0);
	printf("total time in seconds:\n%Lf\n", timediffs[0]);
	printf("detection time in seconds:\n%Lf\n", timediffs[1]);
	printf("image loading time in seconds:\n%Lf\n", timediffs[2]);
	printf("GPU initialization time:\n%Lf\n", timediffs[4]);
	printf("total time, minus GPU init:\n%Lf\n", timediffs[0] - timediffs[4]);

	return 0;
}
