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

// enables timers on detection, GPU init., &c
#define TIME_DETECTION
// enables image output with white rectangles around detected faces
#define DRAW_RECTANGLES

#ifdef TIME_DETECTION
	#include <time.h>
	#include <sys/time.h>
	#include "timers.h"
#endif

using namespace std;

int main (int argc, char *argv[]) {
#ifdef TIME_DETECTION
	startTimer(0);

	startTimer(4);
	init_GPU();
	stopTimer(4);
#endif

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

#ifdef DRAW_RECTANGLES
	if (argc != 3) {
		fprintf(stderr, "%s%s%s\n", "Usage: ", argv[0], " in.pgm out.pgm");
		return 1;
	}
#else
	if (argc != 2) {
		fprintf(stderr, "%s%s%s\n", "Usage: ", argv[0], " in.pgm");
		return 1;
	}
#endif

#ifdef TIME_DETECTION
	startTimer(2);
#endif
	flag = readPgm(argv[1], image);
	if (flag == -1) {
		printf( "Unable to open input image\n");
		return 1;
	}
#ifdef TIME_DETECTION
	stopTimer(2);
#endif

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

#ifdef TIME_DETECTION
	startTimer(3);
#endif
	read_text_classifiers();
#ifdef TIME_DETECTION
	stopTimer(3);
#endif

	std::vector<MyRect> result;

	printf("-- detecting faces --\r\n");

#ifdef TIME_DETECTION
	startTimer(1);
#endif
	result = detectObjects(image, minSize, maxSize, cascade, scaleFactor, minNeighbours);
#ifdef TIME_DETECTION
	stopTimer(1);
#endif

	printf("-- detected %d faces --\r\n", result.size());
	printf("-- printing rectangles --\r\n");
	printf("format: (x0, y0), (x2, y2)\r\n");

	for (i = 0; i < result.size(); i++) {
		MyRect r = result[i];
		// printf("Face number %d found at coordinates x: %d , y: %d - With width: %d , height: %d\n",i,r.x,r.y,r.width,r.height);
		printf("(%d, %d), (%d, %d)\r\n", r.x, r.y, r.x + r.width, r.y + r.height);
#ifdef DRAW_RECTANGLES
		drawRectangle(image, r);
#endif
	}

#ifdef DRAW_RECTANGES
	printf("-- saving output --\r\n"); 
	flag = writePgm(argv[2], image); 
	printf("-- image saved --\r\n");
#endif

	free_text_classifiers();
	free_GPU_pointers();
	freeImage(image);

#ifdef TIME_DETECTION
	stopTimer(0);
	printf("total time in seconds:\n%Lf\n", timediffs[0]);
	printf("detection time in seconds:\n%Lf\n", timediffs[1]);
	printf("image loading time in seconds:\n%Lf\n", timediffs[2]);
	printf("GPU initialization time:\n%Lf\n", timediffs[4]);
	printf("total time, minus GPU init:\n%Lf\n", timediffs[0] - timediffs[4]);
#endif

	return 0;
}
