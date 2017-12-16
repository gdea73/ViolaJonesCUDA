/* Cascade segment breakdown:
 * Each segment has a corresponding kernel, in which each thread evaluates
 * one detection window. The set of detection windows passed to successive
 * kernels will decrease as windows are ruled out.
 * 
 * Memory usage based on 18 ints (or floats) per node == 72 B.
 * # / first stage / last stage / SHMEM usage / # nodes	/ words read per thread
 * 1 | 1		   | 7			| 17 K		  | 251		| 4.412
 * 2 | 8           | 11         | 21 K        | 345		| 6.064
 * 3 | 12          | 15         | 36 K        | 513		| 9.017
 * 4 | 16          | 19         | 44 K        | 620 	| 10.90
 * 5 | 20          | 22         | 40 K        | 574	 	| 10.09
 * 6 | 23          | 25         | 43 K        | 610	 	| 10.72
 */

#ifndef SEGMENTS_H
#define SEGMENTS_H

#define SEG1_NODES 251
#define SEG1_MIN_WORDS_PER_THREAD 251 * 18 / 1024
#define SEG2_NODES 345
#define SEG2_MIN_WORDS_PER_THREAD 345 * 18 / 1024
#define SEG3_NODES 513
#define SEG3_MIN_WORDS_PER_THREAD 513 * 18 / 1024
#define SEG4_NODES 620
#define SEG4_MIN_WORDS_PER_THREAD 620 * 18 / 1024
#define SEG5_NODES 574
#define SEG5_MIN_WORDS_PER_THREAD 574 * 18 / 1024
#define SEG6_NODES 610
#define SEG6_MIN_WORDS_PER_THREAD 610 * 18 / 1024

#endif
