#define N_TIMERS 10

struct timeval timevals[N_TIMERS];
long double timediffs[N_TIMERS];

void startTimer(int index);
void stopTimer(int index);

void startTimer(int index) {
	if (index >= N_TIMERS) {
		fprintf(stderr, "number of timers exceeded.\n");
	}
	gettimeofday(timevals + index, NULL);
}

void stopTimer(int index) {
	if (index >= N_TIMERS) {
		fprintf(stderr, "number of timers exceeded.\n");
	}
	struct timeval start_time = timevals[index];
	struct timeval end_time;
	gettimeofday(&end_time, NULL);
	long double start_s = (double) start_time.tv_sec + 1.e-6
						* (double) start_time.tv_usec;
	long double end_s = (double) end_time.tv_sec + 1.e-6
					  * (double) end_time.tv_usec;
	timediffs[index]  = (end_s - start_s);
}
