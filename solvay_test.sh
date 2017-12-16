#!/bin/bash
# Expects several versions of an idential image in ./img, preferably with
# a wide range of scales. In my testing, it seems that increasing image size
# is less detrimental to performance when scanning in parallel than on the CPU.
iterations=10
printf "Timing Viola-Jones CUDA with several image sizes.\n"
printf "Iterations per scale: $iterations.\n"
for scale in img/solvay*.pgm; do
	avg_total=0
	avg_img_load=0
	avg_gpu_init=0
	avg_det_time=0
	avg_total_less_gpu=0
	i=0
	while [ $i -lt $iterations ]; do
		times=()
		while IFS= read -r line; do
			times+=("$line")
		done < <(./vj "$scale" output.pgm | tail -n 10)
		# times=($(./vj "img/$scale" output.pgm | tail -n 10))
		avg_total=$(bc <<< "scale=4; $avg_total + ${times[1]}")
		avg_det_time=$(bc <<< "scale=4; $avg_det_time + ${times[3]}")
		avg_img_load=$(bc <<< "scale=4; $avg_img_load + ${times[5]}")
		avg_gpu_init=$(bc <<< "scale=4; $avg_gpu_init + ${times[7]}")
		avg_total_less_gpu=$(bc <<< "scale=4; $avg_total_less_gpu + ${times[9]}")
		i=$((i+1))
	done
	printf "Image file: $scale\n"
	printf "average total: $(bc <<< "scale=4;"`
		  `"$avg_total / $iterations") s\n"
	printf "average image load: $(bc <<< "scale=4;"`
		  `"$avg_img_load / $iterations") s\n"
	printf "average GPU init: $(bc <<< "scale=4;"`
		  `"$avg_gpu_init / $iterations") s\n"
	printf "average detection time: $(bc <<< "scale=4;"`
		  `"$avg_det_time / $iterations") s\n"
	printf "average total less GPU init: $(bc <<< "scale=4;"`
		  `"$avg_total_less_gpu / $iterations") s\n"
done
